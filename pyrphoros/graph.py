import abc
from collections import OrderedDict
import contextlib
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Tuple,
    overload,
    override,
)
import warnings

import torch
from torch import nn
from torch.utils.data import Dataset


class GraphError(Exception): ...


class ValueHandle: ...


class _AbstractNodeBuilder(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ValueHandle: ...


class _AbstractGraphBase(abc.ABC):
    @abc.abstractmethod
    def add_input(self, name: str, *args, **kwargs) -> ValueHandle:
        """Add an input to the graph."""
        ...

    @abc.abstractmethod
    def add_output(self, **kwargs: ValueHandle):
        """Add outputs to the graph."""
        ...

    @abc.abstractmethod
    def prefixed(self, prefix: str) -> "_AbstractGraphBase":
        """Get a graph object that accesses the root graph while applying a prefix to the used names."""
        ...

    @overload
    @abc.abstractmethod
    def add_node(self, __name: str, __node: Callable): ...

    @overload
    @abc.abstractmethod
    def add_node(self, __node: Callable): ...

    @abc.abstractmethod
    def add_node(self, *args) -> _AbstractNodeBuilder: ...

    @abc.abstractmethod
    @overload
    def slice(self, __stop): ...

    @abc.abstractmethod
    @overload
    def slice(self, __start, __stop): ...

    @abc.abstractmethod
    def slice(self, *args) -> "_AbstractGraphBase":
        """Return a subgraph that contains the specified inputs and outputs."""
        ...


class _GraphBase(_AbstractGraphBase):
    def __init__(self) -> None:
        self.inputs: Dict[str, ValueHandle] = {}
        self.outputs: Dict[str, ValueHandle] = {}

    # _AbstractGraphBase
    def add_input(self, name: str, *args, **kwargs):
        handle = ValueHandle(*args, **kwargs)
        self.inputs[name] = handle

        return handle

    def add_output(self, **kwargs: ValueHandle):
        self.outputs.update(**kwargs)


class _NodeBuilder(_AbstractNodeBuilder):
    def __init__(self, graph: "AddNodeMixin", node) -> None:
        self.graph = graph
        self.node = node

        # TODO: Record open task in graph to warn about forgotten calls

    def __call__(self, *args: Any, **kwargs: Any) -> ValueHandle:
        if self.graph is None:
            raise GraphError("Can not call a second time")

        graph = self.graph
        # Prevent a second call
        self.graph = None

        return graph._add_node(self.node, args, kwargs)


class AddNodeMixin:
    @abc.abstractmethod
    def _add_node(self, impl, args, kwargs) -> ValueHandle: ...

    def add_node(self, node: Callable) -> _NodeBuilder:
        return _NodeBuilder(self, node)


class _AbstractDataGraph(abc.ABC): ...


class _PrefixedDataGraph(_AbstractDataGraph):
    def __init__(self, parent: _AbstractDataGraph, prefix: str) -> None:
        super().__init__()


class DataGraph(AddNodeMixin, _GraphBase, _AbstractDataGraph):

    class _DataGraphDataset(Dataset):
        def __init__(self, graph: "DataGraph") -> None:
            super().__init__()

            self.graph = graph

        def __getitems__(self, indices: List) -> List[Mapping[str, Any]]:
            # TODO
            ...

        def __getitem__(self, index) -> Mapping[str, Any]:
            # TODO
            ...

    # _AbstractGraphBase
    def prefixed(self, prefix: str):
        return _PrefixedDataGraph(self, prefix)

    def slice(self, *args) -> "DataGraph": ...

    # AddNodeMixin
    def _add_node(self, node, args, kwargs) -> ValueHandle: ...

    @property
    def dataset(self):
        return self._DataGraphDataset(self)


class _AbstractComputeGraph(_AbstractGraphBase):
    @abc.abstractmethod
    def slice(self, *args) -> "_AbstractComputeGraph": ...

    @abc.abstractmethod
    @contextlib.contextmanager
    def parameter_group(self, name: str):
        """
        Context manager to add nodes to special parameter groups.
        TODO: Elaborate.
        """
        yield ...

    def add_backward_pass(self, loss: ValueHandle):
        """Register a backward pass to calculate the gradient of the supplied loss wrt. the model parameters."""
        ...


class _PrefixedModelGraph(_AbstractComputeGraph):
    def __init__(self, parent: _AbstractComputeGraph, prefix: str) -> None:
        super().__init__()

        self.parent = parent
        self.prefix = prefix

    # _AbstractGraphBase
    def add_input(self, name: str, *args, **kwargs):
        return self.parent.add_input(self.prefix + name, *args, **kwargs)

    def add_output(self, **kwargs: ValueHandle):
        self.parent.add_output(**{self.prefix + k: v for k, v in kwargs.items()})

    def add_node(self, node: Callable):
        return self.parent.add_node(node)

    def prefixed(self, prefix: str):
        return _PrefixedModelGraph(self.parent, self.prefix + prefix)

    def slice(self, *args):
        return _PrefixedModelGraph(self.parent.slice(*args), self.prefix)

    # _AbstractComputeGraph
    def parameter_group(self, name: str):
        return self.parent.parameter_group(self.prefix + name)

    def add_backward_pass(self, loss: ValueHandle):
        return self.parent.add_backward_pass(loss)


@dataclass
class ComputeNode:
    parameter_group: str
    impl: Callable
    args: Tuple
    kwargs: Mapping
    result_handle: ValueHandle


class _ParameterGroup:
    def __init__(self) -> None:
        self.backward_passes: List[ValueHandle] = []

    def add_backward_pass(self, loss: ValueHandle):
        self.backward_passes.append(loss)

    def parameters(self) -> nn.ParameterList:
        """Return an iterator over this parameter group's parameters."""
        return nn.ParameterList()


class _ComputeGraphModule(nn.Module):
    def __init__(self, graph: "ComputeGraph") -> None:
        super().__init__()

        self.graph = graph

        for name, parameter_group in graph.parameter_groups.items():
            setattr(self, name, parameter_group)

    @property
    def parameter_groups(self):
        return self.graph.parameter_groups

    def _read_inputs(
        self, batch: Mapping[str, torch.Tensor | Any]
    ) -> Dict[ValueHandle, torch.Tensor | Any]:
        return {v: batch[k] for k, v in self.graph.inputs.items()}

    def _extract_outputs(
        self, raw_batch: Mapping[ValueHandle, torch.Tensor | Any]
    ) -> Dict[str, torch.Tensor | Any]:
        return {k: raw_batch[v] for k, v in self.graph.outputs.items()}

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Compute one complete forward pass."""

        if self.training and len(self.graph.parameter_groups) > 1:
            warnings.warn(
                f"This module contains multiple parameter_groups. Don't use forward() for training but implement your own training step."
            )

        raw_batch = self._read_inputs(batch)
        for node in self.graph.nodes:
            args = tuple(
                raw_batch[a] if isinstance(a, ValueHandle) else a for a in node.args
            )
            kwargs = {
                n: raw_batch[a] if isinstance(a, ValueHandle) else a
                for n, a in node.kwargs.items()
            }
            result = node.impl(*args, **kwargs)
            raw_batch[node.result_handle] = result

        return self._extract_outputs(raw_batch)

    def _forward_incremental(
        self,
        raw_batch: Dict[ValueHandle, torch.Tensor | Any],
        to: ValueHandle,
    ) -> torch.Tensor | Any: ...


class ComputeGraph(AddNodeMixin, _GraphBase, _AbstractComputeGraph):
    """A graph for building a `torch.nn.Module`."""

    def __init__(self) -> None:
        super().__init__()

        self._current_parameter_group = "root"
        self.nodes: List[ComputeNode] = []
        self.parameter_groups: Dict[str, _ParameterGroup] = {"root": _ParameterGroup()}

    # AddNodeMixin
    def _add_node(self, impl, args, kwargs) -> ValueHandle:
        result_handle = ValueHandle()
        self.nodes.append(
            ComputeNode(
                self._current_parameter_group, impl, args, kwargs, result_handle
            )
        )
        return result_handle

    # _AbstractGraphBase
    def prefixed(self, prefix: str) -> "_PrefixedModelGraph":
        return _PrefixedModelGraph(self, prefix)

    def slice(self, *args) -> "ComputeGraph": ...

    # _AbstractComputeGraph
    @contextlib.contextmanager
    def parameter_group(self, name: str):
        if name in self.parameter_groups:
            raise ValueError(f"parameter_group '{name}' already exists")

        orig_parameter_group = self._current_parameter_group
        self._current_parameter_group = name
        self.parameter_groups[name] = _ParameterGroup()
        yield self
        assert self._current_parameter_group == name
        self._current_parameter_group = orig_parameter_group

    def add_backward_pass(self, loss: ValueHandle):
        """Register a backward pass to calculate the gradient of the supplied loss wrt. the model parameters."""
        self.parameter_groups[self._current_parameter_group].add_backward_pass(loss)

    def build_module(self):
        return _ComputeGraphModule(self)


def build_and_run_GAN():
    import torch
    from torch.optim import Adam, Optimizer  # type: ignore
    from torch.utils.data import DataLoader

    class NoiseSource(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

            self.args = args
            self.kwargs = kwargs

            self.fixed_noise = self.get_noise()

        def get_noise(self):
            return torch.rand(*self.args, **self.kwargs)

        def forward(self):
            if self.training:
                self.get_noise()

            return self.fixed_noise

    class LambdaModule(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    ###

    batch_size = 16
    latent_n_channels = 64
    target_n_channels = 128
    n_epochs = 100

    ###

    datagraph_train = DataGraph()

    datagraph_val = DataGraph()

    ###

    model_graph = ComputeGraph()
    real = model_graph.add_input("real")

    # Generator
    noise = model_graph.add_node(NoiseSource((batch_size, latent_n_channels)))()
    fake = model_graph.add_node(nn.Linear(latent_n_channels, target_n_channels))(noise)

    # Discriminator
    with model_graph.parameter_group("discriminator"):
        discriminator = nn.Sequential(nn.Linear(target_n_channels, 1), nn.Sigmoid())
        generator_fake_score = model_graph.add_node(discriminator)(fake)
        fake_detatched = model_graph.add_node(LambdaModule(lambda x: x.detach()))(fake)
        discriminator_fake_score = model_graph.add_node(discriminator)(fake_detatched)
        discriminator_real_score = model_graph.add_node(discriminator)(real)

        discriminator_fake_loss = model_graph.add_node(nn.BCELoss())(
            discriminator_fake_score, 0.0
        )
        discriminator_real_loss = model_graph.add_node(nn.BCELoss())(
            discriminator_real_score, 1.0
        )

        model_graph.add_output(
            discriminator_fake_loss=discriminator_fake_loss,
            discriminator_real_loss=discriminator_real_loss,
        )
        model_graph.add_backward_pass(discriminator_fake_loss)
        model_graph.add_backward_pass(discriminator_real_loss)

        generator_fake_loss = model_graph.add_node(nn.BCELoss())(
            generator_fake_score, 1.0
        )
        model_graph.add_output(generator_fake_loss=generator_fake_loss)
        model_graph.add_backward_pass(generator_fake_loss)

    ###

    # Todo: Prune datagraph_train/val
    datagraph_train = datagraph_train.slice(model_graph.outputs.keys())
    datagraph_val = datagraph_val.slice(model_graph.outputs.keys())

    dataloader_train = DataLoader(
        datagraph_train.dataset, batch_size=batch_size, shuffle=True
    )

    dataloader_val = DataLoader(
        datagraph_val.dataset, batch_size=batch_size, shuffle=False
    )

    model = model_graph.build_module()

    model.to("cuda:0")

    optimizers: Dict[str, Optimizer] = {
        name: Adam(group.parameters()) for name, group in model.parameter_groups.items()
    }

    for _ in range(n_epochs):
        model.train()

        batch: MutableMapping[str, Any]
        for batch in dataloader_train:
            raw_batch = model._read_inputs(batch)
            for name, group in model.parameter_groups.items():
                group.parameters().zero_grad()

                for backward_pass_loss_handle in group.backward_passes:
                    loss = model._forward_incremental(
                        raw_batch, backward_pass_loss_handle
                    )
                    loss.backward()

                optimizers[name].step()

        model.eval()
        for batch in dataloader_val:
            batch = model.forward(batch)
