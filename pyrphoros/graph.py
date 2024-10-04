import abc
import dataclasses
import functools
import inspect
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Self,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
import varname
from torch import nn
from torch.utils.data import Dataset


class GraphError(Exception): ...


@dataclasses.dataclass(repr=True)
class Reference:
    """
    A reference to a value within the graph, used to manage and track dependencies.
    """

    id: int = dataclasses.field(init=False, default_factory=lambda: uuid.uuid4().int)
    name: str
    owner: Optional["Node"] = None
    type: Optional[Any] = None
    shape: Union[Tuple[int, ...], None, Literal["auto"]] = None
    default: Optional[Any] = None

    @classmethod
    def create_like(
        cls,
        name: str,
        like: Optional["Reference"],
        **kwargs,
    ):
        if like is not None:
            defaults = {k: getattr(like, k) for k in ["type", "n_channels"]}
            defaults.update({k: v for k, v in kwargs.items() if v is not None})
        else:
            defaults = kwargs
        return cls(name, **defaults)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.id == other.id


def _get_traceback() -> List[inspect.Traceback]:
    frame = inspect.currentframe()
    assert frame is not None

    framelist = []
    while frame:
        # Skip frames in the current module or compiled from string
        if frame.f_code.co_filename not in (__file__, "<string>"):
            framelist.append(inspect.getframeinfo(frame))
        frame = frame.f_back
    return framelist


@overload
def _get_return_varname(
    *, frame: int = 1, multi: Literal[False] = False, default=None
) -> str: ...


@overload
def _get_return_varname(
    *, frame: int = 1, multi: Literal[True] = True, default=None
) -> Tuple[str, ...]: ...


def _get_return_varname(*, frame=1, multi=False, default=None):
    try:
        return varname.varname(frame, multi_vars=multi)  # type: ignore
    except varname.ImproperUseError:  # type: ignore
        if default is not None:
            return default

    raise ValueError("No default name provided")


@dataclasses.dataclass(repr=False)
class Node:
    """
    Represents a computational node in the graph, responsible for executing an operation with specified inputs.
    """

    component: "Component"
    name: str
    operation: Callable
    args: Tuple[Reference, ...]
    kwargs: Mapping[str, Reference]
    results: Tuple[Reference, ...]

    traceback: Optional[List[inspect.Traceback]] = dataclasses.field(
        default_factory=_get_traceback,
        init=False,
    )

    @property
    def qualifiedname(self):
        """Short representation of the node."""
        if self.component.prefix:
            return f"{self.component.prefix}.{self.name}"
        return self.name

    def _get_required_args_for(
        self, results: Container[Reference]
    ) -> Iterable[Reference]:
        """Return arguments required to compute the given results."""
        return self.args

    def bound_args(self) -> Iterator[Tuple[str, Reference]]:
        """Return iterator of (argname, handle) tuples."""
        raise NotImplementedError()

    def __str__(self):
        results = ", ".join(f"{r.name}" for r in self.results)
        args = ", ".join(f"{arg.name}" for arg in self.args)
        return f"{results} = {self.qualifiedname}({args})"

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class _NodeBuilder:
    """
    A NodeBuilder enables the following:
        Graph.add_node(name, callable)(arg, ...)
    """

    def __init__(
        self,
        graph: "Graph",
        component: "Component",
        name: str,
        operation: Callable,
        type: Optional[str],
        shape: Union[Tuple[int, ...], None, Literal["auto"]],
        like: Optional[Reference],
    ) -> None:
        self.graph = graph
        self.component = component
        self.name = name
        self.operation = operation
        self.type = type
        self.shape = shape
        self.like = like

    def __call__(self, *args: Any, **kwargs: Any) -> Reference:
        if self.graph is None:
            raise GraphError("Can not call a second time")

        graph = self.graph
        # Prevent a second call
        self.graph = None

        return graph._add_node(
            self.component,
            self.name,
            self.operation,
            self.type,
            self.shape,  # type: ignore
            self.like,
            args,
            kwargs,
        )


class _GraphInterface(abc.ABC):
    @abc.abstractmethod
    def add_input(self, name: str, *args, **kwargs) -> Reference:
        """Add an input to the graph."""
        ...

    @abc.abstractmethod
    def add_output(self, **kwargs: Reference):
        """Add outputs to the graph."""
        ...

    @abc.abstractmethod
    def create_component(self, name: str) -> "_GraphInterface":
        """Get a component object which also implements the _GraphInterface."""
        ...

    @abc.abstractmethod
    def add_node(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        shape: Union[Tuple[int, ...], None, Literal["auto"]] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        """
        Add a node to the graph.
        """
        ...


class Component(_GraphInterface):
    """
    A logical unit within the graph that groups nodes together, allowing for structured organization and independent parameter management.
    """

    def __init__(self, graph: "Graph", parent: "Self | None", name: str | None) -> None:
        super().__init__()

        self.graph = graph
        self.parent = parent
        self.name = name

    @functools.cached_property
    def prefix(self) -> str:
        return ".".join(n.name for n in self.ancestors if n.name)

    @functools.cached_property
    def ancestors(self) -> Tuple[Self, ...]:
        c = self
        ancestors = []
        while c is not None:
            ancestors.append(c)
            c = c.parent

        return tuple(ancestors)

    def add_input(self, name: str, *args, **kwargs) -> Reference:
        """Add an input to the graph."""

        return self.graph.add_input(f"{self.prefix}.{name}", *args, **kwargs)

    def add_output(self, **kwargs: Reference):
        """Add outputs to the graph."""
        return self.graph.add_output(
            **{f"{self.prefix}.{k}": v for k, v in kwargs.items()}
        )

    def create_component(self, name: str) -> "Component":
        """Get a component object which also implements the _GraphInterface."""
        return self.graph._add_component(self, name)

    def add_node(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        shape: Union[Tuple[int, ...], None, Literal["auto"]] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        """
        Add a node to the graph.
        """
        return _NodeBuilder(self.graph, self, name, operation, type, shape, like)

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        pass


TComponent = TypeVar("TComponent", bound=Component)
TNode = TypeVar("TNode", bound=Node)


class Graph(_GraphInterface, Generic[TComponent, TNode]):
    """
    Represents the entire computational graph, managing components and nodes, and supporting methods to construct the graph structure.
    """

    ComponentType: Type[TComponent]
    NodeType: Type[TNode]

    def __init__(self) -> None:
        self.inputs: Dict[str, Reference] = {}
        self.outputs: Dict[str, Reference] = {}
        self.nodes: List[TNode] = []

        self.root_component: TComponent = self.ComponentType(self, None, None)

    def add_input(self, name: str, *args, **kwargs):
        handle = Reference(name, None, *args, **kwargs)
        self.inputs[name] = handle

        return handle

    def add_output(self, **kwargs: Reference):
        self.outputs.update(**kwargs)

    def _add_component(self, parent: TComponent | None, name: str) -> TComponent:
        component = self.ComponentType(self, parent, name)
        return component

    def create_component(self, name: str) -> TComponent:
        return self._add_component(self.root_component, name)

    @abc.abstractmethod
    def _add_node(
        self,
        component: TComponent,
        name: str,
        operation: Callable,
        type: Optional[str],
        shape: Union[Tuple[int, ...], None, Literal["auto"]],
        like: Optional[Reference],
        args,
        kwargs,
    ) -> Reference: ...

    def add_node(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        shape: Union[Tuple[int, ...], None, Literal["auto"]] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        """
        Add a node to the graph.
        """
        return _NodeBuilder(
            self, self.root_component, name, operation, type, shape, like
        )

    @abc.abstractmethod
    @overload
    def slice(self, __stop): ...

    @abc.abstractmethod
    @overload
    def slice(self, __start, __stop): ...

    @abc.abstractmethod
    def slice(self, *args) -> "_GraphInterface":
        """Return a subgraph that contains the specified inputs and outputs."""
        ...


class DataGraph(Graph):

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


class _NNModuleGraphInterface(_GraphInterface):
    @abc.abstractmethod
    def add_loss(self, ref: Reference):
        """
        Add a loss for to the graph or component.

        Each individual loss triggers a separate backwards pass during training.
        """


class NNModuleComponent(Component, _NNModuleGraphInterface):
    """
    A specialized component for neural network modules, providing additional functionality to interface with PyTorch's module system.
    """

    def __init__(self, graph: Graph, parent: Self | None, name: str) -> None:
        super().__init__(graph, parent, name)

        self.losses: List[Reference] = []

    def add_loss(self, ref: Reference):
        self.losses.append(ref)


@dataclass
class NNModuleNode(Node):

    component: NNModuleComponent


class NNModuleComponentModule(nn.Module):
    """
    A PyTorch nn.Module that wraps a neural network component, enabling it to be integrated into the PyTorch computation flow.
    """

    def __init__(self, component: NNModuleComponent) -> None:
        super().__init__()

        self.component = component


class NNModuleGraphModule(NNModuleComponentModule):
    """
    A PyTorch nn.Module that executes the entire neural network graph.
    """

    def __init__(self, graph: "NNModuleGraph") -> None:
        super().__init__(graph.root_component)

        self.graph = graph
        self.component_modules: Dict[str, NNModuleComponentModule] = {}

        # Register the operation of a node for PyTorch
        for node in self.graph.nodes:
            if isinstance(node.operation, nn.Module):
                self._get_component_module(node.component).add_module(
                    node.name, node.operation
                )

    def _get_component_module(self, component: NNModuleComponent) -> nn.Module:
        """Get or create the module for a component."""
        try:
            return self.component_modules[component.prefix]
        except KeyError:
            pass

        if component == self.graph.root_component:
            self.component_modules[component.prefix] = self
            return self

        assert component.parent is not None

        parent_module = self._get_component_module(component.parent)

        component_module = NNModuleComponentModule(component)
        parent_module.add_module(component.name, component_module)

        self.component_modules[component.prefix] = component_module

        return component_module

    def _read_inputs(
        self, batch: Mapping[str, torch.Tensor | Any]
    ) -> Dict[Reference, torch.Tensor | Any]:
        return {ref: batch.get(k, ref.default) for k, ref in self.graph.inputs.items()}

    def _extract_outputs(
        self, raw_batch: Mapping[Reference, torch.Tensor | Any]
    ) -> Dict[str, torch.Tensor | Any]:
        return {k: raw_batch[v] for k, v in self.graph.outputs.items()}

    def _forward_raw(
        self,
        raw_batch: Dict[Reference, torch.Tensor | Any],
        nodes: List[NNModuleNode] | None = None,
    ) -> Dict[Reference, torch.Tensor | Any]:
        if nodes is None:
            nodes = self.graph.nodes

        for node in nodes:
            args = tuple(
                raw_batch[a] if isinstance(a, Reference) else a for a in node.args
            )
            kwargs = {
                n: raw_batch[a] if isinstance(a, Reference) else a
                for n, a in node.kwargs.items()
            }
            result = node.operation(*args, **kwargs)
            raw_batch[node.results[0]] = result

        return raw_batch

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Compute one complete forward pass."""

        raw_batch = self._read_inputs(batch)
        raw_batch = self._forward_raw(raw_batch)
        return self._extract_outputs(raw_batch)

    def _forward_incremental(
        self,
        raw_batch: Dict[Reference, torch.Tensor | Any],
        target: Reference,
    ) -> torch.Tensor | Any:
        """Given the current contents of raw_batch, execute all nodes required to calculate the target value."""

        try:
            return raw_batch[target]
        except KeyError:
            pass

        required_refs: Set[Reference] = {target}
        available_refs: Set[Reference] = set(raw_batch.keys())
        nodes: List[NNModuleNode] = []

        # Go backwards through the list of nodes
        for node in self.graph.nodes[::-1]:
            if not set(node.results).intersection(required_refs):
                continue

            nodes.insert(0, node)
            required_refs.update(
                set(node._get_required_args_for(required_refs)).difference(
                    available_refs
                )
            )

        self._forward_raw(raw_batch, nodes)

        return raw_batch[target]


class NNModuleGraph(Graph[NNModuleComponent, NNModuleNode], _NNModuleGraphInterface):
    """
    A specialized graph designed for building neural networks using PyTorch modules.
    """

    ComponentType = NNModuleComponent
    NodeType = NNModuleNode

    def __init__(self) -> None:
        super().__init__()

    def _add_node(
        self,
        component: NNModuleComponent,
        name: str,
        operation: Callable,
        type: Optional[str],
        shape: Union[Tuple[int, ...], None, Literal["auto"]],
        like: Optional[Reference],
        args,
        kwargs,
    ) -> Reference:
        result = Reference.create_like(
            _get_return_varname(frame=3), like, type=type, shape=shape
        )

        node = self.NodeType(component, name, operation, args, kwargs, (result,))
        self.nodes.append(node)
        result.owner = node

        return result

    def add_loss(self, ref: Reference):
        self.root_component.add_loss(ref)

    @overload
    def slice(self, __stop): ...

    @overload
    def slice(self, __start, __stop): ...

    def slice(self, *args) -> "_GraphInterface":
        """Return a subgraph that contains the specified inputs and outputs."""
        raise NotImplementedError()

    def build_module(self):
        return NNModuleGraphModule(self)
