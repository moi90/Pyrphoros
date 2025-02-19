import abc
import dataclasses
import functools
import inspect
import sys
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Self,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import varname
from torch import nn
from torch.utils.data import Dataset


class GraphError(Exception): ...


@dataclasses.dataclass()
class Reference:
    """
    A reference to a value within the graph, used to manage and track dependencies.
    """

    id: int = dataclasses.field(init=False, default_factory=lambda: uuid.uuid4().int)
    name: str
    owner: Optional["Node"] = None
    type: Optional[Any] = None
    n_channels: Union[int, None] = None
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

    def __repr__(self):
        return f"<Reference(name={self.name})>"


def _get_traceback() -> List[inspect.Traceback]:
    frame = sys._getframe(1)

    framelist = []
    while frame:
        # Skip frames in the current module or compiled from string
        if frame.f_code.co_filename not in (__file__, "<string>"):
            framelist.append(inspect.getframeinfo(frame))
        frame = frame.f_back
    return framelist


@overload
def _get_return_varname(
    *,
    frame: int = 1,
    multi: Literal[False] = False,
    default: Optional[str | Tuple[str, ...]] = None,
    prefix="",
) -> str: ...


@overload
def _get_return_varname(
    *,
    frame: int = 1,
    multi: Literal[True] = True,
    default: Optional[str | Tuple[str, ...]] = None,
    prefix="",
) -> Tuple[str, ...]: ...


def _get_return_varname(
    *,
    frame=1,
    multi=False,
    default: Optional[str | Tuple[str, ...]] = None,
    prefix="",
):
    try:
        name_s = varname.varname(frame, multi_vars=multi)  # type: ignore
    except varname.ImproperUseError:  # type: ignore
        if default is not None:
            return default

        raise ValueError("No default name(s) provided")

    if prefix:
        if multi:
            name_s = tuple(f"{prefix}.{n}" for n in name_s)
        else:
            name_s = f"{prefix}.{name_s}"

    return name_s


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
        return (a for a in self.args if isinstance(a, Reference))

    def bound_args(self) -> OrderedDict[str, Reference]:
        """
        Bind this node's args and kwargs to the operation's signature and return them as an ordered dictionary.

        This is used in `Graph.to_dot` to connect the node's arguments to the outputs of other nodes.

        Returns:
            OrderedDict[str, Reference]: An ordered dictionary of names and references.
        """
        bound_args = inspect.signature(self.operation).bind(*self.args, **self.kwargs)

        return bound_args.arguments

    def __str__(self):
        results = ", ".join(f"{r.name}" for r in self.results)
        args = ", ".join(f"{arg.name}" for arg in self.args)
        return f"{results} = {self.qualifiedname}({args})"

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class _NodeBuilder:
    """
    NodeBuilder captures node arguments and finally adds the node to the graph.

    It is a helper for `Graph.add_op` and `Component.add_op` to allow for a more fluent API:
        Graph.add_op(name, callable)(arg, ...)
    """

    def __init__(
        self,
        graph: "Graph",
        component: "Component",
        name: str,
        operation: Callable,
        type: Optional[str],
        n_channels: Union[int, None, Literal["auto"]],
        like: Optional[Reference],
    ) -> None:
        self.graph = graph
        self.component = component
        self.name = name
        self.operation = operation
        self.type = type
        self.n_channels = n_channels
        self.like = like

    def __call__(self, *args: Any, **kwargs: Any) -> Reference:
        if self.graph is None:
            raise GraphError("Can not call a second time")

        graph = self.graph
        # Prevent a second call
        self.graph = None

        return graph._add_op(
            self.component,
            self.name,
            self.operation,
            self.type,
            self.n_channels,  # type: ignore
            self.like,
            args,
            kwargs,
        )


class _GraphInterface(abc.ABC):
    @abc.abstractmethod
    def add_input(self, name: str, *args, **kwargs) -> Reference:
        """
        Add an input to the graph with the given name.

        Args:
            name (str): The name of the input to be added.
            *args, **kwargs: See `Reference`.
        """

    @abc.abstractmethod
    def add_output(self, *args: Reference, **kwargs: Reference):
        """
        Add outputs to the graph.

        Args:
            *args (Reference): Positional arguments, each a `Reference` object.
            **kwargs (Reference): Keyword arguments, each a `Reference` object.

        Example:
            .. code-block:: python

                graph = Graph()
                one = graph.add_op("foo", lambda: 1)()
                two = graph.add_op("bar", lambda: 2)()
                graph.add_output(one, two=two)
        """

    @abc.abstractmethod
    def add_component(self, name: str) -> "_GraphInterface":
        """
        Add a component to the graph.

        Example:
            .. code-block:: python

                graph = Graph()
                with graph.add_component("foo") as foo:
                    one = foo.add_op("one", lambda: 1)()
        """

    @abc.abstractmethod
    def add_op(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        n_channels: Union[int, None, Literal["auto"]] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        """
        Add an operation to the graph.

        Example:
            .. code-block:: python

                graph = Graph()
                one = graph.add_op("foo", lambda: 1)()
        """


T = TypeVar("T")
TComponent = TypeVar("TComponent", bound="Component")
TNode = TypeVar("TNode", bound="Node")
TGraph = TypeVar("TGraph", bound="Graph")


class Component(_GraphInterface, Generic[TGraph]):
    """
    A logical unit within the graph that groups nodes together, allowing for structured organization and independent parameter management.

    Components can be nested to any depth, forming a tree structure.

    Components can be used as context managers to create a scoped environment for adding nodes.
    (However, this is not required and purely cosmetic.)
    """

    def __init__(self, graph: TGraph, parent: "Self | None", name: str) -> None:
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

    def add_output(self, *args: Reference, **kwargs: Reference):
        """Add outputs to the graph."""
        return self.graph.add_output(
            *args, **{f"{self.prefix}.{k}": v for k, v in kwargs.items()}
        )

    def add_component(self, name: str) -> "Component[TGraph]":
        return self.graph._add_component(self, name)

    def add_op(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        n_channels: Union[int, None] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        return _NodeBuilder(self.graph, self, name, operation, type, n_channels, like)

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        pass


def _parse_slice_args(*args: T) -> Tuple[T | None, T]:
    try:
        (outputs,) = args
        return None, outputs
    except ValueError:
        pass

    try:
        inputs, outputs = args
        return inputs, outputs
    except ValueError:
        pass

    raise ValueError(f"Unexpected number of arguments: {len(args)}")


def _guess_n_channels(obj, default: Optional[int] = None) -> int:
    """Determine the number of output n_channels of a module."""

    if isinstance(obj, nn.Module):
        try:
            # Linear
            return obj.out_features  # type: ignore
        except AttributeError:
            pass

        try:
            # ConvNd
            return obj.out_channels  # type: ignore
        except AttributeError:
            pass

        # Sequential and other nn.Module's
        children = list(obj.children())

        # Iterate in reverse order to get the last child where n_channels can be guessed.
        # This skips ReLU, and other Modules that don't change the number of channels.
        for child in reversed(children):
            try:
                return _guess_n_channels(child)
            except ValueError:
                pass

    if default is not None:
        return default

    raise ValueError(f"Could not determine n_channels for {obj}")


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

        self.root_component: TComponent = self.ComponentType(self, None, "")

    def add_input(self, name: str, *args, **kwargs) -> Reference:
        handle = Reference(name, None, *args, **kwargs)
        self.inputs[name] = handle

        return handle

    def add_output(self, *args: Reference, **kwargs: Reference):
        self.outputs.update({r.name: r for r in args})
        self.outputs.update(**kwargs)

    def _add_component(self, parent: TComponent | None, name: str) -> TComponent:
        component = self.ComponentType(self, parent, name)
        return component

    def add_component(self, name: str) -> TComponent:
        return self._add_component(self.root_component, name)

    def _add_op(
        self,
        component: TComponent,
        name: str,
        operation: Callable,
        type: Optional[str],
        n_channels: Union[int, None, Literal["auto"]],
        like: Optional[Reference],
        args,
        kwargs,
    ) -> Reference:
        if n_channels == "auto":
            n_channels = _guess_n_channels(operation)

        result = Reference.create_like(
            _get_return_varname(frame=3, prefix=component.prefix),
            like,
            type=type,
            n_channels=n_channels,
        )

        for obj in args:
            if isinstance(obj, _NodeBuilder):
                raise ValueError("Got _NodeBuilder")

        for obj in kwargs.values():
            if isinstance(obj, _NodeBuilder):
                raise ValueError("Got _NodeBuilder")

        node = self.NodeType(component, name, operation, args, kwargs, (result,))
        self.nodes.append(node)
        result.owner = node

        return result

    def add_op(
        self,
        name: str,
        operation: Callable,
        *,
        type: Optional[str] = None,
        n_channels: Union[int, None, Literal["auto"]] = None,
        like: Optional[Reference] = None,
    ) -> _NodeBuilder:
        """
        Add a node to the graph.
        """

        if not isinstance(name, str):
            raise ValueError("name has to be of str type")

        return _NodeBuilder(
            self, self.root_component, name, operation, type, n_channels, like
        )

    def _slice_nodes(
        self, given: Set[Reference], required: Set[Reference]
    ) -> Tuple[List[TNode], Set[Reference]]:
        if given.issuperset(required):
            return [], set()

        nodes: List[TNode] = []

        for node in self.nodes[::-1]:
            provided_by_node = set(node.results)

            # Check if node produces a required result
            if not provided_by_node.intersection(required):
                continue

            # Prepend node to the list of nodes
            nodes.insert(0, node)

            given = given.union(provided_by_node)

            # Update required references: remove provided, add required minus given
            required = required.union(
                set(node._get_required_args_for(required))
            ).difference(given)

        return nodes, required

    @overload
    def slice(
        self,
        __outputs: str | Reference | Sequence[str | Reference],
        /,
    ) -> Self: ...

    @overload
    def slice(
        self,
        __inputs: str | Reference | Sequence[str | Reference],
        __outputs: str | Reference | Sequence[str | Reference],
        /,
    ) -> Self: ...

    def slice(self, *args: str | Reference | Sequence[str | Reference]) -> Self:
        """Return a subgraph that produces the specified outputs."""

        inputs, outputs = _parse_slice_args(*args)

        if isinstance(outputs, (str, Reference)):
            outputs = (outputs,)

        graph = type(self)()

        if inputs is None:
            _inputs = None
        else:
            if isinstance(inputs, (str, Reference)):
                inputs = (inputs,)

            _inputs = {}
            for i in inputs:
                if isinstance(i, str):
                    _inputs[i] = self.inputs[i]
                elif isinstance(i, Reference):
                    _inputs[i.name] = i
                else:
                    raise ValueError(f"Unknown output reference: {i!r}")

        for o in outputs:
            if isinstance(o, str):
                graph.outputs[o] = self.outputs[o]
            elif isinstance(o, Reference):
                graph.add_output(o)
            else:
                raise ValueError(f"Unknown output reference: {o!r}")

        input_set = set(_inputs.values()) if _inputs is not None else set()
        graph.nodes, required_refs = self._slice_nodes(
            input_set,
            set(graph.outputs.values()),
        )

        if _inputs is None:
            graph.inputs = {k: v for k, v in self.inputs.items() if v in required_refs}
        else:
            # Make sure that required_refs is a subset of _inputs.values()
            if not required_refs.issubset(input_set):
                missing_names = ", ".join(
                    sorted((r.name for r in required_refs.difference(input_set)))
                )
                raise ValueError(
                    f"Sliced subgraph requires more inputs than supplied: {missing_names}"
                )

            graph.inputs = _inputs

        graph.root_component = self.root_component

        return graph

    def _read_inputs_batched(self, batch: Mapping[str, Any]) -> Dict[Reference, Any]:
        return {ref: batch.get(k, ref.default) for k, ref in self.inputs.items()}

    def _extract_outputs_batched(
        self, raw_batch: Mapping[Reference, Any]
    ) -> Dict[str, Any]:
        return {k: raw_batch[v] for k, v in self.outputs.items()}

    def _execute_raw(
        self,
        raw_batch: Dict[Reference, Any],
        nodes: List[TNode] | None = None,
    ) -> Dict[Reference, Any]:
        if nodes is None:
            nodes = self.nodes

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

    def __call__(self, *batch_or_args) -> Dict[str, Any]:
        """Run all graph operations and return ."""

        if len(batch_or_args) == 1 and isinstance(batch_or_args[0], Mapping):
            batch = batch_or_args[0]
        else:
            batch = {k: v for k, v in zip(self.inputs.keys(), batch_or_args)}

        missing_arguments = set(self.inputs.keys()).difference(set(batch.keys()))
        if missing_arguments:
            raise ValueError(
                f"Some arguments are missing: {', '.join(sorted(missing_arguments))}"
            )

        raw_batch = self._read_inputs_batched(batch)
        raw_batch = self._execute_raw(raw_batch)
        return self._extract_outputs_batched(raw_batch)

    def _execute_incremental(
        self,
        raw_batch: Dict[Reference, Any],
        target: Reference,
    ) -> Any:
        """Given the current contents of raw_batch, execute all nodes required to calculate the target value."""

        try:
            return raw_batch[target]
        except KeyError:
            pass

        nodes, required_refs = self._slice_nodes(set(raw_batch.keys()), {target})

        if required_refs:
            raise ValueError(
                f"raw_batch does not provide all required refs: {[(r.owner.qualifiedname, r.name) for r in required_refs]}"
            )

        self._execute_raw(raw_batch, nodes)

        return raw_batch[target]

    def to_dot(self):
        import graphviz

        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")

        known_handles = {}

        # Inputs
        dot.attr("node", shape="ellipse")
        for i, (k, ref) in enumerate(self.inputs.items()):
            label = k
            dot.node(f"i{i}", label)

            known_handles[ref.id] = f"i{i}"

        # Nodes
        dot.attr("node", shape="record")
        for i, node in enumerate(self.nodes):
            parts = []
            if len(node.args) > 1:
                # Use named arguments if multiple arguments are given
                for argname, arg in node.bound_args().items():
                    # Don't show input edges which are not required
                    try:
                        from_ = known_handles[arg.id]
                    except KeyError:
                        continue

                    dot.edge(from_, f"n{i}:{argname}")
                parts.append(
                    "{"
                    + " | ".join(
                        f"<{argname}> {argname}"
                        for argname, _ in node.bound_args().items()
                    )
                    + "}"
                )
            else:
                # Otherwise use no ports
                for arg in node.args:
                    dot.edge(known_handles[arg.id], f"n{i}:w")

            parts.append(node.name)

            if len(node.results) > 1:
                for j, arg in enumerate(node.results):
                    known_handles[arg.id] = f"n{i}:o{j}"

                parts.append(
                    "{"
                    + " | ".join(f"<o{j}> {v.name}" for j, v in enumerate(node.results))
                    + "}"
                )
            else:
                for j, arg in enumerate(node.results):
                    known_handles[arg.id] = f"n{i}:o"

            label = "{" + " | ".join(parts) + "}"
            dot.node(f"n{i}", label)

        # Outputs
        dot.attr("node", shape="ellipse")
        for i, (k, ref) in enumerate(self.outputs.items()):
            dot.node(f"o{i}", k)
            dot.edge(known_handles[ref.id], f"o{i}")

        return dot


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

    @property
    @abc.abstractmethod
    def losses(self) -> Iterable[Reference]:
        """Return an iterable of losses in the graph."""


class NNModuleComponent(Component["NNModuleGraph"], _NNModuleGraphInterface):
    """
    A specialized component for neural network modules, providing additional functionality to interface with PyTorch's module system.
    """

    def add_loss(self, ref: Reference):
        self.graph._add_loss(self, ref)

    @property
    def losses(self) -> Iterable[Reference]:
        return self.graph._get_losses(self)


@dataclass
class NNModuleNode(Node):

    component: NNModuleComponent

    def bound_args(self) -> OrderedDict[str, Reference]:
        meth = (
            self.operation.forward
            if isinstance(self.operation, nn.Module)
            else self.operation
        )
        bound_args = inspect.signature(meth).bind(*self.args, **self.kwargs)

        return bound_args.arguments


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

    def forward(self, *args):
        return self.graph(*args)


class NNModuleGraph(Graph[NNModuleComponent, NNModuleNode], _NNModuleGraphInterface):
    """
    A specialized graph designed for building neural networks using PyTorch modules.
    """

    ComponentType = NNModuleComponent
    NodeType = NNModuleNode

    def __init__(self) -> None:
        super().__init__()

        self._component_losses = []

    def _add_loss(self, component: NNModuleComponent, ref: Reference):
        self._component_losses.append((component, ref))

    def add_loss(self, ref: Reference):
        self.root_component.add_loss(ref)

    def _get_losses(self, component: NNModuleComponent) -> Iterable[Reference]:
        return (ref for comp, ref in self._component_losses if comp == component)

    @property
    def losses(self) -> Iterable[Reference]:
        return self._get_losses(self.root_component)

    def build_module(self):
        return NNModuleGraphModule(self)
