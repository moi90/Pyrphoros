from functools import partial
import torch
from pyrphoros.graph import NNModuleGraph
from torch import nn
from torch.testing import assert_close


def test_NNModuleGraph_empty_forward():
    g = NNModuleGraph()
    m = g.build_module()
    m()


def test_NNModuleGraph_forward():
    g = NNModuleGraph()
    inp = g.add_input("input")
    outp = g.add_op(
        "linear",
        nn.Linear(10, 10),
    )(inp)
    g.add_output(outp=outp)

    m = g.build_module()

    result = m(torch.zeros((4, 10)))

    assert "outp" in result

    assert result["outp"].size() == (4, 10)


def test_NNModuleGraph_components():
    """
    Test the NNModuleGraph components and their interactions.

    This test verifies the following:
    - Creation of an NNModuleGraph and adding inputs and operations.
    - Creation of components within the graph and adding operations and outputs to them.
    - Ensuring that components have the correct ancestors.
    - Building the NNModule from the graph and accessing its submodules.
    - Training the NNModule and ensuring that gradients are only updated within the respective components.
    """
    batch_size = 4

    graph = NNModuleGraph()
    x = graph.add_input("input")

    assert x.name == "input"

    x = graph.add_op("linear", nn.Linear(10, 10))(x)
    assert x.name == "x"

    with graph.add_component("a") as component_a:
        dx = component_a.add_op("detach_x", lambda x: x.detach())(x)
        assert dx.name == "a.dx"

        outp_a = component_a.add_op("linear", nn.Linear(10, 1))(dx)
        component_a.add_output(outp_a=outp_a)
        loss = component_a.add_op("loss", nn.BCEWithLogitsLoss())(
            outp_a, torch.ones(batch_size, 1)
        )
        component_a.add_loss(loss)

    assert graph.root_component in component_a.ancestors

    with graph.add_component("b") as component_b:
        dx = component_b.add_op("detach_x", lambda x: x.detach())(x)
        outp_b = component_b.add_op("linear", nn.Linear(10, 1))(dx)
        component_b.add_output(outp_b=outp_b)
        loss = component_b.add_op("loss", nn.BCEWithLogitsLoss())(
            outp_b, torch.ones(batch_size, 1)
        )
        component_b.add_loss(loss)

    assert graph.root_component in component_b.ancestors

    nn_module = graph.build_module()

    nn_module.linear
    nn_module.a.linear
    nn_module.b.linear

    # Train
    batch = {"input": torch.zeros((batch_size, 10))}
    raw_batch = nn_module.graph._read_inputs_batched(batch)

    for current_name, current_component_module in nn_module.component_modules.items():
        # Zero all model gradients to be able to detect gradients "leaking" into other components.
        # (Normally, only the respective component should be zeroed: cm.parameters().zero_grad())
        nn_module.zero_grad()

        for loss_ref in current_component_module.component.losses:
            loss = nn_module.graph._execute_incremental(raw_batch, loss_ref)
            loss.backward()

        # Ensure that only gradients in the respective component were updated
        for other_name, other_component_module in nn_module.component_modules.items():
            if (
                other_component_module.component
                in current_component_module.component.ancestors
            ):
                continue

            gradients = [
                torch.linalg.norm(p.grad)
                for p in other_component_module.parameters()
                if p.grad is not None
            ]
            mean_gradient = torch.mean(torch.stack(gradients)) if gradients else 0

            if other_component_module is current_component_module:
                assert mean_gradient >= 0
            else:
                assert (
                    mean_gradient == 0
                ), f"Component {other_name!r} has a non-zero gradient when processing component {current_name!r} "


def test_NNModuleGraph_slice():
    batch_size = 4

    graph = NNModuleGraph()
    x = graph.add_input("input")

    x = graph.add_op("linear", nn.Linear(10, 20))(x)

    with graph.add_component("a") as component_a:
        dx = component_a.add_op("detach_x", lambda x: x.detach())(x)
        outp_a = component_a.add_op("linear", nn.Linear(20, 1))(dx)
        loss = component_a.add_op("loss", nn.BCEWithLogitsLoss())(
            outp_a, torch.ones(batch_size, 1)
        )
        component_a.add_output(output=outp_a, loss=loss)
        component_a.add_loss(loss)

    with graph.add_component("b") as component_b:
        dx = component_b.add_op("detach_x", lambda x: x.detach())(x)
        outp_b = component_b.add_op("linear", nn.Linear(20, 1))(dx)
        loss = component_b.add_op("loss", nn.BCEWithLogitsLoss())(
            outp_b, torch.ones(batch_size, 1)
        )
        component_b.add_output(output=outp_b, loss=loss)
        component_b.add_loss(loss)

    # Slice towards one named output
    subgraph = graph.slice("b.loss")
    nn_module = subgraph.build_module()
    nn_module(torch.zeros((batch_size, 10)))

    # Slice from input ref to output ref
    subgraph = graph.slice(x, outp_a)
    assert subgraph.inputs == {"x": x}
    assert subgraph.outputs == {"a.outp_a": outp_a}
    nn_module = subgraph.build_module()
    nn_module(torch.zeros((batch_size, 20)))


def test_GAN_example(tmp_path):
    class Const(nn.Module):
        value: torch.Tensor

        def __init__(self, value: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("value", value)

        def forward(self):
            return self.value

    model = NNModuleGraph()

    batch_size = 16
    noise_channels = 64
    output_channels = 64
    criterion = nn.BCELoss()

    real = model.add_input("real")

    noise = model.add_op(
        "noise",
        partial(torch.rand, (batch_size, noise_channels)),
        n_channels=noise_channels,
    )()

    with model.add_component("generator") as g:
        fake = g.add_op(
            "generator",
            nn.Sequential(
                nn.Linear(noise_channels, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_channels),
                # Tanh squashes to range [-1, 1].
                # Use something different if the range of the
                nn.Tanh(),
            ),
        )(noise)

    with model.add_component("discriminator") as d:
        fake_target = d.add_op("fake_target", Const(torch.zeros(batch_size, 1)))()
        real_target = d.add_op("real_target", Const(torch.ones(batch_size, 1)))()

        real_score = d.add_op(
            "discriminator_real",
            nn.Sequential(
                nn.Linear(output_channels, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ),
        )(real)
        discriminator_real_loss = d.add_op("real_criterion", criterion)(
            real_score, real_target
        )
        d.add_loss(discriminator_real_loss)

        fake_detached = d.add_op("detach_fake", lambda x: x.detach())(fake)
        fake_score = model.add_op(
            "descriminator_fake",
            # Extract the discriminator for application real data
            model.slice(real, real_score),
        )(fake_detached)
        discriminator_fake_loss = d.add_op("fake_criterion", criterion)(
            fake_score, fake_target
        )
        d.add_loss(discriminator_fake_loss)

    # Use discriminator score as loss for generator
    with g:
        fake_score = model.add_op(
            "descriminator_fake",
            # Extract the discriminator for application to the real data
            # (Don't convert to nn.Module since we don't need to store its parameters a second time.)
            model.slice(real, real_score),
        )(fake)
        generator_loss = g.add_op("criterion", criterion)(fake_score, real_target)
        g.add_loss(generator_loss)

    dot = model.to_dot()
    dot_fn = tmp_path / "model.dot"
    print(dot_fn)
    dot.save(dot_fn)
