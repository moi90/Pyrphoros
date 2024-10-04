import torch
from pyrphoros.graph import NNModuleGraph
from torch import nn
from torch.testing import assert_close


def test_ComputeGraph_empty_forward():
    g = NNModuleGraph()
    m = g.build_module()
    m.forward({})


def test_ComputeGraph_forward():
    g = NNModuleGraph()
    inp = g.add_input("inp")
    outp = g.add_node("linear", nn.Linear(10, 10))(inp)
    g.add_output(outp=outp)

    m = g.build_module()

    result = m.forward({"inp": torch.zeros((4, 10))})

    assert "outp" in result

    assert result["outp"].size() == (4, 10)


def test_ComputeGraph_components():
    batch_size = 4

    graph = NNModuleGraph()
    x = graph.add_input("inp")

    x = graph.add_node("linear", nn.Linear(10, 10))(x)

    with graph.create_component("a") as component_a:
        dx = component_a.add_node("detach_x", lambda x: x.detach())(x)
        outp_a = component_a.add_node("linear", nn.Linear(10, 1))(dx)
        component_a.add_output(outp_a=outp_a)
        loss = component_a.add_node("loss", nn.BCEWithLogitsLoss())(
            outp_a, torch.ones(batch_size, 1)
        )
        component_a.add_loss(loss)

    assert graph.root_component in component_a.ancestors

    with graph.create_component("b") as component_b:
        dx = component_b.add_node("detach_x", lambda x: x.detach())(x)
        outp_b = component_b.add_node("linear", nn.Linear(10, 1))(dx)
        component_b.add_output(outp_b=outp_b)
        loss = component_b.add_node("loss", nn.BCEWithLogitsLoss())(
            outp_b, torch.ones(batch_size, 1)
        )
        component_b.add_loss(loss)

    assert graph.root_component in component_b.ancestors

    nn_module = graph.build_module()

    nn_module.linear
    nn_module.a.linear
    nn_module.b.linear

    # Train
    batch = {"inp": torch.zeros((batch_size, 10))}
    raw_batch = nn_module._read_inputs(batch)

    for current_name, current_component_module in nn_module.component_modules.items():
        # Zero all model gradients to be able to detect gradients "leaking" into other components.
        # (Normally, only the respective component should be zeroed: cm.parameters().zero_grad())
        nn_module.zero_grad()

        for loss_ref in current_component_module.component.losses:
            loss = nn_module._forward_incremental(raw_batch, loss_ref)
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
