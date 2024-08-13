import torch
from pyrphoros.graph import ComputeGraph
from torch import nn
from torch.testing import assert_close


def test_ComputeGraph_empty_forward():
    g = ComputeGraph()
    m = g.build_module()
    m.forward({})


def test_ComputeGraph_forward():
    g = ComputeGraph()
    inp = g.add_input("inp")
    outp = g.add_node(nn.Linear(10, 10))(inp)
    g.add_output(outp=outp)

    m = g.build_module()

    result = m.forward({"inp": torch.zeros((4, 10))})

    assert "outp" in result

    assert result["outp"].size() == (4, 10)


def test_ComputeGraph_parameter_groups():
    g = ComputeGraph()
    x = g.add_input("inp")

    g._add_node
    x = g.add_node(nn.Linear(10, 10))(x)

    with g.parameter_group("a"):
        outp_a = g.add_node(nn.Linear(10, 1))(x)
        g.add_output(outp_a=outp_a)

    with g.parameter_group("b"):
        outp_b = g.add_node(nn.Linear(10, 1))(x)
        g.add_output(outp_b=outp_b)

    m = g.build_module()

    m.parameter_groups["root"]
    m.parameter_groups["a"]
    m.parameter_groups["b"]

    # Train
    batch = {"inp": torch.zeros((4, 10))}
    raw_batch = m._read_inputs(batch)

    for name, pg in m.parameter_groups.items():
        # Zero all model gradients
        # (Normally, only the respective group should be zeroed: pg.parameters().zero_grad())
        m.zero_grad()

        for loss_handle in pg.backward_passes:
            loss = m._forward_incremental(raw_batch, loss_handle)
            loss.backward()

        # Ensure that only gradients in the respective parameter group were updated
        for name2, pg2 in m.parameter_groups.items():
            if name2 == name:
                grad = torch.mean(
                    torch.concat([torch.linalg.norm(p.grad) if p.grad else 0 for p in pg2.parameters()])  # type: ignore
                )
                assert grad >= 0
            else:
                assert grad == 0
