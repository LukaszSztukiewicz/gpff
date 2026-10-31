import pytest
import torch
from torch import nn

from gpff.cli.train import init_model


class Model(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.representation = nn.Linear(4, 8)
        self.output_modules = nn.ModuleList([head])


def test_init_model_takes_the_representation_under_a_new_head(tmp_path):
    torch.manual_seed(0)
    pretrained = Model(head=nn.Linear(8, 3))
    torch.save(pretrained, tmp_path / "best_model")
    model = Model(head=nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 1)))
    head_before = {k: v.clone() for k, v in model.output_modules.state_dict().items()}

    init_model(model, str(tmp_path / "best_model"), ["representation"])

    for key, value in pretrained.representation.state_dict().items():
        torch.testing.assert_close(model.representation.state_dict()[key], value)
    for key, value in model.output_modules.state_dict().items():
        torch.testing.assert_close(value, head_before[key])


def test_init_model_refuses_a_representation_that_does_not_match(tmp_path):
    pretrained = Model(head=nn.Linear(8, 3))
    pretrained.representation = nn.Linear(4, 16)
    torch.save(pretrained, tmp_path / "best_model")

    with pytest.raises(RuntimeError):
        init_model(Model(head=nn.Linear(8, 1)), str(tmp_path / "best_model"), ["representation"])
