import pytest
import torch

from gpff.sample.base import Calculator


# Lightning runs validation, and the SamplerCallback with it, in inference mode
@pytest.mark.parametrize("inference_mode", [False, True])
def test_std_per_molecule_reuses_the_layout_of_the_same_idx_m(inference_mode):
    torch.manual_seed(0)
    calculator = Calculator(device="cpu")
    forces = torch.randn(7, 3)

    with torch.inference_mode(inference_mode):
        for idx_m in (torch.tensor([0, 0, 1, 1, 1, 2, 2]), torch.tensor([0, 1, 1, 1, 1, 2, 2])):
            for _ in range(2):
                sigma = calculator._compute_std_per_molecule(idx_m, forces)
                expected = torch.stack([forces[idx_m == m].std() for m in range(3)])[idx_m]
                torch.testing.assert_close(sigma, expected.unsqueeze(-1))
            assert calculator._layout_cache[0] is idx_m
