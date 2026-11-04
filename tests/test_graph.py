import math
from collections import Counter

import pytest
import torch

from gpff.model.geometry import molecule_geometry, radius_of_gyration
from gpff.model.graph import KNNGraph, SizeRadiusGraph


def molecules_of(sizes, seed=0, scale=2.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, 3, generator=g, dtype=dtype) * scale for n in sizes]


def batch_of(molecules):
    positions = torch.cat(molecules)
    n_atoms = torch.tensor([len(m) for m in molecules])
    return positions, torch.repeat_interleave(torch.arange(len(molecules)), n_atoms), n_atoms


def build(graph, molecules):
    """{(i, j): weight} of a graph, with the atoms of the molecules numbered one after the other"""
    i, j, weight = graph(*batch_of(molecules))
    assert (i < j).all()
    pairs = dict(zip(zip(i.tolist(), j.tolist()), weight.tolist()))
    assert len(pairs) == len(i), "a pair is listed twice"
    return pairs


def cosine_cutoff(x):
    return 0.5 * (math.cos(math.pi * x) + 1) if x < 1 else 0.0


def reference_knn(molecules, k):
    pairs, offset = {}, 0
    for x in molecules:
        n = len(x)
        d = (x[:, None] - x[None]).norm(dim=-1)
        d.fill_diagonal_(math.inf)
        order = d.argsort(dim=1)
        # the distance to the (k+1)-th nearest atom, if there is one
        rho = d.gather(1, order)[:, k] if n - 1 > k else torch.full((n,), math.inf, dtype=d.dtype)
        union = set()
        for i in range(n):
            for j in order[i, : min(k, n - 1)].tolist():
                union.add((min(i, j), max(i, j)))
        for i, j in union:
            pairs[(offset + i, offset + j)] = cosine_cutoff(d[i, j].item() / max(rho[i].item(), rho[j].item()))
        offset += n
    return pairs


def reference_radius(molecules, radius, coefficient, exponent):
    pairs, offset = {}, 0
    for x in molecules:
        n = len(x)
        rg = ((x - x.mean(0)).pow(2).sum(-1).mean() + 1e-6).sqrt().item()
        r = radius * rg / (coefficient * n**exponent)
        d = (x[:, None] - x[None]).norm(dim=-1)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r:
                    pairs[(offset + i, offset + j)] = cosine_cutoff(d[i, j].item() / r)
        offset += n
    return pairs


def assert_same_graph(actual, expected, tolerance=1e-9):
    assert actual.keys() == expected.keys()
    for pair, weight in expected.items():
        assert actual[pair] == pytest.approx(weight, abs=tolerance), pair


def random_rigid_motion(seed, reflect=False):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if reflect:
        q[:, 0] = -q[:, 0]
    return q, torch.randn(3, generator=g, dtype=torch.float64) * 5


SIZES = [[5, 1, 12, 2, 30], [40], [3, 3], [17, 17, 17]]


@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("k", [1, 4, 16])
def test_knn_graph_matches_a_brute_force_graph(sizes, k):
    molecules = molecules_of(sizes)

    assert_same_graph(build(KNNGraph(k), molecules), reference_knn(molecules, k))


@pytest.mark.parametrize("sizes", SIZES)
def test_size_radius_graph_matches_a_brute_force_graph(sizes):
    molecules = molecules_of(sizes)
    graph = SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.2, rg_ref_exponent=0.45)

    assert_same_graph(build(graph, molecules), reference_radius(molecules, 2.5, 1.2, 0.45))


def test_knn_graph_is_the_symmetric_union_within_molecules():
    sizes, k = [20, 7, 33, 1], 5
    pairs = build(KNNGraph(k), molecules_of(sizes))

    molecule_of = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)).tolist()
    degree = Counter()
    for i, j in pairs:
        assert molecule_of[i] == molecule_of[j]
        degree[i] += 1
        degree[j] += 1
    for atom, molecule in enumerate(molecule_of):
        assert degree[atom] >= min(k, sizes[molecule] - 1), atom
    # in the union an atom has more than k neighbors if others chose it
    assert max(degree.values()) > k


def test_molecules_with_at_most_k_other_atoms_are_fully_connected_with_full_weights():
    sizes, k = [1, 2, 5, 9], 8
    pairs = build(KNNGraph(k), molecules_of(sizes))

    assert len(pairs) == sum(n * (n - 1) // 2 for n in sizes)
    assert all(weight == pytest.approx(1.0) for weight in pairs.values())


@pytest.mark.parametrize("graph", [KNNGraph(3), SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.0, rg_ref_exponent=0.5)])
def test_a_graph_changes_continuously_as_the_atoms_move(graph):
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(10, 3, generator=g, dtype=torch.float64) * 2
    velocity = torch.randn(10, 3, generator=g, dtype=torch.float64)

    previous, changes, largest = None, 0, 0.0
    for s in torch.linspace(0, 1.5, 3000):
        current = build(graph, [x0 + s * velocity])
        if previous is not None:
            pairs = previous.keys() | current.keys()
            largest = max(largest, max(abs(previous.get(p, 0.0) - current.get(p, 0.0)) for p in pairs))
            changes += previous.keys() != current.keys()
        previous = current

    # pairs entered and left the graph, each with a weight that was already close to 0
    assert changes > 0
    assert largest < 0.05


@pytest.mark.parametrize("graph", [KNNGraph(4), SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.2, rg_ref_exponent=0.45)])
def test_graph_of_a_batch_is_the_graphs_of_its_molecules(graph):
    molecules = molecules_of([5, 12, 9, 1, 20])
    parts, offset = {}, 0
    for molecule in molecules:
        parts.update({(i + offset, j + offset): w for (i, j), w in build(graph, [molecule]).items()})
        offset += len(molecule)

    assert_same_graph(build(graph, molecules), parts)


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("graph", [KNNGraph(4), SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.2, rg_ref_exponent=0.45)])
def test_graph_does_not_depend_on_where_and_how_the_molecules_are(graph, reflect):
    molecules = molecules_of([14, 9])
    q, shift = random_rigid_motion(0, reflect)

    moved = [m @ q.T + shift for m in molecules]

    assert_same_graph(build(graph, moved), build(graph, molecules))


def test_size_radius_graph_does_not_depend_on_the_scale_of_the_molecule():
    graph = SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.2, rg_ref_exponent=0.45)
    molecules = molecules_of([14, 25])

    # up to the small floor on the mean square radius
    assert_same_graph(build(graph, [20.0 * m for m in molecules]), build(graph, molecules), tolerance=1e-5)


@pytest.mark.parametrize("graph", [KNNGraph(3), SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.0, rg_ref_exponent=0.5)])
def test_weights_have_finite_gradients(graph):
    positions, idx_m, n_atoms = batch_of(molecules_of([1, 2, 6, 15]))
    positions.requires_grad_()

    _, _, weight = graph(positions, idx_m, n_atoms)
    weight.sum().backward()

    assert torch.isfinite(positions.grad).all()
    assert positions.grad.abs().sum() > 0


def test_graph_in_single_precision_is_the_graph_in_double_precision():
    molecules = molecules_of([14, 9, 30])

    single = build(KNNGraph(4), [m.float() for m in molecules])

    assert_same_graph(single, build(KNNGraph(4), molecules), tolerance=1e-4)


@pytest.mark.parametrize("make", [lambda: KNNGraph(0), lambda: SizeRadiusGraph(0.0, 1.0, 0.5), lambda: SizeRadiusGraph(1.0, -1.0, 0.5)])
def test_graph_rejects_impossible_parameters(make):
    with pytest.raises(ValueError):
        make()


def test_radius_of_gyration_is_the_root_mean_square_distance_from_the_centroid():
    molecules = molecules_of([1, 2, 9, 30])
    positions, idx_m, n_atoms = batch_of(molecules)

    expected = torch.stack([((m - m.mean(0)).pow(2).sum(-1).mean() + 1e-6).sqrt() for m in molecules])

    torch.testing.assert_close(radius_of_gyration(positions, idx_m, n_atoms), expected)
    torch.testing.assert_close(molecule_geometry(positions, idx_m, n_atoms).log_rg, expected.log())


def test_geometry_invariants_tell_a_ball_from_a_line_and_do_not_depend_on_pose_or_size():
    g = torch.Generator().manual_seed(0)
    ball = torch.randn(4000, 3, generator=g, dtype=torch.float64)
    line = torch.zeros(50, 3, dtype=torch.float64)
    line[:, 2] = torch.linspace(-3, 3, 50, dtype=torch.float64)
    molecules = [ball, line, torch.randn(9, 3, generator=g, dtype=torch.float64)]
    positions, idx_m, n_atoms = batch_of(molecules)

    shape = molecule_geometry(positions, idx_m, n_atoms).shape

    # both invariants are 1 for three equal principal variances and 0 for a line
    torch.testing.assert_close(shape[0], torch.ones(2, dtype=torch.float64), atol=0.05, rtol=0)
    torch.testing.assert_close(shape[1], torch.zeros(2, dtype=torch.float64), atol=1e-6, rtol=0)

    q, shift = random_rigid_motion(1, reflect=True)
    moved = molecule_geometry(torch.cat([m @ q.T * 3 + shift for m in molecules]), idx_m, n_atoms)
    reference = molecule_geometry(positions, idx_m, n_atoms)
    torch.testing.assert_close(moved.shape, reference.shape, atol=1e-5, rtol=0)
    torch.testing.assert_close(moved.radial, reference.radial, atol=1e-5, rtol=0)
    torch.testing.assert_close(moved.log_rg - reference.log_rg, torch.full((3,), math.log(3), dtype=torch.float64), atol=1e-5, rtol=0)
    torch.testing.assert_close(moved.direction, reference.direction @ q.T, atol=1e-5, rtol=0)
