from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import gpff.cli  # registers the resolvers the configs use
from gpff.model.graph import KNNGraph, SizeRadiusGraph
from gpff.nn.radial import HybridGaussianRBF, LogGaussianRBF, UniformGaussianRBF

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"


def config_of(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        return compose("train", overrides=[*overrides, "run.path=/runs", "run.data_dir=/data"])


def model_of(*overrides):
    return hydra.utils.instantiate(config_of(*overrides).model)


def resolved(config, *keys):
    return {key: OmegaConf.to_container(config[key], resolve=True) for key in keys}


def test_scalable_experiment_with_its_defaults_is_the_gpff_experiment():
    gpff, scalable = config_of("experiment=gpff"), config_of("experiment=gpff_scalable")

    # the same data, task and noise ...
    assert resolved(scalable, "data", "task", "prior", "generative_transform") == resolved(
        gpff, "data", "task", "prior", "generative_transform"
    )
    # ... and a model with the same weights, the same Gaussians
    base, model = hydra.utils.instantiate(gpff.model), hydra.utils.instantiate(scalable.model)
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in base.state_dict().items()}
    assert model.representation.radial_basis.n_rbf == 600
    distances = torch.rand(20, 1) * 100
    torch.testing.assert_close(
        model.representation.radial_basis(distances), base.representation.radial_basis(distances), rtol=0, atol=0
    )
    assert getattr(model.representation, "graph", None) is None
    assert not model.representation.global_channel


@pytest.mark.parametrize("experiment", ["gpff_scalable", "gpff_scalable_graph"])
@pytest.mark.parametrize(
    "name, cls, n_rbf", [("uniform", UniformGaussianRBF, 600), ("hybrid", HybridGaussianRBF, 42), ("log", LogGaussianRBF, 32)]
)
def test_radial_basis_is_chosen_by_its_group(experiment, name, cls, n_rbf):
    representation = model_of(f"experiment={experiment}", f"model/representation/radial_basis={name}").representation

    assert isinstance(representation.radial_basis, cls)
    assert representation.radial_basis.n_rbf == n_rbf
    assert representation.filter_net.in_features == n_rbf


def test_uniform_basis_takes_its_size_and_range_from_the_globals():
    basis = model_of("experiment=gpff_scalable", "globals.n_rbf=64", "globals.rbf_cutoff=300").representation.radial_basis

    assert basis.n_rbf == 64
    assert basis.offsets[-1] == 300


def test_filter_mlp_and_global_channel_are_switched_by_globals():
    representation = model_of(
        "experiment=gpff_scalable", "globals.filter_hidden=64", "globals.global_channel=true"
    ).representation

    assert representation.filter_hidden_net.out_features == 64
    assert representation.filter_net.in_features == 64
    assert len(representation.global_channels) == representation.n_interactions == 4


def test_graph_experiment_builds_the_graph_in_the_model_and_not_in_the_data():
    config = config_of("experiment=gpff_scalable_graph")
    model = hydra.utils.instantiate(config.model)

    assert isinstance(model.representation.graph, KNNGraph)
    assert model.representation.graph.k == 16
    assert len(model.input_modules) == 0
    targets = [transform.get("_target_") for transform in config.data.transforms]
    assert "gpff.utils.neighbors.AllToAllNeighborList" not in targets
    assert targets[-1] == "schnetpack.transform.CastTo32"


def test_graph_experiment_takes_k_and_the_size_radius_graph_from_the_globals():
    assert model_of("experiment=gpff_scalable_graph", "globals.knn_k=8").representation.graph.k == 8

    graph = model_of(
        "experiment=gpff_scalable_graph",
        "model/representation/graph=size_radius",
        "globals.rg_ref_coefficient=1.5",
        "globals.rg_ref_exponent=0.4",
    ).representation.graph

    assert isinstance(graph, SizeRadiusGraph)
    assert (graph.radius, graph.rg_ref_coefficient, graph.rg_ref_exponent) == (5.0, 1.5, 0.4)
