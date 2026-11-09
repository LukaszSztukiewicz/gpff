import re
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from schnetpack import properties
from schnetpack.utils import str2class

import gpff.cli  # registers the resolvers the configs use
from gpff import keys
from gpff.sample.jump_optimizers import MDMin

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"
EXPERIMENTS = sorted(path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml"))


def test_multirun_jobs_use_the_single_run_folder():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("train", overrides=["run.path=/runs"], return_hydra_config=True)

    # a sweep job resumes with run.id=<id> run.path=<path> like a single run
    sweep = config.hydra.sweep
    assert f"{sweep.dir}/{sweep.subdir}" == config.hydra.run.dir


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_builds(experiment):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train", overrides=[f"experiment={experiment}", "run.path=/runs", "run.data_dir=/data"]
        )

    # what gpff_train instantiates before it reads any data
    model = hydra.utils.instantiate(config.model)
    hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None,
    )
    hydra.utils.instantiate(config.data, train_sampler_cls=None)


def build_block_adam_task(experiment, block_update, *overrides):
    """What gpff_train instantiates of the model and the task with task/optimizer=block_adam, before any data."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train",
            overrides=[
                f"experiment={experiment}",
                "task/optimizer=block_adam",
                f"task.optimizer_args.block_update={block_update}",
                "run.path=/runs",
                "run.data_dir=/data",
                *overrides,
            ],
        )
    model = hydra.utils.instantiate(config.model)
    task = hydra.utils.instantiate(
        config.task, model=model, optimizer_cls=str2class(config.task.optimizer_cls), scheduler_cls=None
    )
    return config, task


@pytest.mark.parametrize("block_update", ["rms", "muon", "normuon", "muonh"])
def test_block_adam_takes_the_weight_normalized_blocks_of_the_stable_recipe(block_update):
    config, task = build_block_adam_task("gpff_stable", block_update)

    assert config.globals.weight_norm
    blocks = [group for group in task.configure_optimizers().param_groups if group["block"]]
    assert blocks and all(group["normalized"] for group in blocks)


def test_muonh_is_refused_without_weight_norm_when_the_task_is_built():
    # hydra wraps the ValueError of the task in an InstantiationException that quotes it
    with pytest.raises((ValueError, InstantiationException), match="muonh"):
        build_block_adam_task("gpff_stable", "muonh", "globals.weight_norm=false")


@pytest.mark.parametrize("experiment", ["dm_no_t", "dm_t_aware"])
def test_generate_reads_the_noise_the_diffusion_baselines_predict(experiment):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        train = compose(
            "train", overrides=[f"experiment={experiment}", "run.path=/runs", "run.data_dir=/data"]
        )
        generate = compose("generate")

    assert generate.sampler.calculator.noise_key == train.model.output_modules[0].output_key


def test_generate_help_lists_every_sampler_config():
    help_text = (CONFIG_DIR / "generate.yaml").read_text()
    listed = set(re.findall(r"- sampler=(\w+)", help_text))
    assert listed == {path.stem for path in (CONFIG_DIR / "sampler").glob("*.yaml")}


@pytest.mark.parametrize("sampler", ["ctmc_types", "continuous_types"])
def test_type_sampler_configs_build(sampler):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("generate", overrides=[f"sampler={sampler}"])

    calculator = hydra.utils.instantiate(config.sampler.calculator, model=None)
    hydra.utils.instantiate(config.sampler, calculator=calculator)


@pytest.mark.parametrize("experiment, process", [("gpff_ctmc_types", "AbsorbingTypes"), ("gpff_continuous_types", "ContinuousTypes")])
def test_type_experiments_train_the_process_their_head_keeps(experiment, process):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("train", overrides=[f"experiment={experiment}", "run.path=/runs", "run.data_dir=/data"])

    model = hydra.utils.instantiate(config.model)
    transform = next(t for t in hydra.utils.instantiate(config.data.transforms) if hasattr(t, "process"))
    head = model.output_modules[1]
    assert type(head.process).__name__ == type(transform.process).__name__ == process
    assert head.process.vocabulary.numbers == (1, 6, 7, 8, 9, 99)


@pytest.mark.parametrize("experiment", ["gpff_ctmc_types", "gpff_continuous_types"])
def test_type_experiments_take_absent_canvases(experiment):
    overrides = [f"experiment={experiment}", "globals.absent=true", "globals.canvas_size=29"]
    if experiment == "gpff_continuous_types":
        overrides.append("globals.type_process.mask=true")
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("train", overrides=[*overrides, "run.path=/runs", "run.data_dir=/data"])

    model = hydra.utils.instantiate(config.model)
    canvas, *_ = hydra.utils.instantiate(config.data.transforms)
    prior = hydra.utils.instantiate(config.prior)
    process = model.output_modules[1].process
    assert process.vocabulary.numbers == (1, 6, 7, 8, 9, 0, 99)
    assert canvas.size == prior.size == 29
    assert getattr(process, "mask", True)


def test_adaptive_direct_denoising_config_builds_a_sampler_with_a_jump_optimizer():
    overrides = ["sampler=adaptive_direct_denoising", "sampler.jump_optimizer={_target_:gpff.sample.jump_optimizers.MDMin}"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("generate", overrides=overrides)

    calculator = hydra.utils.instantiate(config.sampler.calculator, model=None)
    sampler = hydra.utils.instantiate(config.sampler, calculator=calculator)
    assert isinstance(sampler.jump_optimizer, MDMin)
    assert len(sampler.transport_levels) == config.sampler.transport_steps + 1


@pytest.mark.parametrize("geometry_sampler", ["vp_geometries", "shape_geometries"])
def test_geometry_sampler_config_builds_a_transform(geometry_sampler):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train",
            overrides=[
                "experiment=gpff",
                "run.path=/runs",
                "run.data_dir=/data",
                f"generative_transform/pseudo_pes/geometry_sampler={geometry_sampler}",
            ],
        )

    transform = hydra.utils.instantiate(config.generative_transform)
    inputs = {properties.R: torch.randn(6, 3, dtype=torch.float64), properties.Z: torch.tensor([6, 6, 1, 1, 1, 1])}
    outputs = transform(inputs)
    assert outputs[keys.forces].shape == (6, 3)


def compose_transform(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train", overrides=["experiment=gpff", "run.path=/runs", "run.data_dir=/data", *overrides]
        )
    return hydra.utils.instantiate(config.generative_transform)


def molecule(n_atoms=6):
    torch.manual_seed(0)
    return {
        properties.R: torch.randn(n_atoms, 3, dtype=torch.float64),
        properties.Z: torch.tensor([6, 6, 1, 1, 1, 1][:n_atoms]),
    }


@pytest.mark.parametrize("weight_generator", ["floored_sigma_weights", "soft_sigma_weights"])
def test_sigma_weight_generator_configs_build_a_transform(weight_generator):
    transform = compose_transform(f"generative_transform/pseudo_pes/weight_generator={weight_generator}")

    outputs = transform(molecule())
    assert outputs[keys.forces_weight].shape == (6, 1)
    assert torch.isfinite(outputs[keys.forces_weight]).all()


@pytest.mark.parametrize("weight_generator", ["floored_sigma_weights", "soft_sigma_weights"])
def test_anchors_get_a_zero_force_label_and_a_finite_weight(weight_generator):
    transform = compose_transform(
        f"generative_transform/pseudo_pes/weight_generator={weight_generator}",
        "pseudo_pes.geometry_sampler.anchor_fraction=1.0",
    )

    outputs = transform(molecule())
    assert (outputs[keys.sigma_curr] == 0).all()
    # the permutation alignment recenters x_t, which leaves a rounding error
    torch.testing.assert_close(outputs[keys.forces], torch.zeros(6, 3, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.isfinite(outputs[keys.forces_weight]).all()


def test_align_rotation_max_sigma_reaches_the_transform():
    transform = compose_transform(
        "globals.align_rotation=true", "generative_transform.align_rotation_max_sigma=1.0"
    )

    assert transform.align_rotation and transform.align_rotation_max_sigma == 1.0
    assert compose_transform().align_rotation_max_sigma is None


def test_edm_preconditioning_recipe_builds():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose(
            "train",
            overrides=[
                "experiment=gpff",
                "run.path=/runs",
                "run.data_dir=/data",
                "globals.preconditioned_forces=true",
                "globals.sigma_data=2.5",
                "globals.cutoff=10",
                "globals.n_rbf=100",
                "pseudo_pes.weight_generator.max_clipping=1000",
            ],
        )

    model = hydra.utils.instantiate(config.model)
    assert model.representation.sigma_data == 2.5
    assert model.representation.cutoff == 10
    assert model.representation.radial_basis.n_rbf == 100
    assert model.output_modules[0].preconditioned and not model.output_modules[0].bounded
    assert hydra.utils.instantiate(config.generative_transform).weight_generator.max_clipping == 1000
