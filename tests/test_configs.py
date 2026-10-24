from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from schnetpack.utils import str2class

import gpff.cli  # registers the resolvers the configs use

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
        scheduler_cls=str2class(config.task.scheduler_cls),
    )
    hydra.utils.instantiate(config.data, train_sampler_cls=None)


@pytest.mark.parametrize("experiment", ["dm_no_t", "dm_t_aware"])
def test_generate_reads_the_noise_the_diffusion_baselines_predict(experiment):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        train = compose(
            "train", overrides=[f"experiment={experiment}", "run.path=/runs", "run.data_dir=/data"]
        )
        generate = compose("generate")

    assert generate.sampler.calculator.noise_key == train.model.output_modules[0].output_key
