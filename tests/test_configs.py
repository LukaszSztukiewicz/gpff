from pathlib import Path

from hydra import compose, initialize_config_dir

import gpff.cli  # registers the resolvers the configs use

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"


def test_multirun_jobs_use_the_single_run_folder():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        config = compose("train", overrides=["run.path=/runs"], return_hydra_config=True)

    # a sweep job resumes with run.id=<id> run.path=<path> like a single run
    sweep = config.hydra.sweep
    assert f"{sweep.dir}/{sweep.subdir}" == config.hydra.run.dir
