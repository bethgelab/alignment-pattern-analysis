"""Gridsearch launcher.

This script launches a gridsearch over evaluation hyperparameters for a single model.

Usage:
```
python launch.py --model-config conigs/models/<model>.yaml --output-path <output_path>
```
"""

import os
import shutil
from pathlib import Path
from typing import Generator

import click
from benedict import benedict
from executor import execute

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"


@click.command()
@click.option(
    "--benchmark-config",
    "benchmark_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_BENCHMARK_CONFIG,
)
@click.option(
    "--model-config",
    "model_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, path_type=Path),
    required=True,
)
@click.option(
    "--job-name",
    "job_name",
    type=str,
    default="gridsearch",
    help="Prefix for the SLURM job names.",
)
@click.option(
    "--exclude",
    type=str,
    default="",
    help="Comma-separated list of nodes to exclude.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="If enabled, the jobs will not be submitted to the SLURM scheduler.",
)
def gridsearch(
    benchmark_config_path: Path,
    model_config_path: Path,
    output_path: Path,
    job_name: str,
    exclude: str,
    dry_run: bool,
):
    benchmark_config = benedict.from_yaml(benchmark_config_path)
    model_config = benedict.from_yaml(model_config_path)
    benchmark_config.merge(model_config)

    if "alignment.n_samples" in benchmark_config:
        raise ValueError(
            "The analysis is restricted to the first n_samples. "
            "Please remove the alignment.n_samples key from the config."
        )

    (output_path / "output").mkdir(exist_ok=False, parents=True)
    (output_path / "logs").mkdir(exist_ok=False, parents=True)

    prepare_container(output_path)
    prepare_code(output_path)
    prepare_configs(output_path, benchmark_config)
    prepare_slurm_script(output_path)
    submit_jobs(output_path, job_name, exclude, dry_run)

    click.echo("Gridsearch launched successfully.")
    click.echo("Good luck and Godspeed. 🤞")


def create_configs(base_config: benedict) -> Generator[benedict, None, None]:
    base_config["feature_reduction.method"] = "pca"

    for n_components in [8, 16, 32, 64, 128, 256, 512, 1024]:
        config = base_config.deepcopy()
        config["feature_reduction.n_components"] = n_components
        yield config

    # for one_alpha_per_voxel in [True]:
    #     for scoring in ['r2', 'pearson_r']:
    #         for feature_extraction_method in ['srp']:
    #             if feature_extraction_method == 'pca':
    #                 for n_components in [0.99, 0.95, 0.90, 0.80, 0.75, 0.70, 0.65]:
    #                     config = base_config.deepcopy()
    #                     config["versa.voxel_encoding.one_alpha_per_voxel"] = one_alpha_per_voxel  # noqa: E501
    #                     config["versa.voxel_encoding.params.scoring"] = scoring
    #                     config["feature_reduction.method"] = feature_extraction_method
    #                     config["feature_reduction.n_components"] = n_components
    #                     yield config
    #             elif feature_extraction_method == 'srp':
    #                 config["versa.voxel_encoding.one_alpha_per_voxel"] = one_alpha_per_voxel  # noqa: E501
    #                 config["versa.voxel_encoding.params.scoring"] = scoring
    #                 config["feature_reduction.method"] = feature_extraction_method
    #                 yield config


def prepare_container(output_path: Path):
    """Prepares the container for running the benchmark."""
    click.echo("Symlinking container...")
    source = PROJECT_ROOT / "singularity" / "multitasking.sif"
    if not source.exists():
        raise FileNotFoundError(f"Container {source} not found.")
    destination = output_path / "container.sif"
    os.symlink(source.resolve(), destination)


def prepare_code(output_path: Path):
    """Copies the code for running the benchmark."""
    click.echo("Copying code...")
    source = PROJECT_ROOT / "multitasking"
    destination = output_path / "multitasking"
    shutil.copytree(source, destination)


def prepare_configs(output_path: Path, base_config: benedict):
    """Saves the configs for the individual gridsearch runs."""
    click.echo("Preparing configs...")

    configs_path = output_path / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)

    for config_index, config in enumerate(create_configs(base_config)):
        config_path = configs_path / f"{config_index:03d}.yaml"
        config.to_yaml(filepath=config_path)


def prepare_slurm_script(output_path: Path):
    """Prepares the SLURM script for running the benchmark."""
    click.echo("Preparing SLURM script...")
    source = PROJECT_ROOT / "tasks" / "gridsearch" / "run.sh"
    destination = output_path / "run.sh"
    shutil.copy(source, destination)


def submit_jobs(output_path: Path, job_name: str, exclude: str, dry_run: bool):
    """Submits the jobs to the SLURM scheduler."""
    click.echo("Submitting jobs...")

    for config_path in sorted(output_path.glob("configs/*.yaml")):
        config = benedict.from_yaml(config_path)
        extra_slurm_options = config.get("slurm", {})
        extra_slurm_options_str = " ".join(
            [f"--{k} {v}" for k, v in extra_slurm_options.items()]
        )

        config_path = config_path.relative_to(output_path)

        command = " ".join(
            [
                f"CONFIG_PATH={config_path} ",
                "sbatch ",
                f"--job-name {job_name}/{config_path.stem} ",
                f"--output logs/%j_{config_path.stem}.log ",
                f"--error logs/%j_{config_path.stem}.log ",
                f"--exclude {exclude} " if len(exclude) > 0 else "",
                extra_slurm_options_str,
                "run.sh",
            ]
        )
        if dry_run:
            click.echo(command)
        else:
            execute(command, directory=output_path)


if __name__ == "__main__":
    gridsearch()
