"""Benchmark launcher.

This script launches a benchmark for all models specified in `configs/models`,
parallelized using SLURM.

Usage:
    python launch.py --output-path wherever/you/like
"""

import os
import shlex
import shutil
from pathlib import Path

import click
from benedict import benedict
from executor import execute

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_ROOT = PROJECT_ROOT / "configs"


def slurm_options_from_env(partition_variables: tuple[str, ...]) -> str:
    """Builds the site-specific sbatch flags from the environment.

    The `#SBATCH` lines in run.sh are comments that sbatch reads before any
    shell runs, so they cannot expand environment variables. Site-specific
    settings therefore have to arrive on the sbatch command line, where they
    override the matching directive in the script. That keeps the names of
    partitions, accounts and filesystems out of the repository.

    Args:
        partition_variables: Environment variables to consult for the
            partition, in order of preference. The first one set wins.

    Returns:
        The flags, ready to splice into the sbatch command line. Empty if the
        environment sets nothing, in which case the cluster defaults and the
        directives in run.sh apply unchanged.
    """
    options: list[str] = []

    for variable in partition_variables:
        partition = os.environ.get(variable, "").strip()
        if partition:
            options.append(f"--partition {shlex.quote(partition)}")
            break

    account = os.environ.get("MULTITASKING_SLURM_ACCOUNT", "").strip()
    if account:
        options.append(f"--account {shlex.quote(account)}")

    # Anything else this site needs -- --qos, --constraint, a different
    # --gres -- passed through verbatim.
    extra_args = os.environ.get("MULTITASKING_SLURM_EXTRA_ARGS", "").strip()
    if extra_args:
        options.append(extra_args)

    return " ".join(options)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CONFIG_ROOT / "benchmark.yaml",
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--job-name",
    "job_name",
    type=str,
    default="multitasking",
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
def run_benchmark(
    config_path: Path,
    output_path: Path,
    job_name: str,
    exclude: str,
    dry_run: bool,
):
    if output_path.exists():
        raise ValueError(f"Output path {output_path} already exists.")

    base_config = benedict.from_yaml(config_path)

    if "alignment.n_samples" in base_config:
        raise ValueError(
            "The analysis is restricted to the first n_samples. "
            "Please remove the alignment.n_samples key from the config."
        )

    (output_path / "output").mkdir(exist_ok=False, parents=True)
    (output_path / "logs").mkdir(exist_ok=False, parents=True)

    prepare_container(output_path)
    prepare_code(output_path)
    prepare_configs(output_path, base_config)
    prepare_slurm_script(output_path, base_config)
    submit_jobs(output_path, job_name, exclude, dry_run)

    click.echo("Benchmark launched successfully.")
    click.echo("Good luck and Godspeed. 🤞")


def prepare_container(output_path: Path):
    """Links the two container images into the job directory.

    The benchmark needs two images: models whose name starts with "mmaction2"
    run in the mmaction2 image, every other model in the mmflow image. Both
    are linked here as container_<variant>.sif. submit_jobs then picks the one
    each model needs and passes that filename to run.sh as $CONTAINER.

    Args:
        output_path: The job directory to link the containers into.

    Raises:
        FileNotFoundError: If a container variant is missing, naming where it
            is expected.
    """
    click.echo("Symlinking containers...")
    for variant in ("mmflow", "mmaction2"):
        source = PROJECT_ROOT / "singularity" / f"multitasking_{variant}.sif"
        if not source.exists():
            raise FileNotFoundError(
                f"Container {source} not found. Build it from "
                "singularity/container.def, or symlink an existing image to "
                "that path."
            )
        os.symlink(source.resolve(), output_path / f"container_{variant}.sif")


def prepare_code(output_path: Path):
    """Copies the code for running the benchmark."""
    click.echo("Copying code...")
    source = PROJECT_ROOT / "multitasking"
    destination = output_path / "multitasking"
    shutil.copytree(source, destination)


def prepare_configs(output_path: Path, base_config: benedict):
    """Prepares the config files for the individual models."""
    output_path = output_path / "configs"
    output_path.mkdir(exist_ok=True, parents=True)

    model_config_paths = list((CONFIG_ROOT / "models").glob("*.yaml"))

    click.echo(f"Preparing configs for {len(model_config_paths)} models...")

    for model_config_path in model_config_paths:
        model_overrides = benedict.from_yaml(model_config_path)

        model_config = base_config.deepcopy()
        model_config.merge(model_overrides)

        model_name = model_config_path.stem
        model_config_path = output_path / f"{model_name}.yaml"
        model_config.to_yaml(filepath=model_config_path)


def prepare_slurm_script(output_path: Path, base_config: benedict):
    """Prepares the SLURM script for running the benchmark."""
    click.echo("Preparing SLURM script...")
    source = PROJECT_ROOT / "tasks" / "benchmark" / "run.sh"
    destination = output_path / "run.sh"
    shutil.copy(source, destination)


def submit_jobs(output_path: Path, job_name: str, exclude: str, dry_run: bool):
    """Submits the jobs to the SLURM scheduler."""
    click.echo("Submitting jobs...")
    env_slurm_options = slurm_options_from_env(("MULTITASKING_SLURM_PARTITION",))
    for config_path in sorted(output_path.glob("configs/*.yaml")):
        config = benedict.from_yaml(config_path)
        extra_slurm_options = config.get("slurm", {})
        extra_slurm_options_str = " ".join(
            [f"--{k} {v}" for k, v in extra_slurm_options.items()]
        )

        container_variant = (
            "mmaction2"
            if config["feature_extraction"]["model"].startswith("mmaction2")
            else "mmflow"
        )

        config_path = config_path.relative_to(output_path)

        command = " ".join(
            [
                f"CONFIG_PATH={config_path} ",
                f"CONTAINER=container_{container_variant}.sif ",
                "sbatch ",
                f"--job-name {job_name}/{config_path.stem} ",
                f"--output logs/%j_{config_path.stem}.log ",
                f"--error logs/%j_{config_path.stem}.log ",
                f"--exclude {exclude} " if len(exclude) > 0 else "",
                f"{env_slurm_options} " if env_slurm_options else "",
                # Listed last so a config's own slurm: block overrides the site
                # defaults from the environment.
                extra_slurm_options_str,
                "run.sh",
            ]
        )
        if dry_run:
            click.echo(command)
        else:
            execute(command, directory=output_path)


if __name__ == "__main__":
    run_benchmark()
