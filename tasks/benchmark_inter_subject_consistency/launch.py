"""Benchmark launcher.

This script launches a "benchmark" computing intersubject consistency.

Usage:
    python launch.py --output-path wherever/you/like
"""

import os
import shutil
from pathlib import Path

import click
import numpy as np
from benedict import benedict
from executor import execute

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CONFIG_ROOT / "benchmark.yaml",
    required=True,
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
    help="Prefix for the SLURM job names."
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
@click.option(
    "--pairwise",
    is_flag=True,
    default=False,
    help=("If enabled, start pairwise subject-comparison intersubject runs. "
        "Else (default), compares each subject to the leave-one-out-mean of "
        "all other subjects."),
)
@click.option(
    "--n-partners-pairwise",
    type=int,
    default=1,
    help="Only used with --pairwise. Number of partners to compare each target to.",
)
def run_benchmark(
    config_path: Path,
    output_path: Path,
    job_name: str,
    exclude: str,
    dry_run: bool,
    pairwise: bool,
    n_partners_pairwise: int = 1,
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
    if pairwise:
        prepare_configs_pairwise(output_path, base_config,
                                 n_partners=n_partners_pairwise)
    else:
        prepare_configs_leave_one_out_mean(output_path, base_config)
    prepare_slurm_script(output_path, base_config)
    submit_jobs(output_path, job_name, exclude, dry_run)

    click.echo("Benchmark launched successfully.")
    click.echo("Good luck and Godspeed. 🤞")


def prepare_container(output_path: Path):
    """Prepares the containers for running the benchmark."""
    click.echo("Symlinking containers...")
    source = PROJECT_ROOT / "singularity" / "multitasking.sif"
    if not source.exists():
        raise FileNotFoundError(f"Container {source} not found.")
    destination = output_path / "container.sif"
    os.symlink(source, destination)


def prepare_code(output_path: Path):
    """Copies the code for running the benchmark."""
    click.echo("Copying code...")
    source = PROJECT_ROOT / "multitasking"
    destination = output_path / "multitasking"
    shutil.copytree(source, destination)


def prepare_configs_leave_one_out_mean(output_path: Path, base_config: benedict):
    """Prepares the config files for each subject as target, vs. LOOM as model."""
    output_path = output_path / "configs"
    output_path.mkdir(exist_ok=True, parents=True)

    # model_config_paths = list((CONFIG_ROOT / "models").glob("*.yaml"))
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    click.echo(f"Preparing configs for {len(subjects)} subjects...")

    for actual_subject in subjects:
    # if True:
        subject = "all" # important
        model_config = base_config.deepcopy()
        model_config["fmri"]["intersubject"] = benedict(
            {"mode": "leave-one-out-mean"})
        model_config["fmri"]["sub_id"] = subject
        model_config["feature_extraction"]["model"] = "all-other-subjects"

        subject = actual_subject
        model_config["fmri"]["intersubject"]["target_sub_ids"] = [subject]


        model_name = f"{subject}-vs-all-other-subjects"
        model_config_path = output_path / f"{model_name}.yaml"
        model_config.to_yaml(filepath=model_config_path)


def prepare_configs_pairwise(output_path: Path, base_config: benedict,
                             n_partners: int = 1):
    """Prepares the config files for each pairwise subject comparison."""
    output_path = output_path / "configs"
    output_path.mkdir(exist_ok=True, parents=True)

    # model_config_paths = list((CONFIG_ROOT / "models").glob("*.yaml"))
    target_subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    assert n_partners < len(target_subjects), ( "n_partners must be at most one"
                                    " less than the number of target subjects")

    # Select random source subject for each target subject, avoiding double comparisons
    source_subjects: list[list[str]] = []
    for j, sub in enumerate(target_subjects):
        if n_partners == len(target_subjects) - 1:
            # sample all other subjects once; include duplicate pairs
            source_subjects = [s for s in target_subjects if s != sub]
        else:
            # exclude self and subjects for which we are the source
            exclude = [sub] + [s for (i, s) in enumerate(target_subjects[:j]) \
                if source_subjects[i] == sub]
            include = [s for s in target_subjects if s not in exclude]
            if len(include) == 0:
                raise ValueError(f"Error, subject {sub} has no source subjects left to "
                                f"compare to. All possible subjects likely "
                                "already had this subject sampled as source for them."
                                f"j: {j}, source_subjects: {source_subjects}")
            source_subjects_ = np.random.choice(include,
                                                n_partners,
                                                replace=False).tolist()
            # click.echo(f"Source subjects: {source_subjects_}")
        source_subjects.append(source_subjects_)

    click.echo(f"Preparing pairwise-comparison intersubject configs "
    f"for {len(target_subjects)} subjects...")

    for source_subjects_, target_subject_ in list(zip(
            source_subjects, target_subjects, strict=True)):
        for source_subject_ in source_subjects_:
            model_config = base_config.deepcopy()
            model_config["fmri"]["intersubject"] = benedict({"mode": "pairwise"})
            # determines which subject's data is loaded:
            model_config["fmri"]["sub_id"] = [source_subject_, target_subject_]
            model_config["feature_extraction"]["model"] = source_subject_

            model_config["fmri"]["intersubject"]["target_sub_ids"] = [target_subject_]
            model_config["fmri"]["intersubject"]["source_sub_ids"] = [[source_subject_]]


            run_name = f"{source_subject_}-vs-{target_subject_}"
            model_config_path = output_path / f"{run_name}.yaml"
            model_config.to_yaml(filepath=model_config_path)


def prepare_slurm_script(output_path: Path, base_config: benedict):
    """Prepares the SLURM script for running the benchmark."""
    click.echo("Preparing SLURM script...")
    source =( PROJECT_ROOT / "tasks" /
                "bold_moments_benchmark_intersubject_consistency" /
                "run.sh")
    destination = output_path / "run.sh"
    shutil.copy(source, destination)


def submit_jobs(output_path: Path, job_name: str, exclude: str, dry_run: bool):
    """Submits the jobs to the SLURM scheduler."""
    click.echo("Submitting jobs...")
    for config_path in sorted(output_path.glob("configs/*.yaml")):
        config = benedict.from_yaml(config_path)
        extra_slurm_options = config.get("slurm", {})
        extra_slurm_options_str = " ".join([
            f"--{k} {v}" for k, v in extra_slurm_options.items()
        ])

        config_path = config_path.relative_to(output_path)

        command = " ".join([
            f"CONFIG_PATH={config_path} ",
            "CONTAINER=container.sif ",
            "sbatch ",
            f"--job-name {job_name}/{config_path.stem} ",
            f"--output logs/%j_{config_path.stem}.log ",
            f"--error logs/%j_{config_path.stem}.log ",
            f"--exclude {exclude} " if len(exclude) > 0 else "",
            extra_slurm_options_str,
            "run.sh",
        ])

        if dry_run:
            click.echo(command)
        else:
            execute(command, directory=output_path)


if __name__ == "__main__":
    run_benchmark()
