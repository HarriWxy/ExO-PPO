"""Launch one or more flow experiments as independent processes.

Examples::

    # Two ExO runs, one per GPU, with different random seeds.
    python main.py --algorithm flow --gpus 0,1 --seeds 0,1 -- \
        --env-id Walker2d-v5 --env-backend gymnasium --device auto \
        --total-steps 1000000

    # Two standard PPO runs using the same shared framework.
    python main.py --algorithm ppo --gpus 0,1 --seeds 0,1 -- \
        --env-id Walker2d-v5 --env-backend gymnasium \
        --total-steps 1000000 --replay-N 1 --warmup-rollouts 1

    # Define fully independent runs in a JSON file.
    python main.py --config runs.json --max-parallel 2

The launcher uses ``subprocess.Popen`` instead of importing the training
module in this process.  Therefore each child receives its GPU environment
before ``torch`` is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent
ALGORITHM_MODULES = {
    "flow": "flow.torch_train",
    "ppo": "flow.torch_ppo_train",
}


@dataclass(frozen=True)
class Job:
    index: int
    name: str
    algorithm: str
    module: str
    gpu: str | None
    args: list[str]


def _resolve_module(algorithm: str) -> str:
    module = ALGORITHM_MODULES.get(algorithm, algorithm)
    if "." not in module:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; use flow, ppo, or a dotted module name"
        )
    return module


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "run"


def _split_csv(value: str | None) -> list[str | None]:
    if value is None or not value.strip():
        return [None]
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return [None]
    return [None if item.lower() in {"auto", "inherit"} else item for item in items]


def _split_seeds(value: str | None) -> list[int | None]:
    if value is None or not value.strip():
        return [None]
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--seeds must be a comma-separated list of integers") from error


def _broadcast(values: list[Any], count: int, label: str) -> list[Any]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"{label} has {len(values)} values, but {count} jobs are required; "
            "provide one value or one value per job"
        )
    return values


def _option_value(args: Sequence[str], option: str) -> str | None:
    prefix = option + "="
    for index, argument in enumerate(args):
        if argument == option and index + 1 < len(args):
            return args[index + 1]
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def _set_option(args: list[str], option: str, value: str) -> None:
    """Replace an option in-place, or append it if it is not present."""

    cleaned: list[str] = []
    index = 0
    prefix = option + "="
    while index < len(args):
        argument = args[index]
        if argument == option:
            index += 2
            continue
        if argument.startswith(prefix):
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    cleaned.extend([option, value])
    args[:] = cleaned


def _parameters_to_cli(parameters: Mapping[str, Any]) -> list[str]:
    """Convert JSON parameter names/values to argparse-style arguments."""

    aliases = {
        "actor-learning-rate": "actor-lr",
        "critic-learning-rate": "critic-lr",
        "entropy-coefficient": "entropy-coef",
        "ofp-coefficient": "ofp-coef",
    }
    result: list[str] = []
    for key, value in parameters.items():
        option = key if key.startswith("--") else "--" + key.replace("_", "-")
        option = "--" + aliases.get(option[2:], option[2:])
        if value is None or value is False:
            continue
        if value is True:
            result.append(option)
            continue
        if isinstance(value, (list, tuple)):
            if option == "--hidden-sizes":
                value = ",".join(str(item) for item in value)
            else:
                value = ",".join(str(item) for item in value)
        result.extend([option, str(value)])
    return result


def _default_job_name(
    index: int, algorithm: str, gpu: str | None, args: Sequence[str]
) -> str:
    seed = _option_value(args, "--seed") or "default"
    gpu_label = gpu or "inherit"
    return _safe_name(f"{algorithm}-seed{seed}-gpu{gpu_label}-run{index}")


def _make_job(
    index: int,
    algorithm: str,
    gpu: str | None,
    args: list[str],
    name: str | None = None,
) -> Job:
    return Job(
        index=index,
        name=_safe_name(name or _default_job_name(index, algorithm, gpu, args)),
        algorithm=algorithm,
        module=_resolve_module(algorithm),
        gpu=gpu,
        args=args,
    )


def _jobs_from_cli(
    algorithm: str,
    gpus: str | None,
    seeds: str | None,
    forwarded_args: Sequence[str],
) -> list[Job]:
    gpu_values = _split_csv(gpus)
    seed_values = _split_seeds(seeds)
    count = max(len(gpu_values), len(seed_values))
    gpu_values = _broadcast(gpu_values, count, "--gpus")
    seed_values = _broadcast(seed_values, count, "--seeds")

    jobs: list[Job] = []
    for index, (gpu, seed) in enumerate(zip(gpu_values, seed_values)):
        args = list(forwarded_args)
        if seed is not None:
            _set_option(args, "--seed", str(seed))
        jobs.append(_make_job(index, algorithm, gpu, args))
    return jobs


def _jobs_from_config(
    path: Path, forwarded_args: Sequence[str]
) -> tuple[list[Job], int | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read launcher config {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in launcher config {path}: {error}") from error

    if isinstance(payload, list):
        defaults: Mapping[str, Any] = {}
        common: Mapping[str, Any] = {}
        run_entries = payload
        configured_max_parallel = None
    elif isinstance(payload, Mapping):
        defaults = payload
        common_value = payload.get("common", {})
        if not isinstance(common_value, Mapping):
            raise ValueError("launcher config 'common' must be an object")
        common = common_value
        run_entries = payload.get("runs", [])
        configured_max_parallel = payload.get("max_parallel")
        if not isinstance(run_entries, list) or not run_entries:
            raise ValueError("launcher config must contain a non-empty 'runs' list")
    else:
        raise ValueError("launcher config must be a JSON array or object")

    jobs: list[Job] = []
    reserved = {"name", "algorithm", "module", "gpu", "args", "params"}
    for index, entry in enumerate(run_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"runs[{index}] must be an object")

        parameters: dict[str, Any] = dict(common)
        direct_parameters = {
            key: value for key, value in entry.items() if key not in reserved
        }
        parameters.update(direct_parameters)

        explicit_args = entry.get("args", entry.get("params", {}))
        if isinstance(explicit_args, Mapping):
            parameters.update(explicit_args)
            args = list(forwarded_args) + _parameters_to_cli(parameters)
        elif isinstance(explicit_args, list):
            args = list(forwarded_args) + [str(item) for item in explicit_args]
            args.extend(_parameters_to_cli(parameters))
        else:
            raise ValueError(f"runs[{index}].args must be an object or array")

        algorithm = str(
            entry.get("algorithm", entry.get("module", defaults.get("algorithm", "flow")))
        )
        gpu_value = entry.get("gpu", defaults.get("gpu"))
        gpu = None if gpu_value is None else str(gpu_value)
        jobs.append(
            _make_job(
                index,
                algorithm,
                gpu,
                args,
                name=str(entry["name"]) if entry.get("name") is not None else None,
            )
        )

    max_parallel = None
    if configured_max_parallel is not None:
        try:
            max_parallel = int(configured_max_parallel)
        except (TypeError, ValueError) as error:
            raise ValueError("launcher config 'max_parallel' must be an integer") from error
    return jobs, max_parallel


def _prepare_child(job: Job) -> tuple[list[str], dict[str, str]]:
    args = list(job.args)
    base_log_dir = _option_value(args, "--log-dir") or "logs"
    isolated_log_dir = Path(os.path.expanduser(base_log_dir)) / "launcher" / job.name
    _set_option(args, "--log-dir", str(isolated_log_dir))

    checkpoint_dir = _option_value(args, "--checkpoint-dir")
    if checkpoint_dir:
        isolated_checkpoint_dir = (
            Path(os.path.expanduser(checkpoint_dir)) / "launcher" / job.name
        )
        _set_option(args, "--checkpoint-dir", str(isolated_checkpoint_dir))

    child_env = {key: value for key, value in os.environ.items()}
    child_env["PYTHONUNBUFFERED"] = "1"
    if job.gpu is not None:
        if job.gpu.lower() == "cpu":
            child_env["CUDA_VISIBLE_DEVICES"] = ""
            default_device = "cpu"
        elif job.gpu.lower() in {"auto", "inherit"}:
            default_device = "auto"
        else:
            child_env["CUDA_VISIBLE_DEVICES"] = job.gpu
            default_device = "cuda"
        if _option_value(args, "--device") is None:
            _set_option(args, "--device", default_device)

    return args, child_env


def _default_parallelism(jobs: Sequence[Job]) -> int:
    assigned_gpus = [job.gpu for job in jobs if job.gpu is not None]
    if assigned_gpus and len(set(assigned_gpus)) < len(jobs):
        return max(1, len(set(assigned_gpus)))
    return max(1, len(jobs))


def _terminate_processes(active: Mapping[int, tuple[Job, subprocess.Popen[Any]]]) -> None:
    for _, process in active.values():
        if process.poll() is None:
            process.terminate()
    for _, process in active.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_jobs(
    jobs: Sequence[Job],
    max_parallel: int,
    *,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    prepared = {job.index: _prepare_child(job) for job in jobs}

    if dry_run:
        for job in jobs:
            args, child_env = prepared[job.index]
            command = [sys.executable, "-m", job.module, *args]
            gpu_display = child_env.get("CUDA_VISIBLE_DEVICES", "inherit")
            print(
                f"[dry-run] {job.name} CUDA_VISIBLE_DEVICES={gpu_display!r} "
                f"{shlex.join(command)}"
            )
        return 0

    active: dict[int, tuple[Job, subprocess.Popen[Any]]] = {}
    failures: list[tuple[Job, int]] = []
    next_job = 0
    try:
        while next_job < len(jobs) or active:
            while next_job < len(jobs) and len(active) < max_parallel:
                job = jobs[next_job]
                args, child_env = prepared[job.index]
                command = [sys.executable, "-m", job.module, *args]
                gpu_display = child_env.get("CUDA_VISIBLE_DEVICES", "inherit")
                print(
                    f"[launcher] starting {job.name} "
                    f"CUDA_VISIBLE_DEVICES={gpu_display!r}: {shlex.join(command)}",
                    flush=True,
                )
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=child_env,
                    )
                except OSError as error:
                    print(f"[launcher] failed to start {job.name}: {error}", flush=True)
                    failures.append((job, 1))
                    next_job += 1
                    if not keep_going:
                        _terminate_processes(active)
                        return 1
                    continue
                active[process.pid] = (job, process)
                next_job += 1

            if active:
                time.sleep(0.25)
            for pid, (job, process) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[pid]
                if return_code == 0:
                    print(f"[launcher] finished {job.name} successfully", flush=True)
                else:
                    failures.append((job, return_code))
                    print(
                        f"[launcher] {job.name} exited with code {return_code}",
                        flush=True,
                    )
                    if not keep_going:
                        _terminate_processes(active)
                        return return_code or 1
    except KeyboardInterrupt:
        print("\n[launcher] interrupted; stopping active jobs", flush=True)
        _terminate_processes(active)
        return 130

    if failures:
        print("[launcher] failed jobs:", flush=True)
        for job, return_code in failures:
            print(f"  - {job.name}: exit {return_code}", flush=True)
        return 1
    print(f"[launcher] all {len(jobs)} jobs finished successfully", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch multiple flow.torch_train/flow.torch_ppo_train jobs"
    )
    parser.add_argument(
        "--algorithm",
        default="flow",
        help="flow, ppo, or a dotted Python module name",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="comma-separated GPU IDs, cpu, or auto; one value is broadcast",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seeds; one value is broadcast to all jobs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file containing independent runs; overrides --algorithm/--gpus/--seeds",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="maximum concurrent processes; 0 chooses a safe default",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue other jobs after one job exits with an error",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print child commands without starting training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    launcher_args, forwarded_args = parser.parse_known_args(argv)
    forwarded_args = [argument for argument in forwarded_args if argument != "--"]

    try:
        if launcher_args.config is not None:
            jobs, configured_parallelism = _jobs_from_config(
                launcher_args.config, forwarded_args
            )
            if launcher_args.max_parallel > 0:
                max_parallel = launcher_args.max_parallel
            elif configured_parallelism is not None:
                max_parallel = configured_parallelism
            else:
                max_parallel = _default_parallelism(jobs)
        else:
            jobs = _jobs_from_cli(
                launcher_args.algorithm,
                launcher_args.gpus,
                launcher_args.seeds,
                forwarded_args,
            )
            max_parallel = (
                launcher_args.max_parallel
                if launcher_args.max_parallel > 0
                else _default_parallelism(jobs)
            )
        return run_jobs(
            jobs,
            max_parallel,
            dry_run=launcher_args.dry_run,
            keep_going=launcher_args.keep_going,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
