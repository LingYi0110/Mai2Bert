from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REQUIRED_KEYS = ("state_dict", "hyper_parameters")

_RESUME_KEYS = ("optimizer_states", "lr_schedulers", "scheduler_states")


def _collect_checkpoints(paths: list[str]) -> list[Path]:
    checkpoints: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            checkpoints.append(path)
        elif path.is_dir():
            checkpoints.extend(sorted(path.rglob("*.ckpt")))
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(path for path in set(checkpoints) if not path.name.endswith(".slim.ckpt"))


def _strip_paths(config: dict[str, object]) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return
    for key in paths:
        paths[key] = f"<{key}>"


def slim_payload(
    payload: dict[str, object],
    *,
    keep_optimizer: bool,
    keep_paths: bool = False,
    keep_config: bool = False,
) -> dict[str, object]:
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"not a Lightning checkpoint (missing {missing})")
    slimmed: dict[str, object] = {key: payload[key] for key in _REQUIRED_KEYS}
    if keep_optimizer:
        for key in _RESUME_KEYS:
            if key in payload:
                slimmed[key] = payload[key]
    hyper_parameters = slimmed.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        return slimmed
    config = hyper_parameters.get("config")
    if keep_config:
        if not keep_paths and isinstance(config, dict):
            _strip_paths(config)
    elif config is not None:
        del hyper_parameters["config"]
    return slimmed


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slim_checkpoint",
        description="Strip training-only state from Lightning checkpoints.",
    )
    parser.add_argument("checkpoints", nargs="+", help="checkpoint files or directories")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory (default: next to each input, as {stem}.slim.ckpt)",
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="retain optimizer/scheduler state so the slimmed checkpoint can resume training",
    )
    parser.add_argument(
        "--keep-paths",
        action="store_true",
        help="keep the original config.paths values (absolute paths, may contain user names) "
        "in hyper_parameters; by default they are replaced with placeholders",
    )
    parser.add_argument(
        "--keep-config",
        action="store_true",
        help="retain the full dumped config in hyper_parameters (paths still scrubbed "
        "unless --keep-paths); by default config is dropped entirely",
    )
    args = parser.parse_args(argv)

    checkpoints = _collect_checkpoints(args.checkpoints)
    if not checkpoints:
        print("no .ckpt files found", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Path, str]] = []
    total_before = 0
    total_after = 0
    for checkpoint in checkpoints:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload is not a dict")
            slimmed = slim_payload(
                payload,
                keep_optimizer=args.keep_optimizer,
                keep_paths=args.keep_paths,
                keep_config=args.keep_config,
            )
            target = (
                output_dir / f"{checkpoint.stem}.slim.ckpt"
                if output_dir is not None
                else checkpoint.with_name(f"{checkpoint.stem}.slim.ckpt")
            )
            torch.save(slimmed, target)
            reloaded = torch.load(target, map_location="cpu", weights_only=False)
            for key in _REQUIRED_KEYS:
                if key not in reloaded:
                    raise ValueError(f"round-trip lost required key {key!r}")
        except Exception as error:
            failures.append((checkpoint, f"{type(error).__name__}: {error}"))
            continue
        before = checkpoint.stat().st_size
        after = target.stat().st_size
        total_before += before
        total_after += after
        print(
            f"{checkpoint.name}: {_format_size(before)} -> {_format_size(after)} "
            f"({100 * (1 - after / before):.1f}% smaller) -> {target}"
        )

    print(
        f"total: {_format_size(total_before)} -> {_format_size(total_after)}"
        + (f" ({100 * (1 - total_after / total_before):.1f}% smaller)" if total_before else "")
    )
    for source, message in failures:
        print(f"failed {source}: {message}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
