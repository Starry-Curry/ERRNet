"""Average two compatible checkpoints without changing model architecture."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a two-checkpoint model soup.")
    parser.add_argument("--ckpt_a", required=True, help="base checkpoint; non-averaged keys are kept from this file")
    parser.add_argument("--ckpt_b", required=True, help="second checkpoint")
    parser.add_argument("--alpha", type=float, required=True, help="theta = (1-alpha) * A + alpha * B")
    parser.add_argument("--out", required=True, help="output checkpoint path")
    parser.add_argument(
        "--module_filter",
        default=None,
        help="optional comma-separated substrings; only matching parameter keys are averaged",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_key(checkpoint: Mapping[str, Any]) -> str | None:
    for key in ("model", "state_dict", "icnn"):
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get(key), Mapping):
            return key
    return None


def _get_state(checkpoint: Mapping[str, Any]) -> tuple[Dict[str, Any], str | None]:
    key = _state_key(checkpoint)
    if key is None:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Unsupported checkpoint format: expected a mapping.")
        return dict(checkpoint), None
    return dict(checkpoint[key]), key


def _matches_filter(name: str, filters: list[str]) -> bool:
    return not filters or any(item in name for item in filters)


def average_states(state_a: Mapping[str, Any], state_b: Mapping[str, Any], alpha: float, filters: list[str]) -> tuple[Dict[str, Any], int, int]:
    import torch

    averaged: Dict[str, Any] = {}
    averaged_count = 0
    skipped_count = 0
    for key, value_a in state_a.items():
        value_b = state_b.get(key)
        if (
            _matches_filter(key, filters)
            and isinstance(value_a, torch.Tensor)
            and isinstance(value_b, torch.Tensor)
            and value_a.shape == value_b.shape
            and value_a.is_floating_point()
            and value_b.is_floating_point()
        ):
            averaged[key] = (1.0 - alpha) * value_a + alpha * value_b
            averaged_count += 1
        else:
            averaged[key] = value_a
            skipped_count += 1
    return averaged, averaged_count, skipped_count


def main() -> None:
    import torch

    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}.")

    ckpt_a_path = Path(args.ckpt_a)
    ckpt_b_path = Path(args.ckpt_b)
    out_path = Path(args.out)
    checkpoint_a = _load_checkpoint(ckpt_a_path)
    checkpoint_b = _load_checkpoint(ckpt_b_path)
    state_a, state_key_a = _get_state(checkpoint_a)
    state_b, _ = _get_state(checkpoint_b)
    filters = [item.strip() for item in str(args.module_filter or "").split(",") if item.strip()]

    averaged_state, averaged_count, skipped_count = average_states(state_a, state_b, args.alpha, filters)
    output = dict(checkpoint_a)
    if state_key_a is None:
        output = averaged_state
    else:
        output[state_key_a] = averaged_state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"saved: {out_path}")
    print(f"averaged tensors: {averaged_count}")
    print(f"kept/skipped keys: {skipped_count}")
    if filters:
        print(f"module_filter: {filters}")


if __name__ == "__main__":
    main()
