"""EpisodeSink that writes raw per-episode .npz files, for simulators whose Python env can't install LeRobot.
`scripts/npz_to_lerobot.py` replays them into a LeRobot dataset through LeRobotSink."""
import json
from pathlib import Path
from typing import Iterator
import numpy as np

ARRAY_KEYS = {
    "observation.state": "state",
    "action": "action",
    "observation.images.agentview": "agentview",
    "observation.images.eye_in_hand": "eye_in_hand",
}


class NpzSink:
    """<root>/episodes/episode_{i:06d}.npz per saved episode, numbered in save order (matches the
    episode_index that generate_scripted writes to <root>/meta/episodes_extra.jsonl)."""

    def __init__(self, root: str | Path):
        self.dir = Path(root) / "episodes"
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"{self.dir} is not empty; use a new output directory per shard")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._buf: list[dict] = []
        self.saved = 0
        self.finalized = False

    def add_frame(self, frame: dict) -> None:
        self._buf.append(frame)

    def save_episode(self) -> None:
        arrays = {short: np.stack([f[key] for f in self._buf]) for key, short in ARRAY_KEYS.items()}
        np.savez_compressed(self.dir / f"episode_{self.saved:06d}.npz", task=self._buf[0]["task"], **arrays)
        self.saved += 1
        self._buf = []

    def discard_episode(self) -> None:
        self._buf = []

    def finalize(self) -> None:
        self.finalized = True


def iter_npz_frames(path: str | Path) -> Iterator[dict]:
    """Yield frames in the build_frame() layout (the dict LeRobotSink.add_frame expects)."""
    with np.load(path) as data:
        arrays = {key: data[short] for key, short in ARRAY_KEYS.items()}
        task = str(data["task"])
    for i in range(len(arrays["action"])):
        yield {**{key: arr[i] for key, arr in arrays.items()}, "task": task}


def validate_shard(shard: str | Path) -> list[str]:
    """Count checks before aggregating: manifest n_episodes == metadata lines == .npz files, and each episode's
    recorded frame count (when present) == its .npz length."""
    shard = Path(shard)
    problems = []
    lines = [json.loads(line) for line in (shard / "meta" / "episodes_extra.jsonl").read_text().splitlines()
             if line.strip()]
    files = sorted((shard / "episodes").glob("episode_*.npz"))
    manifest = shard / "manifest.yaml"
    if manifest.exists():
        import yaml
        n = int(yaml.safe_load(manifest.read_text())["n_episodes"])
        if not n == len(lines) == len(files):
            problems.append(f"{shard.name}: {n} in manifest, {len(lines)} metadata lines, {len(files)} .npz files")
    for meta in lines:
        npz = shard / "episodes" / f"episode_{meta['episode_index']:06d}.npz"
        if not npz.is_file():
            problems.append(f"{shard.name}: missing {npz.name}")
        elif "frames" in meta:
            with np.load(npz) as data:
                if len(data["action"]) != meta["frames"]:
                    problems.append(f"{shard.name}: {npz.name} has {len(data['action'])} frames, "
                                    f"metadata says {meta['frames']}")
    return problems


def merge_raw_shards(shard_dirs: list[str | Path]) -> Iterator[tuple[dict, Path]]:
    """Yield (episode metadata re-indexed across shards, npz path) in the given shard order. Every shard is validated
    first (validate_shard); any count mismatch raises ValueError before a single episode is yielded."""
    shard_dirs = list(map(Path, shard_dirs))
    problems = [p for shard in shard_dirs for p in validate_shard(shard)]
    if problems:
        raise ValueError("raw shards fail the count checks: " + "; ".join(problems))
    index = 0
    for shard in shard_dirs:
        for line in (shard / "meta" / "episodes_extra.jsonl").read_text().splitlines():
            meta = json.loads(line)
            npz = shard / "episodes" / f"episode_{meta['episode_index']:06d}.npz"
            yield {**meta, "episode_index": index, "raw_shard": shard.name}, npz
            index += 1
