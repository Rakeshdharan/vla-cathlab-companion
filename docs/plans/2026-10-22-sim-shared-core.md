# Simulation Shared Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulator-independent shared core:
- the variation spec and `EpisodeConfig` sampler;
- the scripted expert;
- a kinematic fake simulator;
- the generation driver and LeRobot recorder.

The Isaac Sim and robosuite tracks plug into this core by implementing one small adapter each.

**Architecture:** One seeded `VariationSpec` (`configs/variation/cathlab_v1.yaml`) produces the same `EpisodeConfig` list for both simulators, so episodes are paired. The expert core plans waypoints from privileged object poses with per-class grasp templates and emits **normalized 7-D EEF-delta actions** (LIBERO / robosuite `OSC_POSE` convention). A simulator only needs to implement the `ExpertSim` protocol. `generate_scripted` keeps successful episodes and writes them through an `EpisodeSink`: `MemorySink` for tests, `LeRobotSink` for real data.

**Tech Stack:** Python ≥3.10, `pydantic>=2`, `numpy`, `pyyaml`, `pytest`; `lerobot` 0.5.x for `LeRobotSink` only (optional, lazy import).

Spec: [`docs/specs/2026-10-22-sim-framework-design.md`](../specs/2026-10-22-sim-framework-design.md)

## Global Constraints

- **Prerequisite:** `contracts/` (foundation plan Tasks 1–3) is merged on `main`. This plan imports `contracts.StepInfo`, `STAGES`, `TaskSpec`, `DatasetManifest`, `SceneSpec`, `SimBackend`, `load_yaml`, `save_yaml`.
- Robot: single-arm Franka Panda + Franka hand. Robot base frame: x forward, y left, z up; table top at z = 0.
- Action: `float32[7]` **normalized to [-1, 1]**: `[dx, dy, dz]/0.05 m`, `[drx, dry, drz]/0.5 rad` (axis-angle, base frame), gripper `-1` = open, `+1` = close.
- State: `float32[8]` = EEF position (3), EEF axis-angle (3), finger joints (2).
- Cameras: `observation.images.agentview`, `observation.images.eye_in_hand`, 256×256 RGB. Control and recording at 20 Hz. Max 300 steps per episode.
- Quaternions are `[w, x, y, z]`; poses are `[x, y, z, qw, qx, qy, qz]`.
- Task objects: exactly `bottle`, `vial`, `syringe`. Train grid = 3 × 8 layouts × 4 object colours × 4 table colours = 384. Held-out: layouts 8–9, object colour 4, table colour 4.
- Dataset `source` = `scripted`; manifest at `<dataset_root>/manifest.yaml`; per-episode metadata at `<dataset_root>/meta/episodes_extra.jsonl`.
- Tests run without GPU, simulator or network. The LeRobot test skips if `lerobot` is missing.
- Commits are authored by the human teammate only (no AI `Co-authored-by` trailers).

## File Structure

| File | Responsibility |
|---|---|
| `configs/variation/cathlab_v1.yaml` | Scene geometry, object specs, layouts seed, colour palettes, train/held-out splits, physics/lighting ranges |
| `configs/tasks/cathlab_pick.yaml` | `TaskSpec` for "Hand me the {object}" with the 3 objects |
| `dataengine/__init__.py` | Package marker (empty) |
| `dataengine/variation.py` | `VariationSpec`, `ObjectSpec`, `EpisodeConfig`, `layouts`, `object_poses`, `train_configs`, `eval_configs`, `TIERS` |
| `dataengine/geometry.py` | Quaternion helpers |
| `dataengine/expert.py` | `ExpertSim` protocol, grasp templates, waypoint planner, action law, `run_expert_episode` |
| `dataengine/fake_sim.py` | `FakeExpertSim`: kinematic simulator implementing `ExpertSim` + `SimBackend` |
| `dataengine/generate.py` | `EpisodeSink`, `MemorySink`, `build_frame`, `generate_scripted`, `GenerationReport` |
| `dataengine/lerobot_sink.py` | `LeRobotSink`, `panda_libero_features` |
| `tests/dataengine/test_*.py` | Tests per module |

---

### Task 1: Variation spec, task config and EpisodeConfig sampler

**Files:**
- Create: `configs/variation/cathlab_v1.yaml`, `configs/tasks/cathlab_pick.yaml`, `dataengine/__init__.py` (empty), `dataengine/variation.py`
- Test: `tests/dataengine/test_variation.py`

**Interfaces:**
- Consumes: nothing beyond pydantic/numpy/pyyaml.
- Produces:
  - `VariationSpec.load(path) -> VariationSpec` (fields `scene`, `objects: dict[str, ObjectSpec]`, splits, ranges);
  - `ObjectSpec.center_z() -> float`, `ObjectSpec.half_extents() -> np.ndarray(3)`;
  - `EpisodeConfig(object, layout_id, obj_color_id, table_color_id, mass_scale, friction, light_intensity, light_temp_k, seed, tier)`;
  - `layouts(spec) -> list[dict[str, (x, y, yaw)]]`;
  - `object_poses(spec, cfg) -> dict[str, np.ndarray(7)]`;
  - `train_configs(spec, seed=0) -> list[EpisodeConfig]` (384);
  - `eval_configs(spec, tier, seeds) -> list[EpisodeConfig]`;
  - `TIERS`.

- [ ] **Step 1: Write the failing test** `tests/dataengine/test_variation.py`

```python
from pathlib import Path
import numpy as np
import pytest
from dataengine.variation import TIERS, VariationSpec, eval_configs, layouts, object_poses, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


def test_spec_loads_three_objects_and_five_palettes():
    assert list(SPEC.objects) == ["bottle", "vial", "syringe"]
    assert len(SPEC.obj_colors) == 5 and len(SPEC.table_colors) == 5


def test_layouts_are_deterministic_separated_and_in_region():
    a, b = layouts(SPEC), layouts(SPEC)
    assert a == b and len(a) == 10
    cx, cy = SPEC.scene.pick_region_center
    sx, sy = SPEC.scene.pick_region_size
    for lay in a:
        pts = np.array([p[:2] for p in lay.values()])
        assert np.all(np.abs(pts[:, 0] - cx) <= sx / 2) and np.all(np.abs(pts[:, 1] - cy) <= sy / 2)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(3, 1)]
        assert d.min() >= SPEC.scene.min_separation


def test_train_grid_is_384_and_uses_only_train_values():
    cfgs = train_configs(SPEC)
    assert len(cfgs) == 3 * 8 * 4 * 4 == 384
    assert {c.layout_id for c in cfgs} == set(SPEC.layout_split.train)
    assert {c.obj_color_id for c in cfgs} == set(SPEC.obj_color_split.train)
    assert {c.table_color_id for c in cfgs} == set(SPEC.table_color_split.train)
    assert all(0.7 <= c.mass_scale <= 1.3 and 0.5 <= c.friction <= 1.0 for c in cfgs)
    assert train_configs(SPEC) == cfgs


def test_eval_tiers_hold_out_the_right_axes():
    seeds = [1000, 1001, 1002, 1003]
    allh = eval_configs(SPEC, "all_held_out", seeds)
    assert len(allh) == 12
    assert all(c.layout_id in (8, 9) and c.obj_color_id == 4 and c.table_color_id == 4 for c in allh)
    ind = eval_configs(SPEC, "in_distribution", seeds)
    assert all(c.layout_id < 8 and c.obj_color_id < 4 and c.table_color_id < 4 for c in ind)
    lay = eval_configs(SPEC, "held_out_layout", seeds)
    assert all(c.layout_id in (8, 9) and c.obj_color_id < 4 for c in lay)
    assert len(TIERS) == 5


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        eval_configs(SPEC, "nope", [1])


def test_object_poses_rest_on_table():
    poses = object_poses(SPEC, train_configs(SPEC)[0])
    assert poses["bottle"][2] == pytest.approx(0.125)
    assert poses["syringe"][2] == pytest.approx(0.0075)
    assert all(abs(np.linalg.norm(p[3:]) - 1) < 1e-9 for p in poses.values())
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/dataengine/test_variation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataengine'`

- [ ] **Step 3: Create `configs/variation/cathlab_v1.yaml`**

```yaml
variation_id: cathlab_v1

# Robot base frame: x forward, y left, z up; table top at z = 0.
scene:
  table_size: [0.8, 1.2]
  tray_center: [0.55, -0.30]
  tray_inner: [0.25, 0.32]
  tray_floor_z: 0.01
  pick_region_center: [0.55, 0.15]
  pick_region_size: [0.30, 0.40]
  min_separation: 0.12

objects:
  bottle:  {shape: cylinder, radius: 0.031,  length: 0.250, upright: true,  mass: 0.30,  grasp: side}
  vial:    {shape: cylinder, radius: 0.0135, length: 0.042, upright: true,  mass: 0.02,  grasp: top}
  syringe: {shape: syringe,  radius: 0.0075, length: 0.140, upright: false, mass: 0.015, grasp: top}

n_layouts: 10
layout_seed: 1234

# Five palettes; each maps object -> RGBA.
obj_colors:
  - {bottle: [0.85, 0.85, 0.90, 1.0], vial: [0.55, 0.30, 0.10, 1.0], syringe: [0.95, 0.95, 0.95, 1.0]}
  - {bottle: [0.20, 0.45, 0.85, 1.0], vial: [0.90, 0.80, 0.20, 1.0], syringe: [0.30, 0.75, 0.40, 1.0]}
  - {bottle: [0.80, 0.25, 0.25, 1.0], vial: [0.25, 0.60, 0.85, 1.0], syringe: [0.95, 0.60, 0.20, 1.0]}
  - {bottle: [0.35, 0.70, 0.45, 1.0], vial: [0.75, 0.35, 0.75, 1.0], syringe: [0.20, 0.35, 0.80, 1.0]}
  - {bottle: [0.95, 0.75, 0.30, 1.0], vial: [0.15, 0.15, 0.15, 1.0], syringe: [0.85, 0.20, 0.45, 1.0]}

table_colors:
  - [0.80, 0.80, 0.80, 1.0]
  - [0.55, 0.40, 0.25, 1.0]
  - [0.30, 0.45, 0.60, 1.0]
  - [0.90, 0.90, 0.85, 1.0]
  - [0.25, 0.30, 0.25, 1.0]

layout_split:      {train: [0, 1, 2, 3, 4, 5, 6, 7], held_out: [8, 9]}
obj_color_split:   {train: [0, 1, 2, 3], held_out: [4]}
table_color_split: {train: [0, 1, 2, 3], held_out: [4]}

mass_scale: [0.7, 1.3]
friction: [0.5, 1.0]
light_intensity: [0.7, 1.3]
light_temp_k: [4000.0, 7500.0]
```

- [ ] **Step 4: Create `configs/tasks/cathlab_pick.yaml`**

```yaml
task_id: cathlab_pick
instruction_templates:
  - "Hand me the {object}"
  - "Please pass the {object}"
  - "Can you give me the {object}"
objects: [bottle, vial, syringe]
success_predicate: "target object at rest inside the tray and gripper open"
stage_predicates:
  reach: "end-effector within 5 cm of the target grasp point"
  grasp: "target attached to / held by the gripper"
  lift: "target bottom > table top + 5 cm"
  transport: "target xy inside the tray inner bounds"
  place: "target released and at rest inside the tray"
```

- [ ] **Step 5: Create `dataengine/variation.py`**

```python
import itertools
from pathlib import Path
from typing import Literal
import numpy as np
import yaml
from pydantic import BaseModel

TIERS: tuple[str, ...] = (
    "in_distribution",
    "held_out_layout",
    "held_out_obj_color",
    "held_out_table_color",
    "all_held_out",
)


class SceneGeometry(BaseModel):
    table_size: tuple[float, float]
    tray_center: tuple[float, float]
    tray_inner: tuple[float, float]
    tray_floor_z: float
    pick_region_center: tuple[float, float]
    pick_region_size: tuple[float, float]
    min_separation: float


class ObjectSpec(BaseModel):
    shape: Literal["cylinder", "syringe"]
    radius: float
    length: float
    upright: bool
    mass: float
    grasp: Literal["side", "top"]

    def center_z(self) -> float:
        return self.length / 2 if self.upright else self.radius

    def half_extents(self) -> np.ndarray:
        """Half sizes in the object frame (long axis = z if upright, x if lying)."""
        if self.upright:
            return np.array([self.radius, self.radius, self.length / 2])
        return np.array([self.length / 2, self.radius, self.radius])


class AxisSplit(BaseModel):
    train: list[int]
    held_out: list[int]


class VariationSpec(BaseModel):
    variation_id: str
    scene: SceneGeometry
    objects: dict[str, ObjectSpec]
    n_layouts: int
    layout_seed: int
    obj_colors: list[dict[str, list[float]]]
    table_colors: list[list[float]]
    layout_split: AxisSplit
    obj_color_split: AxisSplit
    table_color_split: AxisSplit
    mass_scale: tuple[float, float]
    friction: tuple[float, float]
    light_intensity: tuple[float, float]
    light_temp_k: tuple[float, float]

    @classmethod
    def load(cls, path: str | Path) -> "VariationSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class EpisodeConfig(BaseModel):
    object: str
    layout_id: int
    obj_color_id: int
    table_color_id: int
    mass_scale: float
    friction: float
    light_intensity: float
    light_temp_k: float
    seed: int
    tier: str


def layouts(spec: VariationSpec) -> list[dict[str, tuple[float, float, float]]]:
    """Deterministic layouts: object -> (x, y, yaw), non-overlapping inside the pick region."""
    cx, cy = spec.scene.pick_region_center
    sx, sy = spec.scene.pick_region_size
    out = []
    for i in range(spec.n_layouts):
        rng = np.random.default_rng(spec.layout_seed + i)
        while True:
            placed: dict[str, tuple[float, float, float]] = {}
            for name, obj in spec.objects.items():
                margin = obj.length / 2 if not obj.upright else obj.radius
                x = rng.uniform(cx - sx / 2 + margin, cx + sx / 2 - margin)
                y = rng.uniform(cy - sy / 2 + margin, cy + sy / 2 - margin)
                placed[name] = (float(x), float(y), float(rng.uniform(-np.pi, np.pi)))
            pts = np.array([p[:2] for p in placed.values()])
            dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(len(pts), 1)]
            if dists.min() >= spec.scene.min_separation:
                out.append(placed)
                break
    return out


def object_poses(spec: VariationSpec, cfg: EpisodeConfig) -> dict[str, np.ndarray]:
    """Initial object centre poses [x, y, z, qw, qx, qy, qz] (yaw about z) for a config."""
    layout = layouts(spec)[cfg.layout_id]
    poses = {}
    for name, (x, y, yaw) in layout.items():
        z = spec.objects[name].center_z()
        poses[name] = np.array([x, y, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return poses


def _continuous(spec: VariationSpec, rng: np.random.Generator) -> dict[str, float]:
    return {
        "mass_scale": float(rng.uniform(*spec.mass_scale)),
        "friction": float(rng.uniform(*spec.friction)),
        "light_intensity": float(rng.uniform(*spec.light_intensity)),
        "light_temp_k": float(rng.uniform(*spec.light_temp_k)),
    }


def train_configs(spec: VariationSpec, seed: int = 0) -> list[EpisodeConfig]:
    """Full training grid: objects × train layouts × train object colours × train table colours."""
    grid = itertools.product(
        spec.objects, spec.layout_split.train, spec.obj_color_split.train, spec.table_color_split.train
    )
    out = []
    for i, (obj, layout, oc, tc) in enumerate(grid):
        rng = np.random.default_rng(seed * 1_000_003 + i)
        out.append(EpisodeConfig(object=obj, layout_id=layout, obj_color_id=oc, table_color_id=tc,
                                 seed=seed * 1_000_003 + i, tier="train", **_continuous(spec, rng)))
    return out


def eval_configs(spec: VariationSpec, tier: str, seeds: list[int]) -> list[EpisodeConfig]:
    """Eval episodes for one tier: every object × every seed. Seeds should not overlap training seeds."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    held_layout = tier in ("held_out_layout", "all_held_out")
    held_oc = tier in ("held_out_obj_color", "all_held_out")
    held_tc = tier in ("held_out_table_color", "all_held_out")
    lay = spec.layout_split.held_out if held_layout else spec.layout_split.train
    oc = spec.obj_color_split.held_out if held_oc else spec.obj_color_split.train
    tc = spec.table_color_split.held_out if held_tc else spec.table_color_split.train
    out = []
    for obj in spec.objects:
        for s in seeds:
            rng = np.random.default_rng(s)
            out.append(EpisodeConfig(object=obj, layout_id=lay[s % len(lay)], obj_color_id=oc[s % len(oc)],
                                     table_color_id=tc[s % len(tc)], seed=s, tier=tier, **_continuous(spec, rng)))
    return out
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest tests/dataengine/test_variation.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add configs/variation configs/tasks/cathlab_pick.yaml dataengine/__init__.py dataengine/variation.py tests/dataengine/test_variation.py
git commit -m "feat(dataengine): variation spec and EpisodeConfig sampler"
```

### Task 2: Quaternion geometry helpers

**Files:**
- Create: `dataengine/geometry.py`
- Test: `tests/dataengine/test_geometry.py`

**Interfaces:**
- Produces: `quat_mul(a, b)`, `quat_conj(q)`, `quat_from_axis_angle(v)`, `quat_to_axis_angle(q)`, `quat_rotate(q, v)`, `yaw_quat(yaw)`, `quat_yaw(q) -> float`, `rot_error(q_target, q_current) -> axis-angle (3,)`. All quaternions are `[w, x, y, z]` numpy arrays.

- [ ] **Step 1: Write the failing test** `tests/dataengine/test_geometry.py`

```python
import numpy as np
import pytest
from dataengine.geometry import (quat_from_axis_angle, quat_mul, quat_rotate, quat_to_axis_angle, quat_yaw,
                                 rot_error, yaw_quat)


def test_axis_angle_roundtrip():
    v = np.array([0.1, -0.4, 0.3])
    np.testing.assert_allclose(quat_to_axis_angle(quat_from_axis_angle(v)), v, atol=1e-9)


def test_rotate_x_by_90_maps_z_to_minus_y():
    q = quat_from_axis_angle(np.array([np.pi / 2, 0, 0]))
    np.testing.assert_allclose(quat_rotate(q, np.array([0, 0, 1.0])), [0, -1, 0], atol=1e-9)


def test_yaw_roundtrip():
    assert quat_yaw(yaw_quat(0.7)) == pytest.approx(0.7)


def test_rot_error_takes_current_to_target():
    cur = yaw_quat(0.2)
    tgt = quat_mul(quat_from_axis_angle(np.array([0, 0.3, 0])), cur)
    err = rot_error(tgt, cur)
    np.testing.assert_allclose(quat_mul(quat_from_axis_angle(err), cur), tgt, atol=1e-9)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/dataengine/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataengine.geometry'`

- [ ] **Step 3: Create `dataengine/geometry.py`**

```python
"""Quaternion helpers. Convention: q = [w, x, y, z]; poses = [x, y, z, w, qx, qy, qz]."""
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(v) / angle
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.zeros(3)
    return 2 * np.arctan2(s, q[0]) * q[1:] / s


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Axis-angle (base frame) rotating q_current onto q_target."""
    return quat_to_axis_angle(quat_mul(q_target, quat_conj(q_current)))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/dataengine/test_geometry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dataengine/geometry.py tests/dataengine/test_geometry.py
git commit -m "feat(dataengine): quaternion helpers"
```

### Task 3: Scripted expert core and kinematic fake simulator

**Files:**
- Create: `dataengine/expert.py`, `dataengine/fake_sim.py`
- Test: `tests/dataengine/test_expert.py`

**Interfaces:**
- Consumes: `contracts.StepInfo`, `STAGES`, `SceneSpec`, `SimBackend`; Task 1 `VariationSpec`, `EpisodeConfig`, `ObjectSpec`, `object_poses`; Task 2 helpers.
- Produces:
  - `ExpertSim` protocol: `reset(cfg) -> obs`, `get_object_pose(name) -> (7,)`, `get_ee_pose() -> (7,)`, `step(action) -> (obs, StepInfo)`, `tray_contains(name) -> bool`. **This is what the Isaac and robosuite adapters implement.** `obs` keys are `state` (8,), `agentview` and `eye_in_hand` (H, W, 3) uint8;
  - `Q_DOWN`, `GRIPPER_OPEN = -1.0`, `GRIPPER_CLOSE = 1.0`;
  - `ExpertParams` dataclass;
  - `Waypoint`, `ExpertResult(success, steps, final_info, waypoints_reached, trace)`;
  - `grasp_pose(obj, obj_pose, q_down) -> (pos, quat, approach)`;
  - `plan_waypoints(spec, target, obj_pose, params, q_down) -> list[Waypoint]`;
  - `waypoint_action(ee_pose, wp, params) -> float32 (7,)`;
  - `run_expert_episode(sim, spec, cfg, params, q_down, record=None) -> ExpertResult`;
  - `FakeExpertSim(spec, max_dpos=0.05, max_drot=0.5, image_size=256)`.

- [ ] **Step 1: Write the failing test** `tests/dataengine/test_expert.py`

```python
from pathlib import Path
import numpy as np
import pytest
from dataengine.expert import (GRIPPER_CLOSE, GRIPPER_OPEN, ExpertParams, grasp_pose, plan_waypoints,
                               run_expert_episode, waypoint_action)
from dataengine.fake_sim import FakeExpertSim
from dataengine.geometry import quat_rotate
from dataengine.variation import VariationSpec, eval_configs, object_poses, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
CFG = train_configs(SPEC)[0]


def test_side_grasp_for_bottle_approaches_along_plus_y_below_centre():
    pose = object_poses(SPEC, CFG)["bottle"]
    pos, quat, approach = grasp_pose(SPEC.objects["bottle"], pose)
    np.testing.assert_allclose(approach, [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(quat_rotate(quat, np.array([0, 0, 1.0])), approach, atol=1e-9)
    assert pos[2] == pytest.approx(pose[2] - 0.025)


def test_top_grasp_for_syringe_points_down_at_centre():
    pose = object_poses(SPEC, CFG)["syringe"]
    pos, quat, approach = grasp_pose(SPEC.objects["syringe"], pose)
    np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(quat_rotate(quat, np.array([0, 0, 1.0])), [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(pos, pose[:3])


def test_waypoint_sequence_and_gripper_commands():
    wps = plan_waypoints(SPEC, "vial", object_poses(SPEC, CFG)["vial"])
    assert [w.name for w in wps] == ["pre_grasp", "grasp", "close", "lift", "above_tray", "lower", "open", "retreat"]
    assert [w.gripper for w in wps] == [GRIPPER_OPEN] * 2 + [GRIPPER_CLOSE] * 4 + [GRIPPER_OPEN] * 2
    assert np.allclose(wps[5].pos[:2], SPEC.scene.tray_center)


def test_actions_are_normalized_and_clipped():
    wps = plan_waypoints(SPEC, "vial", object_poses(SPEC, CFG)["vial"])
    a = waypoint_action(np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0]), wps[0])
    assert a.shape == (7,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)


@pytest.mark.parametrize("obj", ["bottle", "vial", "syringe"])
def test_expert_succeeds_in_fake_sim_for_every_object(obj):
    cfgs = [c for c in train_configs(SPEC) if c.object == obj][:16]
    sim = FakeExpertSim(SPEC)
    results = [run_expert_episode(sim, SPEC, c) for c in cfgs]
    assert all(r.success for r in results), [r.trace for r in results if not r.success]
    assert all(r.steps < ExpertParams().max_steps for r in results)


def test_expert_succeeds_on_held_out_tier():
    sim = FakeExpertSim(SPEC)
    assert all(run_expert_episode(sim, SPEC, c).success for c in eval_configs(SPEC, "all_held_out", [5000, 5001]))


def test_record_callback_gets_every_step():
    seen = []
    r = run_expert_episode(FakeExpertSim(SPEC), SPEC, CFG, record=lambda o, a: seen.append((o["state"].shape, a.shape)))
    assert len(seen) == r.steps and seen[0] == ((8,), (7,))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/dataengine/test_expert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataengine.expert'`

- [ ] **Step 3: Create `dataengine/expert.py`**

```python
"""Sim-agnostic scripted expert: grasp templates -> waypoints -> normalized 7-D EEF-delta actions."""
from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np
from contracts import StepInfo
from dataengine.geometry import quat_from_axis_angle, quat_mul, quat_rotate, quat_yaw, rot_error, yaw_quat
from dataengine.variation import EpisodeConfig, ObjectSpec, VariationSpec

# Panda hand pointing straight down (approach axis = -z base): 180° about x.
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0


class ExpertSim(Protocol):
    """What each simulator adapter must provide to run the expert. Obs must contain
    'state' (8,), 'agentview' (256,256,3) uint8 and 'eye_in_hand' (256,256,3) uint8."""

    def reset(self, cfg: EpisodeConfig) -> dict: ...
    def get_object_pose(self, name: str) -> np.ndarray: ...   # [x,y,z,qw,qx,qy,qz], object centre, base frame
    def get_ee_pose(self) -> np.ndarray: ...                  # [x,y,z,qw,qx,qy,qz], fingertip centre (TCP)
    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]: ...
    def tray_contains(self, name: str) -> bool: ...


@dataclass
class ExpertParams:
    max_dpos: float = 0.05     # metres per unit action (robosuite OSC_POSE default output_max)
    max_drot: float = 0.5      # radians per unit action
    pos_tol: float = 0.008
    rot_tol: float = 0.08
    close_steps: int = 10
    open_steps: int = 8
    max_steps: int = 300
    approach_offset: float = 0.10
    lift_height: float = 0.15
    tray_clearance: float = 0.20
    release_height: float = 0.02


@dataclass
class Waypoint:
    pos: np.ndarray
    quat: np.ndarray
    gripper: float
    hold: int = 0      # extra steps to stay here after reaching it (lets the gripper actuate)
    name: str = ""


@dataclass
class ExpertResult:
    success: bool
    steps: int
    final_info: StepInfo | None
    waypoints_reached: int
    trace: list[str] = field(default_factory=list)


def grasp_pose(obj: ObjectSpec, obj_pose: np.ndarray, q_down: np.ndarray = Q_DOWN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tcp_position, tcp_quat, approach_direction) for the object's grasp template."""
    centre, yaw = obj_pose[:3], quat_yaw(obj_pose[3:])
    if obj.grasp == "side":
        # Side grasp at 40% of the height, approaching along +y, fingers closing horizontally.
        pos = centre + np.array([0.0, 0.0, -0.1 * obj.length])
        quat = quat_mul(quat_from_axis_angle(np.array([np.pi / 2, 0.0, 0.0])), q_down)
        return pos, quat, quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    # Top-down grasp; fingers close perpendicular to the object's long (local x) axis.
    quat = quat_mul(yaw_quat(yaw + np.pi / 2), q_down) if not obj.upright else quat_mul(yaw_quat(yaw), q_down)
    return centre.copy(), quat, np.array([0.0, 0.0, -1.0])


def plan_waypoints(spec: VariationSpec, target: str, obj_pose: np.ndarray, params: ExpertParams = ExpertParams(),
                   q_down: np.ndarray = Q_DOWN) -> list[Waypoint]:
    obj = spec.objects[target]
    g_pos, g_quat, approach = grasp_pose(obj, obj_pose, q_down)
    bottom_z = obj_pose[2] - obj.center_z()
    grasp_above_bottom = g_pos[2] - bottom_z
    tray = np.array([*spec.scene.tray_center, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    place_z = spec.scene.tray_floor_z + params.release_height + grasp_above_bottom
    above_tray = tray + up * (place_z + params.tray_clearance)
    lowered = tray + up * place_z
    return [
        Waypoint(g_pos - approach * params.approach_offset, g_quat, GRIPPER_OPEN, name="pre_grasp"),
        Waypoint(g_pos, g_quat, GRIPPER_OPEN, name="grasp"),
        Waypoint(g_pos, g_quat, GRIPPER_CLOSE, hold=params.close_steps, name="close"),
        Waypoint(g_pos + up * params.lift_height, g_quat, GRIPPER_CLOSE, name="lift"),
        Waypoint(above_tray, g_quat, GRIPPER_CLOSE, name="above_tray"),
        Waypoint(lowered, g_quat, GRIPPER_CLOSE, name="lower"),
        Waypoint(lowered, g_quat, GRIPPER_OPEN, hold=params.open_steps, name="open"),
        Waypoint(lowered + up * params.lift_height, g_quat, GRIPPER_OPEN, name="retreat"),
    ]


def waypoint_action(ee_pose: np.ndarray, wp: Waypoint, params: ExpertParams = ExpertParams()) -> np.ndarray:
    """Normalized 7-D action in [-1, 1]: [dpos/max_dpos, drot/max_drot, gripper]."""
    dpos = np.clip((wp.pos - ee_pose[:3]) / params.max_dpos, -1.0, 1.0)
    drot = np.clip(rot_error(wp.quat, ee_pose[3:]) / params.max_drot, -1.0, 1.0)
    return np.concatenate([dpos, drot, [wp.gripper]]).astype(np.float32)


def _reached(ee_pose: np.ndarray, wp: Waypoint, params: ExpertParams) -> bool:
    return (np.linalg.norm(wp.pos - ee_pose[:3]) < params.pos_tol
            and np.linalg.norm(rot_error(wp.quat, ee_pose[3:])) < params.rot_tol)


def run_expert_episode(sim: ExpertSim, spec: VariationSpec, cfg: EpisodeConfig,
                       params: ExpertParams = ExpertParams(), q_down: np.ndarray = Q_DOWN,
                       record: Callable[[dict, np.ndarray], None] | None = None) -> ExpertResult:
    """Run one scripted episode. `record(obs, action)` is called before every step with the obs the action was chosen from."""
    obs = sim.reset(cfg)
    waypoints = plan_waypoints(spec, cfg.object, sim.get_object_pose(cfg.object), params, q_down)
    steps, idx, held, info, trace = 0, 0, 0, None, []
    while idx < len(waypoints) and steps < params.max_steps:
        wp = waypoints[idx]
        action = waypoint_action(sim.get_ee_pose(), wp, params)
        if record is not None:
            record(obs, action)
        obs, info = sim.step(action)
        steps += 1
        if _reached(sim.get_ee_pose(), wp, params):
            if held >= wp.hold:
                trace.append(wp.name)
                idx, held = idx + 1, 0
            else:
                held += 1
    success = bool(info is not None and info.success and sim.tray_contains(cfg.object))
    return ExpertResult(success=success, steps=steps, final_info=info, waypoints_reached=idx, trace=trace)
```

- [ ] **Step 4: Create `dataengine/fake_sim.py`**

```python
"""Kinematic fake simulator implementing ExpertSim + SimBackend for GPU-free tests and pipeline smoke runs."""
import numpy as np
from contracts import STAGES, SceneSpec, SimBackend, StepInfo
from dataengine.expert import Q_DOWN
from dataengine.geometry import quat_conj, quat_from_axis_angle, quat_mul, quat_rotate, quat_to_axis_angle
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses

HOME_POSE = np.array([0.45, 0.0, 0.40, *Q_DOWN])


class FakeExpertSim(SimBackend):
    """Each action moves the TCP exactly by action*max (metres/radians). A closed gripper picks up the target
    when the TCP is inside its bounding box (+2 cm); opening drops it straight down onto the tray or table."""

    def __init__(self, spec: VariationSpec, max_dpos: float = 0.05, max_drot: float = 0.5, image_size: int = 256):
        self.spec, self.max_dpos, self.max_drot, self.image_size = spec, max_dpos, max_drot, image_size
        self.cfg: EpisodeConfig | None = None

    # ExpertSim -----------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig, *_, **__) -> dict:
        if isinstance(cfg, SceneSpec):
            raise TypeError("FakeExpertSim.reset expects an EpisodeConfig")
        self.cfg = cfg
        self.ee = HOME_POSE.copy()
        self.closed = False
        self.poses = object_poses(self.spec, cfg)
        self.attached: str | None = None
        self.offset: np.ndarray | None = None
        self.released_in_tray: set[str] = set()
        self.reached = False
        self.lifted = False
        return self._obs()

    def get_object_pose(self, name: str) -> np.ndarray:
        return self.poses[name].copy()

    def get_ee_pose(self) -> np.ndarray:
        return self.ee.copy()

    def tray_contains(self, name: str) -> bool:
        return name in self.released_in_tray

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        a = np.asarray(action, dtype=float)
        self.ee[:3] += a[:3] * self.max_dpos
        self.ee[3:] = quat_mul(quat_from_axis_angle(a[3:6] * self.max_drot), self.ee[3:])
        target = self.cfg.object
        if np.linalg.norm(self.ee[:3] - self.poses[target][:3]) < 0.05 + self.spec.objects[target].length / 2:
            self.reached = True
        want_closed = a[6] > 0
        if want_closed and not self.closed:
            self._try_grasp()
        if not want_closed and self.closed and self.attached:
            self._release()
        self.closed = want_closed
        if self.attached:
            self.poses[self.attached][:3] = self.ee[:3] + self.offset
            bottom = self.poses[self.attached][2] - self.spec.objects[self.attached].center_z()
            self.lifted = self.lifted or bottom > 0.05
        return self._obs(), self._info()

    # SimBackend (render for policies) --------------------------------------------
    def render(self) -> np.ndarray:
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    # internals ---------------------------------------------------------------------
    def _try_grasp(self) -> None:
        for name, pose in self.poses.items():
            local = quat_rotate(quat_conj(pose[3:]), self.ee[:3] - pose[:3])
            if np.all(np.abs(local) <= self.spec.objects[name].half_extents() + 0.02):
                self.attached, self.offset = name, pose[:3] - self.ee[:3]
                return

    def _in_tray_xy(self, xy: np.ndarray) -> bool:
        c, s = np.array(self.spec.scene.tray_center), np.array(self.spec.scene.tray_inner)
        return bool(np.all(np.abs(xy - c) <= s / 2 - 0.01))

    def _release(self) -> None:
        name, obj = self.attached, self.spec.objects[self.attached]
        pose = self.poses[name]
        in_tray = self._in_tray_xy(pose[:2])
        pose[2] = (self.spec.scene.tray_floor_z if in_tray else 0.0) + obj.center_z()
        if in_tray:
            self.released_in_tray.add(name)
        self.attached = None

    def _info(self) -> StepInfo:
        t = self.cfg.object
        flags = {
            "reach": bool(self.reached),
            "grasp": self.attached == t or t in self.released_in_tray,
            "lift": bool(self.lifted),
            "transport": self._in_tray_xy(self.poses[t][:2]),
            "place": t in self.released_in_tray,
        }
        return StepInfo(stage_flags=flags, success=all(flags[s] for s in STAGES))

    def _obs(self) -> dict:
        fingers = np.array([0.0, 0.0]) if self.closed else np.array([0.04, 0.04])
        state = np.concatenate([self.ee[:3], quat_to_axis_angle(self.ee[3:]), fingers]).astype(np.float32)
        img = self.render()
        return {"state": state, "agentview": img, "eye_in_hand": img}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/dataengine/test_expert.py -v`
Expected: 9 passed, no warnings

- [ ] **Step 6: Commit**

```bash
git add dataengine/expert.py dataengine/fake_sim.py tests/dataengine/test_expert.py
git commit -m "feat(dataengine): scripted expert core and fake simulator"
```

### Task 4: Generation driver, episode metadata and manifest

**Files:**
- Create: `dataengine/generate.py`
- Test: `tests/dataengine/test_generate.py`

**Interfaces:**
- Consumes: `contracts.DatasetManifest`, `TaskSpec`, `save_yaml`; Task 3 `run_expert_episode`, `ExpertParams`, `ExpertSim`, `Q_DOWN`.
- Produces:
  - `EpisodeSink` protocol: `add_frame(frame)`, `save_episode()`, `discard_episode()`, `finalize()`;
  - `MemorySink`;
  - `build_frame(obs, action, instruction) -> dict` (keys `observation.state`, `observation.images.agentview`, `observation.images.eye_in_hand`, `action`, `task`);
  - `instruction_for(task, cfg) -> str`;
  - `GenerationReport(attempted, succeeded, per_object, success_rate)`;
  - `generate_scripted(sim, spec, task, configs, sink, out_dir, source_sim, dataset_id, version, created_by, params, q_down) -> (GenerationReport, DatasetManifest)`.

- [ ] **Step 1: Write the failing test** `tests/dataengine/test_generate.py`

```python
import json
from pathlib import Path
from contracts import DatasetManifest, TaskSpec, load_yaml
from dataengine.fake_sim import FakeExpertSim
from dataengine.generate import MemorySink, generate_scripted, instruction_for
from dataengine.variation import VariationSpec, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


def test_generate_records_successful_episodes_with_metadata(tmp_path):
    cfgs = train_configs(SPEC)[:6]
    sink = MemorySink()
    report, manifest = generate_scripted(FakeExpertSim(SPEC), SPEC, TASK, cfgs, sink, tmp_path, "fake",
                                         "cathlab/pick_fake", "v1", "test")
    assert report.attempted == 6 and report.succeeded == 6 and sink.finalized
    assert len(sink.episodes) == 6
    frame = sink.episodes[0][0]
    assert set(frame) == {"observation.state", "observation.images.agentview", "observation.images.eye_in_hand",
                          "action", "task"}
    assert frame["action"].shape == (7,) and frame["observation.state"].shape == (8,)
    assert frame["task"] == instruction_for(TASK, cfgs[0])
    lines = [json.loads(l) for l in (tmp_path / "meta/episodes_extra.jsonl").read_text().splitlines()]
    assert [l["episode_index"] for l in lines] == list(range(6)) and lines[0]["source_sim"] == "fake"
    assert load_yaml(tmp_path / "manifest.yaml", DatasetManifest) == manifest
    assert manifest.source == "scripted" and manifest.n_episodes == 6


class _FailingSim(FakeExpertSim):
    def tray_contains(self, name):
        return False


def test_failed_episodes_are_discarded(tmp_path):
    sink = MemorySink()
    report, manifest = generate_scripted(_FailingSim(SPEC), SPEC, TASK, train_configs(SPEC)[:2], sink, tmp_path,
                                         "fake", "cathlab/pick_fake", "v1", "test")
    assert report.succeeded == 0 and sink.episodes == [] and manifest.n_episodes == 0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/dataengine/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataengine.generate'`

- [ ] **Step 3: Create `dataengine/generate.py`**

```python
"""Drive the expert over EpisodeConfigs, record successful episodes into a sink, write metadata + manifest."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import numpy as np
from contracts import DatasetManifest, TaskSpec, save_yaml
from dataengine.expert import ExpertParams, ExpertSim, Q_DOWN, run_expert_episode
from dataengine.variation import EpisodeConfig, VariationSpec


class EpisodeSink(Protocol):
    def add_frame(self, frame: dict) -> None: ...
    def save_episode(self) -> None: ...
    def discard_episode(self) -> None: ...
    def finalize(self) -> None: ...


class MemorySink:
    """In-memory sink for tests and dry runs."""

    def __init__(self):
        self.episodes: list[list[dict]] = []
        self._buf: list[dict] = []
        self.finalized = False

    def add_frame(self, frame: dict) -> None:
        self._buf.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self._buf)
        self._buf = []

    def discard_episode(self) -> None:
        self._buf = []

    def finalize(self) -> None:
        self.finalized = True


def build_frame(obs: dict, action: np.ndarray, instruction: str) -> dict:
    return {
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        "observation.images.agentview": obs["agentview"],
        "observation.images.eye_in_hand": obs["eye_in_hand"],
        "action": np.asarray(action, dtype=np.float32),
        "task": instruction,
    }


def instruction_for(task: TaskSpec, cfg: EpisodeConfig) -> str:
    return task.instructions(cfg.object)[cfg.seed % len(task.instruction_templates)]


@dataclass
class GenerationReport:
    attempted: int = 0
    succeeded: int = 0
    per_object: dict[str, list[int]] = field(default_factory=dict)  # object -> [attempted, succeeded]

    @property
    def success_rate(self) -> float:
        return self.succeeded / max(self.attempted, 1)


def generate_scripted(sim: ExpertSim, spec: VariationSpec, task: TaskSpec, configs: list[EpisodeConfig],
                      sink: EpisodeSink, out_dir: str | Path, source_sim: str, dataset_id: str, version: str,
                      created_by: str, params: ExpertParams = ExpertParams(),
                      q_down: np.ndarray = Q_DOWN) -> tuple[GenerationReport, DatasetManifest]:
    """Keep only successful episodes. Writes <out_dir>/meta/episodes_extra.jsonl and <out_dir>/manifest.yaml."""
    out_dir = Path(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    report = GenerationReport()
    with open(out_dir / "meta" / "episodes_extra.jsonl", "a") as meta:
        for cfg in configs:
            instruction = instruction_for(task, cfg)
            result = run_expert_episode(sim, spec, cfg, params, q_down,
                                        record=lambda o, a: sink.add_frame(build_frame(o, a, instruction)))
            counts = report.per_object.setdefault(cfg.object, [0, 0])
            counts[0] += 1
            report.attempted += 1
            if result.success:
                sink.save_episode()
                counts[1] += 1
                meta.write(json.dumps({"episode_index": report.succeeded, "source_sim": source_sim,
                                       "instruction": instruction, "steps": result.steps,
                                       "config": cfg.model_dump()}) + "\n")
                report.succeeded += 1
            else:
                sink.discard_episode()
    sink.finalize()
    manifest = DatasetManifest(dataset_id=dataset_id, version=version, source="scripted", task_id=task.task_id,
                               scene_id=f"{spec.variation_id}_{source_sim}", n_episodes=report.succeeded,
                               created_by=created_by,
                               notes=f"expert success {report.succeeded}/{report.attempted}; per object {report.per_object}")
    save_yaml(manifest, out_dir / "manifest.yaml")
    return report, manifest
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/dataengine/test_generate.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add dataengine/generate.py tests/dataengine/test_generate.py
git commit -m "feat(dataengine): scripted generation driver with metadata and manifest"
```

### Task 5: LeRobot v3 sink (LIBERO-compatible Panda schema)

**Files:**
- Create: `dataengine/lerobot_sink.py`
- Test: `tests/dataengine/test_lerobot_sink.py`

**Interfaces:**
- Consumes: `lerobot.datasets.lerobot_dataset.LeRobotDataset` (`create`, `add_frame`, `save_episode`, `has_pending_frames`, `clear_episode_buffer`, `finalize`; LeRobot 0.5.x); Task 4 `generate_scripted`.
- Produces: `panda_libero_features(image_size=256) -> dict`, `LeRobotSink(repo_id, root, fps=20, image_size=256)` implementing `EpisodeSink`.

- [ ] **Step 1: Write the failing test** `tests/dataengine/test_lerobot_sink.py`

```python
from pathlib import Path
import pytest

pytest.importorskip("lerobot")

from contracts import TaskSpec, load_yaml  # noqa: E402
from dataengine.fake_sim import FakeExpertSim  # noqa: E402
from dataengine.generate import generate_scripted  # noqa: E402
from dataengine.lerobot_sink import LeRobotSink  # noqa: E402
from dataengine.variation import VariationSpec, train_configs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


def test_lerobot_sink_writes_loadable_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"
    sink = LeRobotSink("local/cathlab_pick_fake", root, fps=20, image_size=64)
    report, _ = generate_scripted(FakeExpertSim(SPEC, image_size=64), SPEC, TASK, train_configs(SPEC)[:2], sink,
                                  root, "fake", "cathlab/pick_fake", "v1", "test")
    ds = LeRobotDataset("local/cathlab_pick_fake", root=root)
    assert ds.num_episodes == report.succeeded == 2 and ds.fps == 20
    item = ds[0]
    assert tuple(item["action"].shape) == (7,) and tuple(item["observation.state"].shape) == (8,)
    assert tuple(item["observation.images.agentview"].shape) == (3, 64, 64)
    assert item["task"].startswith(("Hand me", "Please pass", "Can you give"))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run --extra train pytest tests/dataengine/test_lerobot_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataengine.lerobot_sink'  (without lerobot installed it is SKIPPED instead)`

- [ ] **Step 3: Create `dataengine/lerobot_sink.py`**

```python
"""EpisodeSink backed by a LeRobot v3 dataset (LIBERO-compatible Panda schema). Requires `lerobot`."""
from pathlib import Path

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "finger_left", "finger_right"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def panda_libero_features(image_size: int = 256) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "observation.images.agentview": dict(img),
        "observation.images.eye_in_hand": dict(img),
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


class LeRobotSink:
    def __init__(self, repo_id: str, root: str | Path, fps: int = 20, image_size: int = 256):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=panda_libero_features(image_size),
                                             root=root, robot_type="panda", use_videos=True)

    def add_frame(self, frame: dict) -> None:
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        self.dataset.finalize()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run --extra train pytest tests/dataengine/test_lerobot_sink.py -v`
Expected: 1 passed (writes 2 episodes, reloads them with `LeRobotDataset`, checks shapes and task string)

- [ ] **Step 5: Commit**

```bash
git add dataengine/lerobot_sink.py tests/dataengine/test_lerobot_sink.py
git commit -m "feat(dataengine): LeRobot v3 sink for Panda LIBERO schema"
```

### Task 6: Full check and push

- [ ] **Step 1:** Run `uv run pytest -q`. Expected: all tests pass; the LeRobot test is skipped unless the `train` extra is installed.
- [ ] **Step 2:** Run a smoke generation with the fake sim:

```bash
uv run python -c "
from contracts import TaskSpec, load_yaml
from dataengine.variation import VariationSpec, train_configs
from dataengine.fake_sim import FakeExpertSim
from dataengine.generate import MemorySink, generate_scripted
spec = VariationSpec.load('configs/variation/cathlab_v1.yaml')
task = load_yaml('configs/tasks/cathlab_pick.yaml', TaskSpec)
r, m = generate_scripted(FakeExpertSim(spec), spec, task, train_configs(spec), MemorySink(), '/tmp/fake_gen', 'fake', 'cathlab/pick_fake', 'v0', 'me')
print(r.succeeded, '/', r.attempted)"
```

Expected: `384 / 384`.
- [ ] **Step 3:** Push the branch and open a PR to `main`.

---

## How the simulator tracks use this core

| Track | Owner | Implements |
|---|---|---|
| Isaac Sim (Isaac Lab) | Mitesh | `IsaacExpertSim` (`ExpertSim`) + `IsaacBackend` (`SimBackend`) in `sim/isaac/`; scene/randomiser from `VariationSpec` + `EpisodeConfig`; run `generate_scripted(..., source_sim="isaac")` with `LeRobotSink` |
| robosuite | Rakesh | `RobosuiteExpertSim` + `RobosuiteBackend` in `sim/robosuite/`; same |

Each adapter must:
1. place objects from `object_poses(spec, cfg)` and apply colours, mass × `mass_scale`, `friction` and lights from `cfg`;
2. return poses in the robot base frame with table top at z = 0 (subtract the table height);
3. map the normalized action to its controller (robosuite `OSC_POSE` takes it as-is; Isaac Lab diff-IK scales by 0.05 m / 0.5 rad);
4. return the TCP pose, and pass its own `q_down` (gripper-pointing-down quaternion in that simulator's EEF frame) to `run_expert_episode` / `generate_scripted` if it differs from `Q_DOWN`.

The detailed per-track plans are written by their owners (see `docs/tasks/`).
