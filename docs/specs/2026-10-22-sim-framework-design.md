# Simulation Framework Design: Isaac Sim + robosuite (Panda, LIBERO conventions)

*Status: approved 2026-10-22, about H6.5. Supersedes the robot, scene, DataEngine, model and eval-suite parts of [the original design spec](2026-10-22-cathlab-learning-loop-design.md). The architecture, contracts, the five stages, the loop, versioning and the awareness module still apply.*

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Robot: single-arm Franka Panda** with the standard Franka hand. The FR3 Duo and the provided USD scene are dropped from the training/eval path | Native in robosuite, standard in Isaac Lab, and the embodiment of most public VLA checkpoints and datasets |
| D2 | **Our own simple scene** in both simulators: table + tray + 3 objects | Fast to author identically in two simulators; we control every variation axis |
| D3 | **Task:** *"Hand me the {bottle \| vial \| syringe}"*: pick the named object from the table and place it in the tray | Single-arm, ~10–15 s episodes, easy to script, language-conditioned (3 targets) |
| D4 | **LIBERO-compatible I/O:** 7-D end-effector delta action, 8-D state, `agentview` + `eye_in_hand` at 256×256, 20 Hz | Fine-tune from public LIBERO checkpoints (SmolVLA, π0.5, X-VLA) instead of from scratch |
| D5 | **Both simulators generate data and both are evaluation targets**, driven by one shared variation spec (paired episodes) | Cross-simulator co-training is the measurable claim: better held-out generalisation and a smaller sim2sim gap |
| D6 | **Expert:** privileged analytic grasp templates first, then GraspGen as "expert v2" | Data flowing by H12; GraspGen adds diversity and pitch value without blocking |
| D7 | **Provided FR3 Duo datasets** go only through 01 Ingest and 02 Dataset Intelligence | Shows the pipeline doesn't depend on the robot while keeping the challenge's "prepared demonstrations" in the story |
| D8 | **RoboSimStudio:** prompt → scene showcase plus an optional third, held-out evaluation simulator. Not in the data path | It duplicates robosuite as a MuJoCo-family data source; its unique value is agent-driven scene building |
| D9 | **NVIDIA Cosmos Transfer:** stretch goal on about 50 Isaac episodes | Photoreal augmentation is costly; in-sim randomisation covers visual variation |
| D10 | **Models:** SmolVLA (every version), ACT (v0/v1), one frontier model (π0.5 or X-VLA, chosen by H10) on v1 | Keep the comparison and decide what to drop later |

## 2. Shared I/O contract (identical in both simulators)

| Item | Value |
|---|---|
| Robot | Franka Panda, Franka hand (max opening 80 mm), fixed base |
| Action | `float32[7]` **normalized to [-1, 1]** (LIBERO / robosuite `OSC_POSE` convention): `[Δx, Δy, Δz] / 0.05 m`, `[Δrx, Δry, Δrz] / 0.5 rad` (axis-angle, base frame), gripper (−1 open, +1 close) |
| State | `float32[8]` = end-effector position (3), end-effector axis-angle (3), finger joint positions (2) |
| Cameras | `observation.images.agentview` (third person, looking at the table from the front), `observation.images.eye_in_hand` (wrist); 256×256 RGB |
| Rate | 20 Hz control and recording |
| Controller | robosuite `OSC_POSE` (delta); Isaac Lab `DifferentialIKController` in relative pose mode with matching gains/clipping |
| Episode limit | 300 steps (15 s) |
| Dataset | LeRobot v3; `robot_type: panda`; per-episode `episodes_extra.jsonl` with the `EpisodeConfig`, `source_sim`, success and final placement metrics |

The robot config lives in `configs/robots/panda_libero.yaml`. The 16-D FR3 config remains only for ingesting the provided data.

## 3. Scene and objects

| Element | Spec |
|---|---|
| Table | 0.8 × 1.2 m top, height 0.8 m; colour varies |
| Tray (target) | 0.25 × 0.32 m inner, 0.04 m walls, fixed position at the right of the table |
| Pick region | 0.30 × 0.40 m area left of the tray; all 3 objects present each episode (target + 2 distractors) |
| Bottle | Cylinder Ø62 × 250 mm (cap Ø30 × 20 mm); nominal mass 0.30 kg |
| Vial | Cylinder Ø27 × 42 mm; nominal mass 0.02 kg |
| Syringe | Barrel cylinder Ø15 × 100 mm + plunger Ø8 × 40 mm + flange 30 × 10 × 3 mm; nominal mass 0.015 kg |
| Lights | 1 dome + 1 directional; intensity and colour temperature vary |

Objects start as **primitive composites** with identical dimensions in USD and MJCF. Meshes can replace them later without changing anything else.

## 4. Variation spec (shared, seeded)

`configs/variation/cathlab_v1.yaml` lists the axis values. `variation.sample(split, n, seed) -> list[EpisodeConfig]` turns it into episode configurations.

| Axis | Values | Train | Held-out |
|---|---|---|---|
| target object | bottle, vial, syringe | all 3 | — |
| pose layout | 10 seeded layouts (positions + yaw of all 3 objects in the pick region) | 8 | 2 |
| object colour palette | 5 | 4 | 1 |
| table colour | 5 | 4 | 1 |
| mass scale | U(0.7, 1.3) per episode | continuous | — |
| friction | U(0.5, 1.0) per episode | continuous | — |
| lighting | intensity ×U(0.7, 1.3), colour temperature U(4000, 7500) K | continuous | — |

- **Training grid:** 3 × 8 × 4 × 4 = **384 configurations per simulator**. The same list is used in both simulators, so each pair of episodes differs only in the simulator.
- **`EpisodeConfig`:** `object, layout_id, obj_color_id, table_color_id, mass_scale, friction, light_intensity, light_temp_k, seed, split`.

## 5. Expert (sim-agnostic core + thin sim adapter)

```
EpisodeConfig ─► sim.reset(cfg) ─► obj_pose = sim.get_object_pose(target)   (privileged)
               ─► grasp = template[class](obj_pose)                         (phase 2: GraspGen ranks candidates)
               ─► waypoints: [rise] → approach → pre_grasp → grasp → close → lift → above_tray → lower → open → retreat
                  (horizontal moves at ≥ transit_height 0.32 m; episode ends shortly into the retreat once successful)
               ─► each step: action = clip(K·(waypoint − ee_pose)), gripper cmd ─► sim.step(action)
               ─► success = target in tray, at rest, gripper open  → keep episode, else discard + log
```

| Class | Grasp template |
|---|---|
| bottle | Side grasp at 0.4 × height, approaching from −y, fingers horizontal |
| vial | Top-down grasp at centroid, yaw = current gripper yaw (round object, so no wrist rotation) |
| syringe | Top-down grasp at the object centre (on the barrel), fingers perpendicular to the barrel axis; of the two symmetric yaws the one nearer the current gripper yaw |

The sim adapter interface (each simulator implements it):

```python
class ExpertSim(Protocol):
    def reset(self, cfg: EpisodeConfig) -> dict
    def get_object_pose(self, name: str) -> np.ndarray   # [x, y, z, qw, qx, qy, qz], base frame
    def get_ee_pose(self) -> np.ndarray
    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]
    def tray_contains(self, name: str) -> bool
```

`SimBackend` (from `contracts/`) wraps the same adapter for policy evaluation. Stage flags are computed from the simulator state: reach / grasp / lift / transport / place.

## 6. Simulator tracks

| | Isaac Sim (owner: Mitesh) | robosuite (owner: Rakesh) |
|---|---|---|
| Base | Isaac Lab `ManagerBasedRLEnv` task derived from the Franka lift/stack environments | Custom `CathlabPickPlace(ManipulationEnv)` derived from robosuite `PickPlace` |
| Scene authoring | Python scene config: table/tray/objects as primitive-composite rigid bodies | MJCF objects (`CompositeBodyObject` / primitives), table arena, tray body |
| Randomisation | Isaac Lab events: reset pose from layout, `randomize_rigid_body_material` (friction), mass scale, visual colour, light attributes | Set rgba, pose, mass and friction on the MjModel at reset; light attributes |
| Control | `DifferentialIKController` relative pose + binary gripper | `OSC_POSE` delta + gripper |
| Cameras | `agentview` pose matched to robosuite's; wrist camera on `panda_hand` | robosuite `agentview` + `robot0_eye_in_hand` |
| Throughput target | ≥3 episodes/min per GPU instance (384 in ~2 h) | Parallel CPU shards, 384 in under 1 h |
| Stretch | Cosmos Transfer on ~50 episodes | — |

Camera poses and colour palettes are defined once in the variation config and applied to both simulators, to keep the visual domains aligned.

## 7. Pipeline changes per phase

| Phase | Change |
|---|---|
| 01 Ingest | Sources: `isaac_scripted`, `robosuite_scripted`, and the provided FR3 Duo data (ingest/validate only). Instruction paraphrases per object ("Hand me the syringe", "Pass the vial", ...) |
| 02 Dataset Intel | Coverage map = variation grid (object × layout × object colour × table colour × simulator); gap = empty or failing cell; per-cell expert success |
| DataEngine | Two generators sharing the expert core and the `EpisodeConfig` list. MimicGen and teleop move to the catalogue |
| 03 Train | Start from LIBERO checkpoints. SmolVLA v0/v1/v2, ACT v0/v1, frontier (π0.5 or X-VLA) on v1 |
| 04 Eval | `suite_v2` per simulator: 3 objects × 5 tiers (in-distribution / held-out layout / held-out object colour / held-out table colour / all held-out) × 4 seeds = **60 episodes**; the expert is scored too; wrong-object is a failure class |
| Loop | Failing cells → K new episodes for exactly those cells in both simulators → v2 |
| 05 Serve | Unchanged; the Isaac result is the promotion gate, robosuite is reported alongside |
| Awareness | Tool vocabulary becomes bottle / vial / syringe |

**Data versions:**
- **v0** = Isaac scripted (384).
- **v1** = Isaac + robosuite (768).
- **v2** = v1 + loop-targeted (~100).

**Benchmark matrix:** {expert, SmolVLA v0/v1/v2, ACT v0/v1, frontier v1} × {Isaac, robosuite, RoboSimStudio (stretch)} × {trained, held-out}.

**Claims to prove:**
1. v1 > v0 on held-out cells **and** a smaller Isaac↔robosuite gap.
2. v2 fixes the failing cells.
3. The expert itself reaches ≥80% (our "own expert policy").

## 8. Milestones

| H | Gate |
|---|---|
| 8 | Variation spec, `EpisodeConfig`, `ExpertSim` and the I/O contract frozen; empty scenes load in both simulators |
| 12 | Expert ≥80% success on a 20-episode sample in **both** simulators |
| 16 | 384 episodes per simulator recorded and passing 02 |
| 18 | SmolVLA + ACT v0 training |
| 20 | v1 training (SmolVLA, ACT, frontier) |
| 24 | v0/v1 evaluation matrix in both simulators |
| 28 | Loop → v2 |
| 30 | RoboSimStudio held-out evaluation / Cosmos (stretch) |
| 32 | Freeze |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Isaac Lab diff-IK and OSC behave differently for the same delta | Match gains/clipping; check by replaying one expert trajectory in both simulators and comparing end-effector paths (H10) |
| Bottle side-grasp fails with the Franka hand | Fall back to a top grasp on the cap |
| Syringe slips (thin, light) | Raise friction on the finger pads; grasp the barrel near the flange |
| LIBERO checkpoints expect different camera framing | Match the `agentview` pose to LIBERO's; fine-tuning absorbs the rest |
| Judges expect the provided FR3 Duo scene | Provided data passes 01/02; claim embodiment-agnostic contracts; FR3 Duo port as next step in the pitch |

## 10. Option catalogue changes

- ✅ Chosen: Isaac Lab + robosuite (Panda), analytic expert → GraspGen, LIBERO checkpoints.
- ○ Documented: FR3 Duo provided scene, MimicGen, teleop, Cosmos (stretch), RoboSimStudio data factory, GR00T N1.x.
