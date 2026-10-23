---
applyTo: "assets/**,configs/variation/**,dataengine/variation.py"
---
# Objects and variation specs

- Never edit `configs/variation/cathlab_v1.yaml`; add `cathlab_v<N>.yaml` for new objects or axes.
- Every object needs a grasp the analytic expert understands; the expert must reach >= 80 % on it
  (`python -m engines.robosuite.cli check-expert --objects <name>`).
- Object packages live in `assets/objects/<name>/` (object.yaml, visual.obj, collision/*.obj, object.xml); importers never
  overwrite existing packages. Follow `harness/gates/objects.yaml`.
