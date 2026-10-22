"""Object setup for the provided FR3 Duo scene: bottle, vial, syringe (Y-up stage, metres)."""
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

BOTTLE = "/World/NaturalBostonRoundBottle_A02_PR_NVD_01"
VIAL = "/World/SM_PillBottle_01v"
BOX = "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"
SYRINGE = "/World/Cathlab/Syringe"
MATERIAL = "/World/Cathlab/PhysicsMaterial"

MASS_KG = {"bottle": 0.30, "vial": 0.02, "syringe": 0.015}
FRICTION = 0.8
BARREL_RADIUS, BARREL_LENGTH = 0.0075, 0.100
PLUNGER_RADIUS, PLUNGER_LENGTH = 0.004, 0.040
FLANGE_SIZE = (0.030, 0.003, 0.010)


def _material(stage: Usd.Stage) -> UsdPhysics.MaterialAPI:
    prim = stage.DefinePrim(MATERIAL, "Material")
    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateStaticFrictionAttr(FRICTION)
    mat.CreateDynamicFrictionAttr(FRICTION)
    mat.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim: Usd.Prim, material: UsdPhysics.MaterialAPI) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(material.GetPrim()), UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _collide_convex(prim: Usd.Prim) -> None:
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Gprim):
            UsdPhysics.CollisionAPI.Apply(p)
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")


def _make_dynamic(prim: Usd.Prim, mass: float, material: UsdPhysics.MaterialAPI) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    _collide_convex(prim)
    _bind_material(prim, material)


def _syringe(stage: Usd.Stage) -> Usd.Prim:
    """Barrel + plunger + flange lying flat along +X, resting on the Y=0 plane of its own frame."""
    root = UsdGeom.Xform.Define(stage, SYRINGE)
    barrel = UsdGeom.Cylinder.Define(stage, f"{SYRINGE}/barrel")
    barrel.CreateAxisAttr("X")
    barrel.CreateRadiusAttr(BARREL_RADIUS)
    barrel.CreateHeightAttr(BARREL_LENGTH)
    barrel.AddTranslateOp().Set(Gf.Vec3d(0, BARREL_RADIUS, 0))
    plunger = UsdGeom.Cylinder.Define(stage, f"{SYRINGE}/plunger")
    plunger.CreateAxisAttr("X")
    plunger.CreateRadiusAttr(PLUNGER_RADIUS)
    plunger.CreateHeightAttr(PLUNGER_LENGTH)
    plunger.AddTranslateOp().Set(Gf.Vec3d(-(BARREL_LENGTH + PLUNGER_LENGTH) / 2, BARREL_RADIUS, 0))
    flange = UsdGeom.Cube.Define(stage, f"{SYRINGE}/flange")
    flange.CreateSizeAttr(1.0)
    flange.AddTranslateOp().Set(Gf.Vec3d(-BARREL_LENGTH / 2 + FLANGE_SIZE[1] / 2, BARREL_RADIUS, 0))
    flange.AddScaleOp().Set(Gf.Vec3f(FLANGE_SIZE[1], FLANGE_SIZE[2], FLANGE_SIZE[0]))
    return root.GetPrim()


def _park_box(stage: Usd.Stage) -> None:
    box = stage.GetPrimAtPath(BOX)
    if not box:
        return
    UsdPhysics.RigidBodyAPI.Apply(box).CreateRigidBodyEnabledAttr(False)
    for p in Usd.PrimRange(box):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    UsdGeom.Imageable(box).MakeInvisible()


def setup_objects(stage: Usd.Stage) -> dict[str, str]:
    """Prepare bottle, vial and syringe as stable dynamic bodies; park the provided box."""
    for path in (BOTTLE, VIAL):
        if not stage.GetPrimAtPath(path):
            raise ValueError(f"missing scene prim: {path}")
    UsdGeom.Xform.Define(stage, "/World/Cathlab")
    material = _material(stage)
    _make_dynamic(stage.GetPrimAtPath(BOTTLE), MASS_KG["bottle"], material)
    _make_dynamic(stage.GetPrimAtPath(VIAL), MASS_KG["vial"], material)
    _make_dynamic(_syringe(stage), MASS_KG["syringe"], material)
    _park_box(stage)
    return {"bottle": BOTTLE, "vial": VIAL, "syringe": SYRINGE}
