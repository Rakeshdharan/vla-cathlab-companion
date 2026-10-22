import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from sim.isaac import objects  # noqa: E402


def make_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    for path in (objects.BOTTLE, objects.VIAL, objects.BOX):
        cube = UsdGeom.Cube.Define(stage, path)
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return stage


@pytest.fixture
def stage():
    s = make_stage()
    objects.setup_objects(s)
    return s


def test_returns_prim_paths_that_exist(stage):
    paths = objects.setup_objects(make_stage())
    assert set(paths) == {"bottle", "vial", "syringe"}
    assert paths["syringe"] == "/World/Cathlab/Syringe"


@pytest.mark.parametrize("name,path", [("bottle", objects.BOTTLE), ("vial", objects.VIAL), ("syringe", objects.SYRINGE)])
def test_explicit_mass_on_every_object(stage, name, path):
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get()
    assert mass == pytest.approx(objects.MASS_KG[name])


@pytest.mark.parametrize("path", [objects.BOTTLE, objects.VIAL, objects.SYRINGE])
def test_objects_are_enabled_rigid_bodies(stage, path):
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetRigidBodyEnabledAttr().Get() is True


def test_syringe_parts_are_colliders_with_the_shared_material(stage):
    from pxr import UsdShade

    for part in ("barrel", "plunger", "flange"):
        prim = stage.GetPrimAtPath(f"{objects.SYRINGE}/{part}")
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
    material = stage.GetPrimAtPath(objects.MATERIAL)
    assert UsdPhysics.MaterialAPI(material).GetStaticFrictionAttr().Get() == pytest.approx(objects.FRICTION)
    bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(objects.SYRINGE)).ComputeBoundMaterial("physics")[0]
    assert bound.GetPath() == material.GetPath()


def test_syringe_dimensions_and_resting_height(stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(objects.SYRINGE)).ComputeAlignedRange()
    size = box.GetSize()
    assert size[0] == pytest.approx(0.140, abs=1e-6)  # plunger + barrel along X
    assert size[1] == pytest.approx(0.015, abs=1e-6)  # barrel diameter
    assert box.GetMin()[1] == pytest.approx(0.0, abs=1e-6)  # lies on its own Y=0 plane


def test_box_is_parked(stage):
    box = stage.GetPrimAtPath(objects.BOX)
    assert UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Get() is False
    assert UsdPhysics.CollisionAPI(box).GetCollisionEnabledAttr().Get() is False
    assert UsdGeom.Imageable(box).ComputeVisibility() == UsdGeom.Tokens.invisible


def test_missing_scene_prim_is_reported():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World")
    with pytest.raises(ValueError, match="NaturalBostonRoundBottle"):
        objects.setup_objects(s)
