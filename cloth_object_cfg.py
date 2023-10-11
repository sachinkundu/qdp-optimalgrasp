from dataclasses import MISSING
from omni.isaac.orbit.utils import configclass


@configclass
class MetaInfoCfg:
    usd_path = MISSING
    scale = (1.0, 1.0, 1.0)


@configclass
class ClothBodyPropertiesCfg:
    solver_positon_iteration_count = None
    solver_velocity_iteration_count = None
    max_linear_velocity = 1000.0
    max_angular_velocity = 1000.0
    max_depenetration_velocity = 10.0
    disable_gravity = False


@configclass
class CollisionsPropertiesCfg:
    collision_enabled = None
    contact_offset = None
    rest_offset = None
    torsional_patch_radius = None
    min_torsional_patch_radius = None


@configclass
class ClothMaterialCfg:
    particle_system_path = "/World/Materials/particleSystem"
    particle_material_path = "/World/Materials/particleMaterial"
    material_path = "/Looks/OmniPBR"
    drag = 0.1
    lift = 0.1
    friction = 0.6
    radius = 0.005
    restOffset = radius
    contactOffset = restOffset * 1.5
    stretch_stiffness = 200
    bend_stiffness = 20
    shear_stiffness = 20
    spring_damping = 0.2
    particle_mass = 0.00005
    visible = True


@configclass
class InitialStateCfg:
    cloth_usd_path: str
    cloth_usd_prim_path = "/World/envs/env_0/Cloth"
    cloth_mesh_path: str
    pos = (0.3, 0.3, 0.3)
    rot = (1.0, 0.0, 0.0, 0.0)
    scale = (1.0, 1.0, 1.0)
    lin_vel = (0.0, 0.0, 0.0)
    ang_vel = (0.0, 0.0, 0.0)


class ClothObjectCfg:
    def __init__(self, cloth_usd_path) -> None:
        self.cloth_usd_path = cloth_usd_path
        self.meta_info = MetaInfoCfg()
        self.init_state = InitialStateCfg(cloth_usd_path=self.cloth_usd_path)
        self.cloth_props = ClothBodyPropertiesCfg()
        self.collision_props = CollisionsPropertiesCfg()
        self.cloth_material = ClothMaterialCfg()
