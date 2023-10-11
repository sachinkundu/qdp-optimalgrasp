from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.kit import SimulationApp
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.carb import set_carb_setting
import omni.isaac.orbit.utils.kit as kit_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.orbit.sensors.camera.camera_cfg import PinholeCameraCfg
from omni.isaac.orbit.sensors.camera.camera import Camera

from cloth_object_cfg import ClothObjectCfg
from cloth_object import ClothObject

config = {"headless": False}
simulation_app = SimulationApp(config)


def main():
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda:0")
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=0.0)
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )

    cam_cfg = PinholeCameraCfg(width=1920, height=1080, data_types=["rgb", "distance_to_image_plane"])
    camera = Camera(cfg=cam_cfg, device="cuda")
    camera.spawn("/World", translation=(1.5, 0.0, 4.0), orientation=(0.70711, 0.0, 0.0, 0.70711))
    camera.initialize()

    cloth_mesh_path = "/World/envs/env_0/Cloth/cube_cloth/cube_cloth"

    # rim_mesh_path = "/World/envs/env_0/Cloth/base_rim_sep/Cube_0_1_021_Cube_002"

    cloth_cfg = ClothObjectCfg()
    cloth = ClothObject(cfg=cloth_cfg)
    cloth.spawn(cloth_mesh_path)
    sim.reset()
    cloth.initialize()

    print("Setup Complete")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=True)
            continue

        camera.buffer()

        sim.step()
        sim_time += sim_dt
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
