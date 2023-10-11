import torch
from typing import Optional, Sequence

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.materials import ParticleMaterial
from omni.isaac.core.prims import ParticleSystem, ClothPrim, ClothPrimView
from cloth_object_cfg import ClothObjectCfg
from cloth_object_simulation_data import ClothObjectSimulationData


class ClothObject:
    def __init__(self, cfg: ClothObjectCfg = None) -> None:
        self.cfg: ClothObjectCfg = cfg if cfg else ClothObjectCfg()
        self.prim_path_at_init = None
        self._prim_paths_expr = None
        self.objects: ClothPrimView = None
        self._data = ClothObjectSimulationData()

    def spawn(self, prim_path):
        self.prim_path_at_init = prim_path
        if not prim_utils.is_prim_path_valid(prim_path):
            prim_utils.create_prim(
                self.cfg.init_state.cloth_usd_prim_path,
                usd_path=self.cfg.init_state.cloth_usd_path,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
                scale=self.cfg.init_state.scale
            )

        self._particle_material = ParticleMaterial(
            prim_path=self.cfg.cloth_material.particle_material_path,
            drag=self.cfg.cloth_material.drag,
            lift=self.cfg.cloth_material.lift,
            friction=self.cfg.cloth_material.friction
        )

        self._particle_system = ParticleSystem(
            prim_path=self.cfg.cloth_material.particle_system_path,
            simulation_owner="physicsScene",
            rest_offset=self.cfg.cloth_material.restOffset,
            contact_offset=self.cfg.cloth_material.contactOffset,
            solid_rest_offset=self.cfg.cloth_material.restOffset,
            fluid_rest_offset=self.cfg.cloth_material.restOffset,
            particle_contact_offset=self.cfg.cloth_material.contactOffset
        )

        ClothPrim(prim_path, self.particle_system, self._particle_material,
                  stretch_stiffness=self.cfg.cloth_material.stretch_stiffness,
                  bend_stiffness=self.cfg.cloth_material.bend_stiffness,
                  shear_stiffness=self.cfg.cloth_material.shear_stiffness,
                  spring_damping=self.cfg.cloth_material.spring_damping,
                  particle_mass=self.cfg.cloth_material.particle_mass,
                  visible=True)

    def initialize(self, prim_paths_expr=None):
        self._prim_paths_expr = prim_paths_expr if prim_paths_expr is not None else self.prim_path_at_init
        self.objects = ClothPrimView(self._prim_paths_expr, reset_xform_properties=False)
        self.objects.initialize()
        self.objects.post_reset()
        self._process_info_cfg()
        self._create_buffers()

# Boilerplate code needed for simulation

    def update_buffers(self, dt: float = None):
        position_w, quat_w = self.objects.get_world_poses(indices=self._ALL_INDICES, clone=False)
        self._data.root_state_w[:, 0:3] = position_w
        self._data.root_state_w[:, 3:7] = quat_w
        self._data.root_state_w[:, 7:] = self.objects.get_velocities(indices=self._ALL_INDICES, clone=False)

    def set_root_state(self, root_states: torch.Tensor, env_ids: Optional[Sequence[int]] = None):
        if env_ids is None:
            env_ids = self._ALL_INDICES
        self.objects.set_world_poses(root_states[:, 0:3], root_states[:, 3:7], indices=env_ids)
        self.objects.set_velocities(root_states[:, 7:], indices=env_ids)
        self._data.root_state_w[env_ids] = root_states.clone()

    def get_default_root_state(self, env_ids: Optional[Sequence[int]] = None, clone=True) -> torch.Tensor:
        if env_ids is None:
            env_ids = ...
        if clone:
            return torch.clone(self._default_root_states[env_ids])
        else:
            return self._default_root_states[env_ids]

    def _process_info_cfg(self) -> None:
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        self._default_root_states = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._default_root_states = self._default_root_states.repeat(self.count, 1)

    def _create_buffers(self):
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)
        self._data.root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
