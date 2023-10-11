import torch
from dataclasses import dataclass


@dataclass
class ClothObjectSimulationData:
    root_state_w: torch.Tensor = None

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.root_state_w[:, 3:7]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.root_state_w[:, 7:10]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.root_state_w[:, 10:13]
