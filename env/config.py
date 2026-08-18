import dataclasses
import numpy as np


@dataclasses.dataclass
class VehicularEnvConfig:
    def __init__(self):
        self.road_range: int = 1200
        self.road_width: int = 50

        self.time_slot_start: int = 0
        self.time_slot_end: int = 99

        self.Function_min_task_datasize = 2
        self.Function_max_task_datasize = 5
        self.Function_task_computing_resource: float = 300
        self.Function_min_task_delay: int = 20
        self.Function_max_task_delay: int = 25

        self.min_rsu_task_number: int = 2
        self.max_rsu_task_number: int = 3
        self.min_vehicle_task_number: int = 4
        self.max_vehicle_task_number: int = 5
        self.min_task_datasize: float = 2
        self.max_task_datasize: float = 4

        self.min_vehicle_speed: int = 30
        self.max_vehicle_speed: int = 40
        self.min_vehicle_compute_ability: float = 20000
        self.max_vehicle_compute_ability: float = 25000
        self.vehicle_number = 10
        self.seed = 1
        self.min_vehicle_y_initial_location: float = 0
        self.max_vehicle_y_initial_location: float = 50
        self.vehicle_x_initial_location: list = [0, self.road_range]

        self.rsu_number = 3
        self.min_rsu_compute_ability: float = 25000
        self.max_rsu_compute_ability: float = 30000

        self.rsu_range: int = 400
        self.vehicle_range: int = 200
        self.r2v_B: float = 20
        self.v2v_B: float = 40
        self.rsu_p: float = 50
        self.vehicle_p: float = 10
        self.w: float = 0.001
        self.k: float = 30
        self.theta: int = 2
        self.r2r_onehop_time: float = 8
        self.c2r_rate: float = 0.2
        self.min_transfer_rate: float = 0.01
        self.rsu_connect_time: float = 10000
        self.cloud_connect_time: float = 10000

        self.punishment = -200

        self.action_size = (self.rsu_number + self.vehicle_number + 1) ** 3
        self.high = np.array([np.finfo(np.float32).max for _ in range(self.rsu_number + self.vehicle_number)])
        self.low = np.array([0 for _ in range(self.rsu_number + self.vehicle_number)])
