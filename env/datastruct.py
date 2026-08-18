import numpy as np
from typing import List


class Function(object):

    def __init__(
            self,
            Function_task_datasize: float,
            Function_task_computing_resource: float,
            Function_task_delay: int

    ) -> None:
        self._Function_task_datasize = Function_task_datasize
        self._Function_task_computing_resource = Function_task_computing_resource
        self._Function_task_delay = Function_task_delay

    def get_task_datasize(self) -> float:
        return float(self._Function_task_datasize)

    def get_task_computing_resource(self) -> float:
        return float(self._Function_task_computing_resource)

    def get_task_delay(self) -> float:
        return float(self._Function_task_delay)


class TaskList(object):

    def __init__(
            self,
            task_number: int,
            minimum_datasize: float,
            maximum_datasize: float
    ) -> None:
        self._task_number = task_number
        self._minimum_datasize = minimum_datasize
        self._maximum_datasize = maximum_datasize

        self._datasizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, self._task_number)
        self._task_list = [_ for _ in self._datasizes]

    def get_task_list(self) -> List[float]:
        return self._task_list

    def sum_datasize(self) -> float:
        return sum(self._task_list)

    def add_task_list(self, new_data_size) -> None:
        self._task_list.append(new_data_size)

    def add_by_slot(self, task_number) -> None:
        data_sizes = np.random.uniform(self._minimum_datasize, self._maximum_datasize, task_number)
        for datasize in data_sizes:
            self._task_list.append(datasize)
            self._task_number += 1

    def delete_data_list(self, process_ability) -> None:
        while True:
            if len(self._task_list) == 0:
                break
            elif process_ability >= self._task_list[0]:
                process_ability -= self._task_list[0]
                del self._task_list[0]
            else:
                self._task_list[0] -= process_ability
                break


class Vehicle(object):

    def __init__(
            self,
            road_range: int,
            min_vehicle_speed: float,
            max_vehicle_speed: float,
            min_task_number: float,
            max_task_number: float,
            min_task_datasize: float,
            max_task_datasize: float,
            min_vehicle_compute_ability: float,
            max_vehicle_compute_ability: float,
            vehicle_x_initial_location: list,
            min_vehicle_y_initial_location: float,
            max_vehicle_y_initial_location: float,
            seed: int
    ) -> None:
        self._road_range = road_range
        self._seed = seed
        self._min_vehicle_y_initial_location = min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location = max_vehicle_y_initial_location
        np.random.seed(self._seed)
        self._vehicle_y_initial_location = \
            np.random.randint(self._min_vehicle_y_initial_location, self._max_vehicle_y_initial_location, 1)[0]
        self._vehicle_y_location = self._vehicle_y_initial_location
        np.random.seed(self._seed)
        self._vehicle_x_initial_location = np.random.choice(vehicle_x_initial_location)
        self._vehicle_x_location = self._vehicle_x_initial_location
        np.random.seed(self._seed)
        self._vehicle_speed = np.random.randint(min_vehicle_speed, max_vehicle_speed)
        if self._vehicle_x_initial_location == 0:
            self._vehicle_speed = self._vehicle_speed
        else:
            self._vehicle_speed = -self._vehicle_speed
        self._stay_time = int(self._road_range / self._vehicle_speed)

        self._max_compute_ability = max_vehicle_compute_ability
        self._min_compute_ability = min_vehicle_compute_ability

        self._min_compute_ability = np.random.uniform(self._min_compute_ability, self._max_compute_ability, 1)

        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_datasize = max_task_datasize
        self._min_datasize = min_task_datasize
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._vehicle_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize)

    def get_initial_data(self) -> list:

        data = [self._vehicle_x_initial_location, self._vehicle_y_initial_location, self._vehicle_speed]
        return data

    def get_stay_time(self) -> int:
        return self._stay_time

    def get_location(self) -> list:
        location = [self._vehicle_x_location, self._vehicle_y_location]
        return location

    def change_location(self) -> list:
        self._vehicle_x_location = self._vehicle_x_location + self._vehicle_speed * 1
        self._vehicle_y_location = self._vehicle_y_initial_location
        location = [self._vehicle_x_location, self._vehicle_y_location]
        return location

    def decrease_stay_time(self) -> int:
        self._stay_time -= 1
        return self._stay_time

    def is_out(self) -> bool:
        if self._stay_time <= 5:
            return True
        else:
            return False

    def get_vehicle_speed(self) -> float:
        return self._vehicle_speed

    def get_vehicle_compute_ability(self) -> float:
        return self._max_compute_ability

    def get_task_list(self) -> TaskList:
        return self._vehicle_task_list

    def get_sum_tasks(self) -> float:
        if len(self._vehicle_task_list.get_task_list()) == 0:
            return 0
        else:
            return self._vehicle_task_list.sum_datasize()


class VehicleList(object):

    def __init__(
            self,
            vehicle_number: int,
            road_range: int,
            min_vehicle_speed: float,
            max_vehicle_speed: float,
            min_task_number: float,
            max_task_number: float,
            min_task_datasize: float,
            max_task_datasize: float,
            min_vehicle_compute_ability: float,
            max_vehicle_compute_ability: float,
            vehicle_x_initial_location: list,
            min_vehicle_y_initial_location: float,
            max_vehicle_y_initial_location: float,
            seed: int

    ) -> None:
        self._seed = seed
        self._vehicle_number = vehicle_number
        self._road_range = road_range
        self._min_vehicle_speed = min_vehicle_speed
        self._max_vehicle_speed = max_vehicle_speed
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_datasize = min_task_datasize
        self._max_datasize = max_task_datasize
        self._min_compute_ability = min_vehicle_compute_ability
        self._max_compute_ability = max_vehicle_compute_ability
        self._vehicle_x_initial_location = vehicle_x_initial_location
        self._min_vehicle_y_initial_location = min_vehicle_y_initial_location
        self._max_vehicle_y_initial_location = max_vehicle_y_initial_location

        self.vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                min_vehicle_speed=self._min_vehicle_speed,
                max_vehicle_speed=self._max_vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                vehicle_x_initial_location=self._vehicle_x_initial_location,
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location,
                seed=self._seed + _
            )
            for _ in range(self._vehicle_number)]

    def get_vehicle_number(self) -> int:
        return self._vehicle_number

    def get_vehicle_list(self) -> List[Vehicle]:
        return self.vehicle_list

    def add_stay_vehicle(self, new_vehicle_number, time_now) -> None:
        new_vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                min_vehicle_speed=self._min_vehicle_speed,
                max_vehicle_speed=self._max_vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_vehicle_compute_ability=self._min_compute_ability,
                max_vehicle_compute_ability=self._max_compute_ability,
                vehicle_x_initial_location=self._vehicle_x_initial_location,
                min_vehicle_y_initial_location=self._min_vehicle_y_initial_location,
                max_vehicle_y_initial_location=self._max_vehicle_y_initial_location,
                seed=time_now + _
            )
            for _ in range(new_vehicle_number)]

        self.vehicle_list = self.vehicle_list + new_vehicle_list
        self._vehicle_number += new_vehicle_number

    def delete_out_vehicle(self) -> None:
        i = 0
        while i < len(self.vehicle_list):
            if len(self.vehicle_list) == 0:
                pass
            elif self.vehicle_list[i].is_out():
                del self.vehicle_list[i]
                self._vehicle_number -= 1
            else:
                i += 1


class RSU(object):

    def __init__(
            self,
            min_task_number: float,
            max_task_number: float,
            min_task_datasize: float,
            max_task_datasize: float,
            min_rsu_compute_ability: float,
            max_rsu_compute_ability: float
    ) -> None:
        self._max_compute_ability = max_rsu_compute_ability
        self._min_compute_ability = min_rsu_compute_ability
        self._compute_ability = np.random.uniform(self._min_compute_ability, self._max_compute_ability, 1)

        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_datasize = max_task_datasize
        self._min_datasize = min_task_datasize
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._rsu_task_list = TaskList(self._task_number, self._min_datasize, self._max_datasize)

    def get_rsu_compute_ability(self) -> float:
        return self._compute_ability

    def get_task_list(self) -> TaskList:
        return self._rsu_task_listF

    def get_sum_tasks(self) -> float:
        if len(self._rsu_task_list.get_task_list()) == 0:
            return 0
        else:
            return self._rsu_task_list.sum_datasize()


class RSUList(object):

    def __init__(
            self,
            rsu_number,
            min_task_number: float,
            max_task_number: float,
            min_task_datasize: float,
            max_task_datasize: float,
            min_rsu_compute_ability: float,
            max_rsu_compute_ability: float
    ) -> None:
        self._rsu_number = rsu_number

        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_datasize = min_task_datasize
        self._max_datasize = max_task_datasize
        self._min_compute_ability = min_rsu_compute_ability
        self._max_compute_ability = max_rsu_compute_ability

        self.rsu_list = [
            RSU(
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                min_task_datasize=self._min_datasize,
                max_task_datasize=self._max_datasize,
                min_rsu_compute_ability=self._min_compute_ability,
                max_rsu_compute_ability=self._max_compute_ability
            )
            for _ in range(rsu_number)]

    def get_rsu_number(self):
        return self._rsu_number

    def get_rsu_list(self):
        return self.rsu_list


class TimeSlot(object):

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.slot_length = self.end - self.start

        self.now = start
        self.reset()

    def __str__(self):
        return f"now time: {self.now}, [{self.start} , {self.end}] with {self.slot_length} slots"

    def add_time(self) -> None:
        self.now += 1

    def is_end(self) -> bool:
        return self.now >= self.end

    def get_slot_length(self) -> int:
        return self.slot_length

    def get_now(self) -> int:
        return self.now

    def reset(self) -> None:
        self.now = self.start
