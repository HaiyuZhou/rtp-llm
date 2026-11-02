"""Gang topology derivation for multi-machine AOne execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rtp_llm.test.perf_test.errors import PlanningError


@dataclass(frozen=True)
class GangTopology:
    """The machine layout of one multi-machine gang."""

    world_size: int
    gpus_per_machine: int
    machine_count: int
    local_world_size: int


def derive_gang_topology(world_size: int, gpus_per_machine: int) -> GangTopology:
    """Derive the machine layout of a gang from the model world size.

    ``machine_count`` is always derived as ``ceil(world_size /
    gpus_per_machine)``. A ``gpus_per_machine`` at or above ``world_size``
    still derives a single machine and is normalized to ``world_size``, so an
    oversized request is a no-op that selects the default single-machine
    topology. A gang (``machine_count > 1``) must divide the world size
    evenly, and every machine then hosts ``local_world_size`` ranks.
    """
    if type(world_size) is not int or world_size <= 0:
        raise PlanningError("gang world_size must be a positive integer")
    if type(gpus_per_machine) is not int or gpus_per_machine <= 0:
        raise PlanningError("gpus_per_machine must be a positive integer")
    machine_count = math.ceil(world_size / gpus_per_machine)
    if machine_count > 1 and world_size % gpus_per_machine != 0:
        raise PlanningError(
            f"execution.gpus_per_machine {gpus_per_machine} does not divide "
            f"world_size {world_size} evenly; the derived machine_count "
            f"{machine_count} cannot host a uniform local world size"
        )
    return GangTopology(
        world_size=world_size,
        gpus_per_machine=min(gpus_per_machine, world_size),
        machine_count=machine_count,
        local_world_size=world_size // machine_count,
    )
