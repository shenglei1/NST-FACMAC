from toolz import curry
from numpy import append

from episode_buffer import Trajectory


@curry
def last_observation_bc(traj: Trajectory, add_timestep=False):
    last_observation = traj.observations[-1]

    if add_timestep:
        last_observation = append(last_observation, len(traj.observations))

    return last_observation

@curry
def last_position_bc(traj: Trajectory, add_timestep=False):
    last_position = traj.positions[-1]

    if add_timestep:
        last_position = append(last_position, len(traj.positions))

    return last_position

@curry
def last_observation_and_position_bc(traj: Trajectory, add_timestep=False):
    bc = append(
        last_observation_bc(traj),
        last_position_bc(traj)
    )

    if add_timestep:
        bc = append(bc, len(traj.positions))

    return bc