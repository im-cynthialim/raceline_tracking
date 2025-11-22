import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # adjusts current state to meet desired values, returns steering rate and acceleration (simulator integrates this to update steer angle and velocity)

    sx, sy, delta, v, phi = state
    delta_r, vr = desired
    
    # find difference between desired and current
    delta_error = delta_r - delta
    Kp_delta = 4.0
    v_delta = Kp_delta * delta_error
    v_delta = np.clip(v_delta, -20.0, 20.0)
    
    # calculate acceleration
    v_error = vr - v
    Kp_v = 2.0
    a = Kp_v * v_error
    a = np.clip(a, -10, 10)

    return np.array([v_delta, a]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # responsible for computing desired values (steering angle, velocity) from current state
    
    # current state
    sx, sy, delta, v, phi = state
    
    # desired velocity 
    vr = 25.0  # constant for now

    # desired steering angle
    current_pos = np.array([sx, sy])
    distances = np.linalg.norm(racetrack.centerline - current_pos, axis=1)
    closest_idx = np.argmin(distances)

    print(f"State: sx={sx:.2f}, sy={sy:.2f}, v={v:.2f}, delta={delta:.2f}, phi={phi:.2f}")
    
    # lookahead by 5 points
    lookahead_idx = (closest_idx + 5) % len(racetrack.centerline)
    target_point = racetrack.centerline[lookahead_idx]
    
    # calculate desired heading to reach target point
    dx = target_point[0] - sx
    dy = target_point[1] - sy
    desired_heading = np.arctan2(dy, dx)

    # heading difference
    heading_error = desired_heading - phi
    
    # normalize
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    lwb = parameters[0] # wheelbase

    Kp = 0.5  # kp
    delta_r = (Kp * heading_error * lwb) / max(vr, 0.1)

    # clip steering angle rate
    delta_r = np.clip(delta_r, -0.9, 0.9)
    # print(f"Parameters: {parameters[1]}, {parameters[4]}")

    print(f"Closest: {closest_idx}, Target: {lookahead_idx}")
    print(f"Current heading: {phi:.2f}, Desired: {desired_heading:.2f}, Error: {heading_error:.2f}")
    print(f"Commanding: delta_r={delta_r:.3f}, vr={vr:.1f}")
    
    print("-" * 50)
    
    return np.array([delta_r, vr]).T