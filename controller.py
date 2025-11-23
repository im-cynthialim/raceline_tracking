import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

min_index = 0
integral = 0
prev_heading_error = 0

def find_lookahead_point(centerline, start_idx, current_pos, lookahead_distance):
    
    total_dist = 0.0 
    num_points = len(centerline)
    
    # for each point on racetrack, distance between current index and previous index
    for i in range(1, num_points):
        idx = (start_idx + i) % num_points         
        prev_idx = (start_idx + i - 1) % num_points
        
        # get distance between points
        segment_dist = np.linalg.norm(
            centerline[idx] - centerline[prev_idx]
        )

        # add to total
        total_dist += segment_dist
        
        # once total is greater than desired distance, return index
        if total_dist >= lookahead_distance:
            return idx 
    
    # if loop done without finding point, return last point
    return (start_idx + num_points - 1) % num_points

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # adjusts current state to meet desired values, returns steering rate and acceleration (simulator integrates this to update steer angle and velocity)

    sx, sy, delta, v, phi = state
    delta_r, vr = desired

    min_steering_velocity = parameters[7]
    max_steering_velocity = parameters[9]
    
    # find difference between desired and current
    delta_error = delta_r - delta
    Kp_delta = 8.0
    v_delta = Kp_delta * delta_error
    v_delta = np.clip(v_delta, min_steering_velocity, max_steering_velocity)
    print(f"Min steering vel: {min_steering_velocity}, Max steering vel: {max_steering_velocity}")
    
    # calculate acceleration
    min_accel = parameters[8]
    max_accel = parameters[10]
    v_error = vr - v
    Kp_v = 3.0
    a = Kp_v * v_error
    a = np.clip(a, min_accel, max_accel)

    print(f"Steering rate: {v_delta:.2f}, Acceleration command: {a:.2f}")

    return np.array([v_delta, a]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    global integral, prev_heading_error, min_index
    
    # responsible for computing desired values (steering angle, velocity) from current state
    dt = 1e-1 # this is the sampling period
    
    # current state
    sx, sy, delta, v, phi = state

    current_pos = np.array([sx, sy])
    distances = np.linalg.norm(racetrack.centerline - current_pos, axis=1)
    closest_idx = max(min_index, np.argmin(distances))
    
    min_index = closest_idx

    print(f"State: sx={sx:.2f}, sy={sy:.2f}, v={v:.2f}, delta={delta:.2f}, phi={phi:.2f}")
    
    # lookahead by 5 points
    # lookahead_idx = (closest_idx + 5) % len(racetrack.centerline)
    # target_point = racetrack.centerline[lookahead_idx]

    # desired velocity (based on delta_r)
    # min_steering_angle = parameters[1]
    # max_steering_angle = parameters[4]
    min_velocity = parameters[2]
    max_velocity = parameters[5]

    # vr = -((max_velocity - min_velocity) / max_steering_angle) * abs(delta) + max_velocity # linear interpolation
    # vr = 25.0
    # vr.clip(min_velocity, max_velocity)

    Ld_min = 1.5  # min lookahead
    k = 0.3 # scaling factor
    Ld = Ld_min + k * abs(v)  # dynamic lookahead
    
    print(f"Velocity: {v:.2f}, Lookahead distance: {Ld:.2f}m")
    
    # Find closest point on raceline
    current_pos = np.array([sx, sy])
    distances = np.linalg.norm(racetrack.centerline - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # Find target point at lookahead distance Ld
    target_idx = find_lookahead_point(
        racetrack.centerline, 
        closest_idx, 
        current_pos, 
        Ld
    )
    target_point = racetrack.centerline[target_idx]

    # vr = 50
    # vr = np.clip(vr, min_velocity, max_velocity)   

    # calculate desired heading to reach target point
    dx = target_point[0] - sx
    dy = target_point[1] - sy
    desired_heading = np.arctan2(dy, dx)

    # heading difference
    heading_error = desired_heading - phi
    
    # normalize (wrap between [-pi, pi])
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))


    # varying velocity based on curvature
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # dx_car = cos_phi * dx + sin_phi * dy
    dy_car = -sin_phi * dx + cos_phi * dy
    
    x = dy_car  # Cross-track error
    curvature = (2 * x) / (Ld ** 2)
    

    max_velocity = parameters[5]
    min_velocity = parameters[2]
    k_curvature = 100
    
    vr = max_velocity / (1 + k_curvature * abs(curvature))
    # vr = 50
    vr = np.clip(vr, min_velocity, max_velocity)

    lwb = parameters[0] # wheelbase

    Kp = 0.25
    # Kp = 0.5 # kp
    # Kd = 0.35 
    Kd = 0.2
    # Kd = 0
    # Kp = 2.0
    # Kd = 0.5

    global prev_heading_error

    controller_transfer_fun = Kp * heading_error + (Kd * (heading_error - prev_heading_error) / dt)
    delta_r = (controller_transfer_fun * lwb) / (max(vr * dt, 0.1)) 
    
    prev_heading_error = heading_error

    # clip steering angle rate
    delta_r = np.clip(delta_r, parameters[1], parameters[4])

    print(f"Current heading: {phi:.2f}, Desired: {desired_heading:.2f}, Error: {heading_error:.2f}")
    print(f"Commanding: delta_r={delta_r:.3f}, vr={vr:.1f}")
    
    print("-" * 50)
    
    return np.array([delta_r, vr]).T