import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

prev_velocity_error = 0  # ADD THIS
prev_steering_error = 0  # ADD THIS
k_max_current_turn = 0
prev_k_max = []

def calculate_cross_track_error(current_pos, target_line, closest_idx):
    """
    Calculates the signed Cross-Track Error (CTE).
    Positive CTE typically means the car is to the left of the path; negative to the right.
    """
    n_points = len(target_line)
    
    # P_a: Start point of the current path segment (closest point)
    P_a = target_line[closest_idx]
    
    # P_b: End point of the current path segment (the next point, with wrap-around)
    P_b = target_line[(closest_idx + 1) % n_points]
    
    # Vector v: The path segment vector (P_b - P_a)
    v = P_b - P_a
    
    # Vector w: Vector from the segment start to the car (P_c - P_a)
    w = current_pos - P_a
    
    # Calculate the 2D "cross product" (v_x * w_y - v_y * w_x)
    # This gives the signed magnitude of the parallelogram's area.
    cross_product = (v[0] * w[1]) - (v[1] * w[0])
    
    # Calculate the length of the path segment
    segment_length = np.linalg.norm(v)
    
    # If the segment is too short, return distance to the point
    if segment_length < 1e-6:
        return np.linalg.norm(w)
    
    # CTE = Area / Base (signed projection formula)
    cte = cross_product / segment_length
    
    return cte

def find_lookahead_point(centerline, start_idx, lookahead_distance):
    """
    centerline: Nx2 array of *uniformly-spaced* raceline points (after spline resampling)
    start_idx: closest index to the car
    lookahead_distance: desired lookahead distance (meters)
    """
    num_points = len(centerline)
    total_dist = 0.0

    i = start_idx

    while total_dist < lookahead_distance:
        next_i = (i + 1) % num_points

        segment_dist = np.linalg.norm(centerline[next_i] - centerline[i])
        total_dist += segment_dist
        i = next_i

        # safety: avoid infinite loops
        if i == start_idx:
            break

    return i

def max_curvature(targetline, cur_index, lookahead_index):
    """
    Calculates the maximum curvature found on the targetline between 
    cur_index and lookahead_index using Menger Curvature (1/R).
    ACCOUNTS FOR S-CURVES BY CALCULTING SIGNED MENGER CURVATURE
    
    Args:
        targetline (list or np.array): Array of [x, y] points.
        cur_index (int): Start index.
        lookahead_index (int): End index.
        
    Returns:
        float: The maximum curvature value (1/radius) found in the segment.
               Returns 0.0 if segment is too short or straight.
    """
    points = np.array(targetline)
    n_points = len(points)
    max_k_pos_abs = 0.0
    max_k_neg_abs = 0.0

    # We need at least 3 points to calculate curvature
    if n_points < 3:
        return 0.0                                                    # MIGHT HAVE TO TUNE THIS RETURN

    # Iterate through the requested range
    # We iterate up to lookahead_index. 
    # Logic: To calculate curvature at i, we need i-1, i, and i+1.
    for i in range(cur_index, lookahead_index):
        # Handle wrapping for closed tracks (circular buffers)
        # or boundary checks for open tracks.
        idx_prev = (i - 1) % n_points
        idx_curr = i % n_points
        idx_next = (i + 1) % n_points

        p1 = points[idx_prev]
        p2 = points[idx_curr]
        p3 = points[idx_next]

        # Calculate the curvature at p2 using p1 and p3
        k = calculate_menger_curvature(p1, p2, p3)
        
        if k < 0:
            max_k_neg_abs = max(max_k_neg_abs, abs(k))
        else:
            max_k_pos_abs = max(max_k_pos_abs, abs(k))

    return [max_k_neg_abs, max_k_pos_abs]

def calculate_menger_curvature(p1, p2, p3):
    """
    Calculates the SIGNED Menger curvature (1/R) given three 2D points.
    
    The sign is determined by the sign of the cross-product/signed area:
    - Positive result typically indicates a turn to the LEFT (counter-clockwise).
    - Negative result typically indicates a turn to the RIGHT (clockwise).
    """
    
    # Create vectors for the sides of the triangle (not strictly needed for the calculation, 
    # but kept for clarity regarding original side lengths)
    v1 = p1 - p2
    v2 = p2 - p3
    v3 = p3 - p1

    # Calculate side lengths (Euclidean distance)
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)

    # Avoid division by zero if points are identical or collinear
    if a == 0 or b == 0 or c == 0:
        return 0.0

    # 1. Calculate the SIGNED Area (using the Shoelace formula / 2D Cross Product)
    # Area = 0.5 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    # Note: We REMOVE the np.abs() call from the original function.
    signed_area_2x = (p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    signed_area = 0.5 * signed_area_2x

    # If signed_area is effectively zero, the points are collinear -> curvature is 0
    if np.abs(signed_area) < 1e-9:
        return 0.0

    # 2. Menger curvature denominator: (a * b * c)
    product_of_sides = a * b * c
    
    # 3. Signed Menger curvature formula: 1/R = (4 * Signed Area) / (a * b * c)
    # Since area = (a*b*c) / (4*R), this combines the sign of the area with the magnitude.
    signed_curvature = (4.0 * signed_area) / product_of_sides
    
    return signed_curvature

def calculate_reference_velocity(targetline, cur_index, lookahead_distance, min_velocity, max_velocity, v):
    """
    Calculates the target velocity for the segment based on curvature constraints.
    Uses the physics formula: v_max = sqrt(a_lat / k), clamped between min and max velocity.
    
    If the car has 'infinite grip', treat `lat_accel_limit` as a 'Cornering Aggressiveness'
    factor. Higher values allow faster cornering speeds but may cause the controller to overshoot.

    Args:
        targetline (list or np.array): Array of [x, y] points.
        cur_index (int): Start index.
        lookahead_index (int): End index.
        min_velocity (float): Floor for the velocity (m/s).
        max_velocity (float): Ceiling for the velocity (m/s).

    Returns:
        float: The calculated reference velocity in m/s.
    """
    # 1. Get the sharpest curve in the lookahead
    # IDEA: don't use the cur_index since we don't want the car to slow when coming out of a turn

    # curvature_start_dist = 2500.0 / abs(v)
    curvature_start_dist = 24.0 # 28.0
    curvature_start_index = find_lookahead_point(targetline, cur_index, curvature_start_dist) # start index to look for computing max curvature

    # curvature_lookahead_dist = lookahead_distance - (abs(v) * 0.2) + (abs(v) * 0.74)
    curvature_lookahead_base = 20.0
    curvature_lookahead_scale = 1.8
    curvature_lookahead_dist = curvature_lookahead_base + abs(v) * curvature_lookahead_scale
    # curvature_lookahead_dist = 70.0
    curvature_lookahead_index = find_lookahead_point(targetline, cur_index, curvature_lookahead_dist) # end index to look for computing max curvature

    global k_max_current_turn
    k_neg_max, k_pos_max = max_curvature(targetline, curvature_start_index, curvature_lookahead_index) # returns absolute values
    k_max = k_neg_max + k_pos_max

    # 2. If track is straight (k ~ 0), we can go max speed
    if k_max < 1e-6:
        return max_velocity

    if (k_max < 0.05):
        k_max_current_turn = k_max # don't touch k_max_current_turn until we're done with the turn

    if k_neg_max < 0.03 or k_pos_max < 0.03: # are we only turning one direction?
        k_max = k_max / 2
    
    k_max = max(k_max, k_max_current_turn)
    k_max_current_turn = k_max

    # 3. Calculate velocity limit based on curvature and aggressiveness
    # v = sqrt(a / k)
    a_lat_max = 50.0
    amplification = 1.5

    v_limit = np.clip(np.sqrt(a_lat_max / max(k_max, 1e-6) ** amplification), min_velocity, max_velocity) # exponentiating k_max to amplify speed difference between straight line and curves
    target_v = v_limit

    return target_v

def calculate_reference_steering_angle(targetline, current_pos, lookahead_index, lwb, cur_heading):
    target_point = targetline[lookahead_index]

    # Calculate desired heading to reach target point
    dx = target_point[0] - current_pos[0]
    dy = target_point[1] - current_pos[1]
    desired_heading = np.arctan2(dy, dx)
    
    # Heading error (normalized to [-pi, pi])
    heading_error = desired_heading - cur_heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # Pure pursuit: compute curvature from heading error and lookahead distance
    lookaheadDist = np.linalg.norm(target_point - current_pos)
    curv = 2 * np.sin(heading_error) / max(lookaheadDist, 0.001)
    
    # Convert curvature to steering angle using bicycle model
    steer = np.arctan(lwb * curv)
    return steer

    # ---------------------------------

    # """
    # Calculates the reference steering angle (delta) using the Pure Pursuit algorithm.

    # This method calculates the steering angle needed to drive the vehicle from its
    # current position to a 'goal point' (the lookahead point) that lies on the path.

    # The formula used is: delta = arctan(2 * lwb * sin(alpha) / Ld)
    # where:
    #     - alpha is the angle between the car's current heading and the vector to the lookahead point.
    #     - Ld is the lookahead distance (distance from current position to lookahead point).
    #     - lwb is the wheelbase.

    # Args:
    #     targetline (list or np.array): Array of [x, y] points defining the path.
    #     cur_index (int): The current position index in targetline.
    #     lookahead_index (int): The index of the lookahead point (goal point).
    #     lwb (float): The vehicle's wheelbase (distance between front and rear axles).
    #     cur_heading (float): The vehicle's current heading angle in radians [0, 2pi].

    # Returns:
    #     float: The reference steering angle (delta) in radians. Positive is left, negative is right.
    # """
    # points = np.array(targetline)
    # n_points = len(points)
    
    # # 1. Get the coordinates of the current point and the lookahead point
    # p_current = points[cur_index % n_points]
    # p_lookahead = points[lookahead_index % n_points]
    
    # # 2. Calculate the vector from current position to lookahead point (p_lookahead - p_current)
    # dx = p_lookahead[0] - p_current[0]
    # dy = p_lookahead[1] - p_current[1]
    
    # # 3. Calculate the lookahead distance (Ld)
    # lookahead_distance = np.hypot(dx, dy)
    
    # # Check for zero distance to avoid division by zero
    # if lookahead_distance < 1e-6:
    #     return 0.0

    # # 4. Calculate the absolute angle (bearing) from the current position to the lookahead point
    # # This is the desired path angle
    # desired_angle = np.arctan2(dy, dx)
    
    # # 5. Calculate the difference (error) angle 'alpha'
    # # alpha = desired_angle - current_heading
    # alpha = desired_angle - cur_heading
    
    # # Normalize alpha to [-pi, pi] to ensure the shortest angular distance
    # # The steering formula relies on this for the sign and magnitude of sin(alpha)
    # alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    
    # # 6. Apply the Pure Pursuit steering formula
    # # delta = arctan(2 * lwb * sin(alpha) / Ld)
    # # 
    # steering_angle = np.arctan(2.0 * lwb * np.sin(alpha) / lookahead_distance)
    
    # return steering_angle

def lower_controller(state, desired, parameters):
    global prev_velocity_error, prev_steering_error
    
    sx, sy, delta, v, phi = state
    delta_r, vr = desired
    min_steering_velocity = parameters[7]
    max_steering_velocity = parameters[9]
    
    dt = 0.1  # Sampling time (100ms from simulator)
    
    # ========================================================================
    # STEERING CONTROL - PD Controller
    # ========================================================================
    delta_error = delta_r - delta
    
    # Calculate derivative of steering error
    delta_error_dot = (delta_error - prev_steering_error) / dt
    
    # PD control for steering
    Kp_delta = 17.0
    Kd_delta = 0  # Derivative gain for damping
    
    v_delta = Kp_delta * delta_error + Kd_delta * delta_error_dot
    v_delta = np.clip(v_delta, min_steering_velocity, max_steering_velocity)
    
    # Update previous error
    prev_steering_error = delta_error
    
    # ========================================================================
    # VELOCITY CONTROL - PD Controller
    # ========================================================================
    min_accel = parameters[8]
    max_accel = parameters[10]
    v_error = vr - v
    
    # Calculate derivative of velocity error
    v_error_dot = (v_error - prev_velocity_error) / dt
    
    # PD control for velocity
    Kp_v = 50.0
    Kd_v = 0  # Derivative gain for damping
    
    a = Kp_v * v_error + Kd_v * v_error_dot
    a = np.clip(a, min_accel, max_accel)
    
    # Update previous error
    prev_velocity_error = v_error
    
    return np.array([v_delta, a]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    # current state
    sx, sy, delta, v, phi = state
    target_line = racetrack.raceline

    current_pos = np.array([sx, sy])
    distances = np.linalg.norm(target_line - current_pos, axis=1)
    closest_idx = np.argmin(distances)

    min_steering_angle = parameters[1]
    max_steering_angle = parameters[4]
    min_velocity = parameters[2]
    max_velocity = parameters[5]
    lwb = parameters[0] # wheelbase

    min_lookahead = 4.0
    # max_lookahead = 20.0
    speed_ratio = v / max_velocity # [0, 1]
    # lookahead_dist = min_lookahead + (max_lookahead - min_lookahead) * speed_ratio
    lookahead_dist = min_lookahead + abs(v) * 0.2 + abs(calculate_cross_track_error(current_pos, target_line, closest_idx)) * 2

    target_point = find_lookahead_point(target_line, closest_idx, lookahead_dist)

    # REFERENCE VELOCITY
    vr = calculate_reference_velocity(target_line, closest_idx, lookahead_dist, min_velocity, max_velocity, v)
    vr = np.clip(vr, min_velocity, max_velocity)

    # REFERENCE STEERING
    delta_r = calculate_reference_steering_angle(target_line, current_pos, target_point, lwb, phi)
    delta_r = np.clip(delta_r, min_steering_angle, max_steering_angle)
    
    return np.array([delta_r, vr]).T