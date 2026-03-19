import numpy as np
import pandas as pd
from scipy import stats

def calculate_speed(coords, fps=20, unit='cm/f', exclude_z=False, joint_idx=None):
    """
    Calculate velocity vectors and speed for each joint.
    
    Args:
        coords: coordinate array (frames, joints, dimensions)
        fps: frames per second
        unit: 'cm/s' or 'cm/f'
        exclude_z: whether to exclude z-axis
        joint_idx: if specified, only calculate speed for this joint index
    
    Returns:
        vels: velocity vectors
        speeds: speed values
    """
    if unit == 'cm/s':
        vels = np.diff(coords, axis=0) * fps  # cm/s
    elif unit == 'cm/f':
        vels = np.diff(coords, axis=0)  # cm/f
    
    # Select specific joint if joint_idx is provided
    if joint_idx is not None:
        vels = vels[:, joint_idx:joint_idx+1, :]
    
    if exclude_z and vels.shape[2] == 3:
        vels = vels[:, :, :2]

    zeros = np.zeros((1, vels.shape[1], vels.shape[2]))
    vels = np.concatenate([zeros, vels], axis=0)

    speeds = np.linalg.norm(vels, axis=2)
    return vels, speeds


def calculate_angles(v, ref_v):
    ref_unit = ref_v / np.linalg.norm(ref_v)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = np.divide(v, v_norm, out=np.zeros_like(v), where=v_norm>0)

    dot = np.sum(v_unit * ref_unit, axis=1)
    angles = np.arccos(dot)
    return angles


def calculate_distance(coords, joint_pair=None, exclude_z=False, joint_idx=None, target_point=None):
    """
    Calculate distance between two joints or between a joint and a target point.
    
    Args:
        coords: coordinate array (frames, joints, dimensions)
        joint_pair: tuple of two joint indices (i1, i2) for joint-to-joint distance
        exclude_z: whether to exclude z-axis
        joint_idx: joint index for joint-to-point distance
        target_point: target point coordinates (can be 1D array for all frames or 2D for per-frame)
    
    Returns:
        v: displacement vectors
        dist: distance values
    """
    if joint_idx is not None and target_point is not None:
        # Calculate distance from joint to target point
        v1 = coords[:, joint_idx, :]
        v2 = np.array(target_point)
        
        # Handle target_point dimensions
        if v2.ndim == 1:
            # Single target point for all frames
            v2 = np.tile(v2, (v1.shape[0], 1))
        
        if exclude_z and v1.shape[1] == 3:
            v1 = v1[:, :2]
            v2 = v2[:, :2]
        v = v2 - v1
        dist = np.linalg.norm(v, axis=1)
    elif joint_pair is not None:
        # Calculate distance between two joints (original behavior)
        i1, i2 = joint_pair
        v1 = coords[:, i1, :]
        v2 = coords[:, i2, :]
        if exclude_z and v1.shape[1] == 3:
            v1 = v1[:, :2]
            v2 = v2[:, :2]
        v = v2 - v1
        dist = np.linalg.norm(v, axis=1)
    else:
        raise ValueError("Either joint_pair or (joint_idx and target_point) must be provided")
    
    return v, dist

def filter_behavior_in_time_range(bhvr_tuples_dict, time_range):
    """
    Filter behavior tuples within a time range
    
    Args:
        bhvr_tuples_dict: dict of behavior tuples
        time_range: (start, end) tuple
    
    Returns:
        filtered dict
    """
    filtered = {}
    start, end = time_range
    
    for sess_id, bhvr_dict in bhvr_tuples_dict.items():
        filtered[sess_id] = {}
        
        for bhvr_name, tuples in bhvr_dict.items():
            filtered_tuples = []
            
            for t_start, t_end in tuples:
                # Check if tuple overlaps with time range
                if t_start < end and t_end > start:
                    # Clip to time range
                    clipped_start = max(t_start, start)
                    clipped_end = min(t_end, end)
                    filtered_tuples.append((clipped_start, clipped_end))
            
            filtered[sess_id][bhvr_name] = filtered_tuples
    
    return filtered


def sync_start_time(bhvr_tuples_dict, sync_range):
    """
    Synchronize start time of behavior tuples by shifting the time axis.
    Sets sync_range[0] as the new time zero point.
    
    Args:
        bhvr_tuples_dict: dict of behavior tuples
        sync_range: (new_start, end) tuple - new_start will become time 0
    
    Returns:
        synchronized dict with time-shifted tuples (no clipping)
    """
    synced = {}
    new_start, new_end = sync_range
    
    for sess_id, bhvr_dict in bhvr_tuples_dict.items():
        synced[sess_id] = {}
        
        for bhvr_name, tuples in bhvr_dict.items():
            synced_tuples = []
            
            for t_start, t_end in tuples:
                # Simply shift time axis - no clipping
                synced_start = t_start - new_start
                synced_end = t_end - new_start
                synced_tuples.append((synced_start, synced_end))
            
            synced[sess_id][bhvr_name] = synced_tuples
    
    return synced


def _calculate_tuples_to_bout_durations(tuple_list, fps=None):
    """
    Calculate bout durations from a list of (start, end) tuples.
    
    Args:
        tuple_list: list of (start, end) tuples, e.g., [(0, 10), (20, 35)]
        fps: frames per second, if provided, converts frames to seconds
    
    Returns:
        list of bout durations
    
    Example:
        Input:  [(0, 10), (20, 35), (50, 60)]
        Output: [10, 15, 10]  # if fps=None
                [0.5, 0.75, 0.5]  # if fps=20

        Input:  []
        Output: [np.nan]
    """
    if len(tuple_list) == 0:
        return [np.nan]
    
    durations = []
    for start, end in tuple_list:
        if fps is not None:
            durations.append((end - start) / fps)
        else:
            durations.append(end - start)
    
    return durations


def _calculate_tuples_to_bout_count(tuple_list):
    """
    Count the number of behavior bouts from a list of (start, end) tuples.
    
    Args:
        tuple_list: list of (start, end) tuples, e.g., [(0, 10), (20, 35)]
    
    Returns:
        list with single bout count value
    
    Example:
        Input:  [(0, 10), (20, 35), (50, 60)]
        Output: [3]
        
        Input:  []
        Output: [0]
    """
    return [len(tuple_list)]


def _calculate_tuples_to_average_bout_duration(tuple_list, fps=None, method='mean'):
    """
    Calculate average bout duration from a list of (start, end) tuples.
    
    Args:
        tuple_list: list of (start, end) tuples, e.g., [(0, 10), (20, 35)]
        fps: frames per second, if provided, converts frames to seconds
        method: 'mean', 'median' or 'mode' for averaging method
    
    Returns:
        list with single average bout duration value
    
    Example:
        Input:  [(0, 10), (20, 35), (50, 60)]
        Output: [11.667]  # if fps=None (frames)
                [0.583]   # if fps=20 (seconds)
    """
    if len(tuple_list) == 0:
        return [np.nan]
    
    if method == 'mean':
        durations = [(end - start) / fps if fps is not None else (end - start) 
                     for start, end in tuple_list]
        mean_duration = np.mean(durations)
        return [mean_duration]
    
    elif method == 'median':
        durations = [(end - start) / fps if fps is not None else (end - start) 
                     for start, end in tuple_list]
        median_duration = np.median(durations)
        return [median_duration]
    
    elif method == 'mode':
        durations = [(end - start) / fps if fps is not None else (end - start) 
                     for start, end in tuple_list]
        mode_result = stats.mode(durations)
        mode_duration = mode_result.mode[0]
        return [mode_duration]


def calculate_behavior_bout_durations(tuples_data, fps=None):
    """
    Calculate behavior bout durations from tuples (start, end) format.
    Supports both list and nested dict.

    Args:
        tuples_data: list of tuples or nested dict where leaf nodes are lists of (start, end) tuples
        fps: frames per second, if provided, converts frames to seconds
    
    Returns:
        list with bout duration (if input is list) or nested dict with same structure (if input is dict)
    
    Example:
        Input:  {'session1': {'bhvr1': [(0, 10), (20, 35)]}}
        Output: {'session1': {'bhvr1': [10, 15]}}

    """
    # Direct list processing
    if isinstance(tuples_data, list):
        durations_list = _calculate_tuples_to_bout_durations(tuples_data, fps)
        return durations_list
    
    # Dictionary processing (recursive)
    durations_dict = {}
    for key, values in tuples_data.items():
        if isinstance(values, dict):
            # Recursive case: nested dict
            durations_dict[key] = calculate_behavior_bout_durations(values, fps=fps)
        else:
            # Base case: process list
            durations_dict[key] = _calculate_tuples_to_bout_durations(values, fps)
    
    return durations_dict


def calculate_behavior_bout_count(tuples_data):
    """
    Calculate number of behavior bouts (occurrences).
    Supports both list and nested dict.
    
    Args:
        tuples_data: list of tuples or nested dict where leaf nodes are lists of (start, end) tuples
    
    Returns:
        list with bout count (if input is list) or nested dict with same structure (if input is dict)
    
    Example:
        # List input:
        Input:  [(0, 10), (20, 35), (50, 60)]
        Output: [3]
        
        # Dict input:
        Input:  {'session1': {'walking': [(0, 10), (20, 35), (50, 60)]}}
        Output: {'session1': {'walking': [3]}}
    """
    # Direct list processing
    if isinstance(tuples_data, list):
        bout_count = _calculate_tuples_to_bout_count(tuples_data)
        return bout_count
    
    # Dictionary processing (recursive)
    count_dict = {}
    for key, values in tuples_data.items():
        if isinstance(values, dict):
            # Recursive case: nested dict
            count_dict[key] = calculate_behavior_bout_count(values)
        else:
            # Base case: process list
            count_dict[key] = _calculate_tuples_to_bout_count(values)
    
    return count_dict


def calculate_behavior_average_bout_duration(tuples_data, fps=None):
    """
    Calculate average behavior duration from tuples (start, end) format.
    Supports both list input and nested dictionary structures.
    
    Args:
        tuples_data: list of (start, end) tuples or nested dict where leaf nodes are lists of tuples
        fps: frames per second, if provided, converts frames to seconds
    
    Returns:
        list with average duration (if input is list) or nested dict with same structure (if input is dict)
    
    Example:
        # List input:
        Input:  [(0, 10), (20, 35), (50, 60)]
        Output: [11.667]
        
        # Dict input:
        Input:  {'session1': {'walking': [(0, 10), (20, 35), (50, 60)]}}
        Output: {'session1': {'walking': [11.667]}}
    """
    # Direct list processing
    if isinstance(tuples_data, list):
        avg_duration_list = _calculate_tuples_to_average_bout_duration(tuples_data, fps)
        return avg_duration_list
    
    # Dictionary processing (recursive)
    avg_duration_dict = {}
    for key, values in tuples_data.items():
        if isinstance(values, dict):
            # Recursive case: nested dict
            avg_duration_dict[key] = calculate_behavior_average_bout_duration(values, fps=fps)
        else:
            # Base case: process list
            avg_duration_dict[key] = _calculate_tuples_to_average_bout_duration(values, fps)
    
    return avg_duration_dict

