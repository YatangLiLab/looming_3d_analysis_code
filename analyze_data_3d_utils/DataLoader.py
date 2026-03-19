import numpy as np
import pandas as pd
import scipy
import os

def load_coord_data(mat_dir, pred_type='pred', check_sess=True):
    """
    Load coordinate data from .mat files.
    
    Args:
        mat_dir: directory containing .mat files
        pred_type: 'pred' or 'com'
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and coordinate arrays as values
    """

    # Search for .mat files.
    mat_files = []
    for g in range(1, 6):
        for m in range(1, 15):
            for d in range(1, 6):
                mat_path = os.path.join(mat_dir, f"exp_{g}{m}{d}/")
                if pred_type == 'pred':
                    mat_file = os.path.join(mat_path, f"save_data_AVG0.mat")
                elif pred_type == 'com':
                    mat_file = os.path.join(mat_path, f"com3d_used.mat")
                if os.path.exists(mat_path):
                    if os.path.exists(mat_file):
                        mat_files.append((mat_file, g, m, d))  # (file, group, mouse, day)
                    else:
                        print(f"File not found: {mat_file}")

    # Load data from .mat files.
    coord_dict = {}
    for mat_file, g, m, d in mat_files:
        ses_id = f"G{g}M{m}D{d}"
        data = scipy.io.loadmat(mat_file)
        coord_data = data['com'] if pred_type == 'com' else data['pred']
        # Reorganize predict data.
        if pred_type != 'com':
            joint_names = [
                'EarL', 'EarR', 'Snout', 'SpineF', 'SpineM', 'TailB', 'TailM', 
                'ForepawL', 'WristL', 'ElbowL', 'ShoulderL', 
                'ForepawR', 'WristR', 'ElbowR', 'ShoulderR', 
                'HindpawL', 'AnkleL', 'KneeL', 
                'HindpawR', 'AnkleR', 'KneeR'
            ]
            n_frames = coord_data.shape[0]
            n_joints = coord_data.shape[2]
            coords = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
            for i in range(n_joints):
                point_data = coord_data[:, :, i]  # (frame, 3)
                coords[:, i, :] = point_data
        else:
            joint_names = ['com']
            coords = coord_data.reshape(coord_data.shape[0], 1, 3)
        # Correct coordinates.
        correct_data = np.array([260.5, 245, 10])  # It depends on the position of the chessboard. mm.
        coords_corrected = np.zeros_like(coords)
        coords_corrected[:, :, 0] = -coords[:, :, 1] + correct_data[0]
        coords_corrected[:, :, 1] = -coords[:, :, 0] + correct_data[1]
        coords_corrected[:, :, 2] =  coords[:, :, 2] + correct_data[2]
        coords_corrected /= 10.0  # Convert mm → cm
        # Store in dict
        coord_dict[ses_id] = coords_corrected
    # Check loaded data.
    print('Number of sessions:', len(coord_dict.keys()))
    if check_sess:
        check_sess_id(coord_dict)
    return coord_dict


def load_syl_data(syl_dir, prefix, suffix, check_sess=True):
    """
    Load syllable/state data from CSV files.
    
    Args:
        syl_dir: directory containing syllable CSV files
        prefix: filename prefix to filter
        suffix: filename suffix to filter
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and syllable arrays as values
    """
    # Search for CSV files.
    csv_files = []
    for g in range(1, 6):
        for m in range(1, 15):
            for d in range(1, 6):
                csv_file = os.path.join(syl_dir, f"{prefix}exp_{g}{m}{d}{suffix}")
                if os.path.exists(csv_file):
                    csv_files.append((csv_file, g, m, d))  # (file, group, mouse, day)
    # Load syllable data from CSV files.
    syl_dict = {}
    for csv_file, g, m, d in csv_files:
        syl_array = np.genfromtxt(csv_file, delimiter=',', skip_header=1, usecols=[0], dtype=int)
        sess_id = f"G{g}M{m}D{d}"
        syl_dict[sess_id] = syl_array
    if check_sess:
        check_sess_id(syl_dict)
    return syl_dict


def load_lsf_csv(csv_path, check_sess=True):
    """
    Load looming start frame (lsf) from CSV file
    
    Args:
        csv_path: path to CSV file containing lsf data
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and lists of lsf values
    """
    lsf_dict = {}
    df = pd.read_csv(csv_path)
    # Extract sess_id and lsf data.
    for _, row in df.iterrows():
        # Skip rows with missing data
        if pd.isna(row['trial_id']) or pd.isna(row['looming_start_frame']):
            lsf_dict[row['trial_id']] = []
            continue
        trial_id = row['trial_id']
        # Extract sess_id by removing trial number (e.g., 'G1M1D1T1' -> 'G1M1D1')
        sess_id = trial_id.rsplit('T', 1)[0]
        # Get looming start frame from correct column
        lsf_val = row['looming_start_frame']
        if sess_id not in lsf_dict:
            lsf_dict[sess_id] = []
        lsf_dict[sess_id].append(int(lsf_val))
    if check_sess:
        check_sess_id(lsf_dict)
    return lsf_dict


def check_sess_id(dict):
    """
    Check session IDs in the given dictionary and print them.
    
    Args:
        dict: dictionary with session IDs as keys
    """
    keys = list(dict.keys())
    print('Number of sessions:', len(keys))
    for i in range(0, len(keys), 5):
        print(keys[i:i+5])