import numpy as np
import pandas as pd
import jax.numpy as jnp
na = jnp.newaxis
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import savgol_filter
from analyze_data_3d_utils.DataAnalyzer import calculate_speed, calculate_distance, calculate_angles


def inverse_rigid_transform(Y, v, h):
    """
    Apply the inverse of the rigid transform consisting of
    rotation by h and translation by v to a set of keypoint
    observations.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.

    Returns
    -------
    Y_transformed: jax array of shape (..., k, d)
        Rigidly transformed positions.
    """
    return apply_rotation(Y - v[..., na, :], -h)


def center_embedding(n):
    """
    Generates a matrix ``Gamma`` that maps from a (n-1)-dimensional
    vector space  to the space of k-tuples with zero mean

    Parameters
    ----------
    n : int
        Number of keypoints.

    Returns
    -------
    Gamma: jax array of shape (n, n - 1)
        Matrix to map to centered embedded space.
    """
    X = jnp.tril(jnp.ones((n, n)), k=-1)[1:]
    X = jnp.eye(n)[1:] - X / X.sum(1)[:, na]
    X = X / jnp.sqrt((X**2).sum(1))[:, na]
    return X.T


def apply_rotation(Y, h):
    """
    Rotate ``Y`` by ``h`` radians.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    h : jax array
        Heading angles.

    Returns
    ------
    Y_rot : jax array of shape (..., k, d)
        Rotated keypoint observations.
    """
    d = Y.shape[-1]
    rot_matrix = angle_to_rotation_matrix(h, d)
    return jnp.einsum("...kj,...ij->...ki", Y, rot_matrix)


def angle_to_rotation_matrix(h, d=3):
    """
    Create rotation matrices from an array of angles. If
    ``d > 2`` then rotation is performed in the first two dims.

    Parameters
    ----------
    h : jax array of shape (N, T)
        Heading angles.
    d : int, default=3
        Keypoint dimensionality (either 2 or 3).

    Returns
    ------
    m: jax array of shape (..., d, d)
        Rotation matrices.
    """
    m = jnp.tile(jnp.eye(d), (*h.shape, 1, 1))
    m = m.at[..., 0, 0].set(jnp.cos(h))
    m = m.at[..., 1, 1].set(jnp.cos(h))
    m = m.at[..., 0, 1].set(-jnp.sin(h))
    m = m.at[..., 1, 0].set(jnp.sin(h))
    return m


def vector_to_angle(V):
    """
    Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is multidimensional, the first
    n-1 dimensions are treated as batch dims.

    Parameters
    ----------
    V : jax array of shape (..., 2)
        Batch of 2D vectors.

    Returns
    ------
    h : jax array
        Rotation angles in radians.
    """
    return jnp.arctan2(V[..., 1], V[..., 0])


def align_egocentric(Y, anterior_idxs, posterior_idxs, **kwargs):
    """
    Perform egocentric alignment of keypoints by translating the
    centroid to the origin and rotatating so that the vector pointing
    from the posterior bodyparts toward the anterior bodyparts is
    proportional to (1,0).

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    anterior_idxs : iterable of ints
        Anterior keypoint indices for heading initialization.
    posterior_idxs : iterable of ints
        Posterior keypoint indices for heading initialization.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    Y_aligned : jax array of shape (..., k, d)
        Aligned keypoint coordinates.
    v : jax array of shape (..., d)
        Centroid positions that were used for alignment.
    h : jax array
        Heading angles that were used for alignment.
    """
    posterior_loc = Y[..., posterior_idxs, :2].mean(-2)
    anterior_loc = Y[..., anterior_idxs, :2].mean(-2)
    h = vector_to_angle(anterior_loc - posterior_loc)
    v = Y.mean(-2)
    Y_aligned = inverse_rigid_transform(Y, v, h)
    return Y_aligned, v, h


def get_distance_to_medoid(coordinates: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance from each keypoint to the medoid (median position)
    of all keypoints at each frame.

    Parameters
    -------
    coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates where keypoint_dim is 2 or 3.

    Returns
    -------
    distances: ndarray of shape (n_frames, n_keypoints)
        Euclidean distances from each keypoint to the medoid position at each frame.
    """
    medoids = np.median(coordinates, axis=1)  # (n_frames, keypoint_dim)
    return np.linalg.norm(coordinates - medoids[:, None, :], axis=-1)  # (n_frames, n_keypoints)


def find_medoid_distance_outliers(
    coordinates: np.ndarray, outlier_scale_factor: float = 6.0
) -> dict[str, np.ndarray]:
    """Identify keypoint distance outliers using Median Absolute Deviation (MAD).

    Keypoints are considered outliers when their distance to the medoid at a given timepoint differs
    from its median value by a multiple of the median absolute deviation (MAD) for that keypoint.

    Parameters
    -------
    coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates where keypoint_dim is 2 or 3. Only the first two dimensions (x, y) are
        used for distance calculations.

    outlier_scale_factor: float, default=6.0
        Multiplier used to set the outlier threshold. Higher values result in fewer outliers.

    **kwargs
        Additional keyword arguments (ignored), usually overflow from **config().

    Returns
    -------
    result: dict with the following items

        mask: ndarray of shape (n_frames, n_keypoints)
            Boolean array where True indicates outlier keypoints.

        thresholds: ndarray of shape (n_keypoints,)
            Distance thresholds used to classify outlier timepoints for each keypoint.
    """
    distances = get_distance_to_medoid(coordinates)  # (n_frames, n_keypoints)
    medians = np.median(distances, axis=0)  # (n_keypoints,)
    MADs = np.median(np.abs(distances - medians[None, :]), axis=0)  # (n_keypoints,)
    outlier_thresholds = MADs * outlier_scale_factor + medians  # (n_keypoints,)
    outlier_mask = distances > outlier_thresholds[None, :]  # (n_frames, n_keypoints)
    return {"mask": outlier_mask, "thresholds": outlier_thresholds}

def find_outofbox_outliers(coords, box_bounds):
    n_frames, n_joints, _ = coords.shape
    mask_outbox = np.zeros((n_frames, n_joints), dtype=bool)
    
    if box_bounds is None:
        return {"mask": mask_outbox, "bounds": None}
    
    x_min, x_max, y_min, y_max, z_min, z_max = box_bounds
    
    # Check each dimension
    x_out = (coords[:, :, 0] < x_min) | (coords[:, :, 0] > x_max)
    y_out = (coords[:, :, 1] < y_min) | (coords[:, :, 1] > y_max)
    z_out = (coords[:, :, 2] < z_min) | (coords[:, :, 2] > z_max)
    
    mask_outbox = x_out | y_out | z_out
    
    return {"mask": mask_outbox, "bounds": box_bounds}
    
def find_vectorized_speed_outliers(raw_coords, joint_names, outlier_scale_factor=6.0):
    n_frames, n_joints, _ = raw_coords.shape
    mask_vel = np.zeros((n_frames, n_joints), dtype=bool)
    thresh_vel = np.zeros(n_joints, dtype=float)
    vels, speeds = calculate_speed(raw_coords)
    # calculate_speed already prepends zeros for first frame, so use speeds directly
    med = np.median(speeds, axis=0)
    mad = np.median(np.abs(speeds - med[None, :]), axis=0)
    thresh_vel = med + mad * outlier_scale_factor
    mask_vel = speeds > thresh_vel[None, :]
    return {"mask": mask_vel, "thresholds": thresh_vel}


def find_angle_outliers(ego_coords, joint_names, joint_orders, outlier_scale_factor=6.0):
    n_frames, n_joints, _ = ego_coords.shape
    mask_angle = np.zeros((n_frames, n_joints), dtype=bool)
    thresh_angle = np.zeros(n_joints, dtype=float)
    for i, j in enumerate(joint_names):
        joint_vec = ego_coords[:, i, :]
        angles = calculate_angles(joint_vec, np.array([1, 0, 0]))
        med = np.median(angles)
        mad = np.median(np.abs(angles - med))
        thresh_angle[i] = med + mad * outlier_scale_factor
        outlier_frames = angles > thresh_angle[i]
        for f in np.where(outlier_frames)[0]:
            if joint_orders:
                rank = joint_orders.get(j, 0)
                if rank > 2:
                    mask_angle[f, i] = True
            else:
                mask_angle[f, i] = True
    return {"mask": mask_angle, "thresholds": thresh_angle}


def find_displacement_outliers(ego_coords, joint_names, skel_conns, joint_creds, outlier_scale_factor=6.0):
    n_frames, n_joints, _ = ego_coords.shape
    mask_disp = np.zeros((n_frames, n_joints), dtype=bool)
    thresholds_disp = {}
    for j1, j2 in skel_conns:
        i1, i2 = joint_names.index(j1), joint_names.index(j2)
        v = ego_coords[:, i1, :] - ego_coords[:, i2, :]
        med = np.median(v, axis=0)
        mad = np.median(np.abs(v - med), axis=0)
        thresh = med + mad * outlier_scale_factor
        thresholds_disp[(j1, j2)] = thresh
        outlier_frames = np.any(np.abs(v - med) > mad * outlier_scale_factor, axis=1)
        if joint_creds is not None:
            if joint_creds.get(j1, 1) < joint_creds.get(j2, 1):
                mask_disp[:, i1] |= outlier_frames
            else:
                mask_disp[:, i2] |= outlier_frames
        else:
            mask_disp[:, i2] |= outlier_frames
    return {"mask": mask_disp, "thresholds": thresholds_disp}

def find_RULE_outliers(raw_coords, 
                       joint_names, 
                       joint_orders, 
                       skel_conns, 
                       joint_creds, 
                       prev_outliers=None, 
                       outlier_scale_factors={'velocity': 5, 'angle': 5, 'displacement': 5},
                       box_bounds=(-5, 55, -5, 55, -3, 20)):
    
    # Initialize
    n_frames, n_joints, _ = raw_coords.shape
    mask_all = np.zeros((n_frames, n_joints), dtype=bool)
    thresholds_dict = {}
    
    # Out-of-box outliers (if box_bounds provided)
    result_outbox = find_outofbox_outliers(raw_coords, box_bounds)
    mask_all |= result_outbox['mask']
    if result_outbox['bounds'] is not None:
        thresholds_dict['box_bounds'] = result_outbox['bounds']

    # Velocity outliers.
    result_vel = find_vectorized_speed_outliers(
        raw_coords, 
        joint_names,
        outlier_scale_factor=outlier_scale_factors['velocity'])
    mask_all |= result_vel['mask']
    thresholds_dict['velocity'] = result_vel['thresholds']

    # Egocentric alignment.
    anterior_idxs = [joint_names.index(j) for j in ['EarL', 'EarR', 'Snout']]
    posterior_idxs = [joint_names.index(j) for j in ['SpineM', 'TailB']]
    ego_coords, v, h = align_egocentric(raw_coords, anterior_idxs, posterior_idxs)

    # Angle outliers.
    result_angle = find_angle_outliers(
        ego_coords, 
        joint_names, 
        joint_orders, 
        outlier_scale_factor=outlier_scale_factors['angle'])
    mask_all |= result_angle['mask']
    thresholds_dict['angle'] = result_angle['thresholds']
    
    # Displacement outliers.
    result_disp = find_displacement_outliers(
        ego_coords, 
        joint_names,
        skel_conns, 
        joint_creds, 
        outlier_scale_factor=outlier_scale_factors['displacement'])

    mask_all |= result_disp['mask']
    thresholds_dict['displacement'] = result_disp['thresholds']

    if prev_outliers is not None:
        mask_all |= prev_outliers.get('mask', np.zeros_like(mask_all))
        thresholds_dict['medoid'] = prev_outliers['thresholds']  # ndarray
    return {'mask': mask_all, 'thresholds': thresholds_dict}


def interpolate_along_axis(x, xp, fp, axis=0):
    """Linearly interpolate along a given axis.

    Parameters
    ----------
    x: 1D array
        The x-coordinates of the interpolated values
    xp: 1D array
        The x-coordinates of the data points
    fp: ndarray
        The y-coordinates of the data points. fp.shape[axis] must
        be equal to the length of xp.

    Returns
    -------
    x_interp: ndarray
        The interpolated values, with the same shape as fp except along the
        interpolation axis.
    """
    assert len(xp.shape) == len(x.shape) == 1
    assert fp.shape[axis] == len(xp)
    assert len(xp) > 0, "xp must be non-empty; cannot interpolate without datapoints"

    fp = np.moveaxis(fp, axis, 0)
    shape = fp.shape[1:]
    fp = fp.reshape(fp.shape[0], -1)

    x_interp = np.zeros((len(x), fp.shape[1]))
    for i in range(fp.shape[1]):
        x_interp[:, i] = np.interp(x, xp, fp[:, i])
    x_interp = x_interp.reshape(len(x), *shape)
    x_interp = np.moveaxis(x_interp, 0, axis)
    return x_interp


def interpolate_keypoints(coordinates, outliers):
    """Use linear interpolation to impute the coordinates of outliers.

    Parameters
    ----------
    coordinates : ndarray of shape (num_frames, num_keypoints, dim)
        Keypoint observations.
    outliers : ndarray of shape (num_frames, num_keypoints)
        Binary indicator whose true entries are outlier points.

    Returns
    -------
    interpolated_coordinates : ndarray with same shape as `coordinates`
        Keypoint observations with outliers imputed.
    """
    interpolated_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[1]):
        xp = np.nonzero(~outliers[:, i])[0]
        if len(xp) > 0:
            interpolated_coordinates[:, i, :] = interpolate_along_axis(
                np.arange(coordinates.shape[0]), xp, coordinates[xp, i, :]
            )
    return interpolated_coordinates


def smooth_data(data, method='gaussian', **kwargs):
    """
    Smooth data using various filtering methods. Supports 1D, 2D, and 3D arrays.
    
    Args:
        data: numpy array
            - 1D: (n_frames,) - e.g., speed or single time series
            - 2D: (n_frames, n_features) - e.g., xy coordinates, multiple joints
            - 3D: (n_frames, n_joints, n_dims) - e.g., 3D pose coordinates
        method: str, smoothing method to use
            - 'ma': Simple moving average
            - 'ewma': Exponential weighted moving average
            - 'gaussian': Gaussian filter
            - 'savgol': Savitzky-Golay filter
        **kwargs: method-specific parameters
            For 'ma': window_size (default=5)
            For 'ewma': alpha (default=0.3, range 0-1, smaller=more smoothing)
            For 'gaussian': sigma (default=2.0)
            For 'savgol': window_length (default=11), polyorder (default=3)
    
    Returns:
        smoothed array with same shape as input
    """

    def _rolling_1d(arr, window_size):
        """Apply pandas rolling to 1D array."""
        result = pd.Series(arr).rolling(window=window_size, center=True).to_numpy()
        return result
       
    def _ma_1d(arr, window_size=5):
        """Apply moving average to 1D array."""
        kernel = np.ones(window_size) / window_size
        result = np.convolve(arr, kernel, mode='same')
        return result
    
    def _ewma_1d(arr, alpha=0.3):
        """Apply EWMA to 1D array."""
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for t in range(1, len(arr)):
            result[t] = alpha * arr[t] + (1 - alpha) * result[t-1]
        return result
    
    def _gaussian_filter_1d(arr, sigma=2.0):
        """Apply Gaussian filter to 1D array."""
        result = gaussian_filter1d(arr, sigma=sigma)
        return result
    
    def _savgol_filter_1d(arr, window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter to 1D array."""
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        # Ensure window_length > polyorder
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        # Ensure window_length <= data length
        if window_length > len(arr):
            window_length = len(arr) if len(arr) % 2 == 1 else len(arr) - 1
            window_length = max(window_length, polyorder + 2)
        result = savgol_filter(arr, window_length, polyorder)
        return result
    
    # Select smoothing function based on method
    if method == 'rolling':
        window_size = kwargs.get('window_size', 5)
        smooth_1d = lambda a: _rolling_1d(a, window_size)
    elif method == 'ma':
        window_size = kwargs.get('window_size', 5)
        smooth_1d = lambda a: _ma_1d(a, window_size)
    elif method == 'ewma':
        alpha = kwargs.get('alpha', 0.3)
        smooth_1d = lambda a: _ewma_1d(a, alpha)
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 2.0)
        smooth_1d = lambda a: _gaussian_filter_1d(a, sigma)
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)
        smooth_1d = lambda a: _savgol_filter_1d(a, window_length, polyorder)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    # Apply smoothing based on data dimensionality
    if data.ndim == 1:
        # 1D: (n_frames,) - directly apply
        return smooth_1d(data)
    
    elif data.ndim == 2:
        # 2D: (n_frames, n_features) - apply to each feature
        smoothed = np.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed[:, i] = smooth_1d(data[:, i])
        return smoothed
    
    elif data.ndim == 3:
        # 3D: (n_frames, n_joints, n_dims) - apply to each joint-dimension pair
        smoothed = np.zeros_like(data)
        for j in range(data.shape[1]):
            for c in range(data.shape[2]):
                smoothed[:, j, c] = smooth_1d(data[:, j, c])
        return smoothed
    
    else:
        raise ValueError(f"Unsupported array dimension: {data.ndim}. Only 1D, 2D, and 3D arrays are supported.")



def simplify_coord_dict(coord_dict, joint_names, limb_combos):
    """
    Simplify coordinates by combining related joints.
    
    Args:
        coord_dict: dict of coordinate arrays
        joint_names: list of original joint names
        limb_combos: dict specifying how to combine joints
    
    Returns:
        simplified coord_dict and simplified joint names
    """
    simp_coord_dict = {}
    simp_joint_names = []
    # Define simplification mapping
    if not limb_combos:
        # Default: use all joints
        simp_joint_names = joint_names
        simp_coord_dict = coord_dict
    else:
        for simp_name, joint_list in limb_combos.items():
            simp_joint_names.append(simp_name)
        for sess_id, coords in coord_dict.items():
            simp_coords = []
            for simp_name, joint_list in limb_combos.items():
                joint_indices = [joint_names.index(j) for j in joint_list if j in joint_names]
                if joint_indices:
                    # Average positions of combined joints
                    combined = np.mean(coords[:, joint_indices, :], axis=1)
                    simp_coords.append(combined)
            simp_coord_dict[sess_id] = np.stack(simp_coords, axis=1)
    return simp_coord_dict, simp_joint_names


def map_syllabel_to_behavior(syl_dict, bhvr_map):
    """
    Map syllable labels to behavior names.
    
    Args:
        syl_dict: dict of syllable label arrays
        bhvr_map: dict mapping syllable numbers to behavior names
    
    Returns:
        dict of behavior label arrays
    """
    bhvr_dict = {}
    for sess_id, syl_labels in syl_dict.items():
        bhvr_labels = np.array([bhvr_map.get(int(syl), 'other') for syl in syl_labels], dtype=object)
        bhvr_dict[sess_id] = bhvr_labels
    return bhvr_dict


def filter_short_states(state_series, min_frames=10):
    """
    Filter out short state changes (< min_frames).
    If a continuous segment is shorter than min_frames, replace it with the previous state.
    
    Args:
        state_series: pandas Series of 0s and 1s (or boolean values)
        min_frames: minimum number of consecutive frames to keep a state
    
    Returns:
        filtered pandas Series with the same index
    """    
    # Convert to numpy array for processing
    state_array = state_series.values.astype(int)
    filtered = state_array.copy()
    n = len(filtered)
    i = 0
    while i < n:
        current_state = filtered[i]
        # Find the end of current segment
        j = i
        while j < n and filtered[j] == current_state:
            j += 1
        segment_length = j - i
        # If segment is too short, replace with previous state (if exists)
        if segment_length < min_frames and i > 0:
            filtered[i:j] = filtered[i-1]
        i = j
    # Convert back to pandas Series with original index
    return pd.Series(filtered, index=state_series.index, name=state_series.name)


def convert_bhvr_kpms2series(bhvr_dict):
    """
    Convert behavior array to pandas Series format
    
    Args:
        bhvr_dict: dict of behavior label arrays
    
    Returns:
        dict of behavior Series
    """
    bhvr_series_dict = {}
    for sess_id, bhvr_labels in bhvr_dict.items():
        bhvr_series_dict[sess_id] = {}
        unique_bhvrs = np.unique(bhvr_labels)
        for bhvr in unique_bhvrs:
            mask = bhvr_labels == bhvr
            series = pd.Series(mask.astype(int), name=bhvr)
            bhvr_series_dict[sess_id][bhvr] = series
    return bhvr_series_dict


def convert_bhvr_series2tuples(bhvr_series_dict):
    """
    Convert behavior Series to (start, end) tuple format
    
    Args:
        bhvr_series_dict: dict of behavior Series dicts
    
    Returns:
        dict of behavior tuples
    """
    bhvr_tuples_dict = {}
    for sess_id, bhvr_series in bhvr_series_dict.items():
        bhvr_tuples_dict[sess_id] = {}
        for bhvr_name, series in bhvr_series.items():
            tuples = []
            if isinstance(series, dict):
                # If it's a dict, get the series from it
                series_data = series.get(bhvr_name, pd.Series([]))
            else:
                series_data = series
            # Find contiguous regions
            diff = np.diff(series_data.astype(int), prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for start, end in zip(starts, ends):
                tuples.append((start, end))
            bhvr_tuples_dict[sess_id][bhvr_name] = tuples
    return bhvr_tuples_dict


def save_bhvr_dicts(bhvr_dict, save_folder, save_format='csv', file_prefix=''):
    """
    Save behavior dictionaries to files.

    Args:
        bhvr_dict: dict of behavior data
        save_folder: folder to save files
        save_format: 'csv' or 'pickle'
        file_prefix: prefix for filenames
    """
    import os
    os.makedirs(save_folder, exist_ok=True)
    
    for sess_id, bhvr_data in bhvr_dict.items():
        filename = f"{file_prefix}{sess_id}.{save_format}"
        filepath = os.path.join(save_folder, filename)
        
        if save_format == 'csv':
            if isinstance(bhvr_data, dict):
                df = pd.DataFrame(bhvr_data)
            else:
                df = pd.DataFrame(bhvr_data)
            df.to_csv(filepath, index=False)
