import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from analyze_data_3d_utils.DataProcessor import align_egocentric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _cube_bounds(points):
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    span = np.max(max_vals - min_vals)
    center = (max_vals + min_vals) / 2.0
    half = span / 2.0
    lower = center - half
    upper = center + half
    return lower, upper


def draw_mean_skeleton(coords, joint_names, skel_conns):
    """
    Draw mean skeleton with interactive 3D viewer.

    Args:
        coords: array of shape (n_frames, n_joints, 3)
        joint_names: list of joint names
        skel_conns: list of (joint1_idx, joint2_idx) skeleton connections
    """
    mean_coords = np.mean(coords, axis=0)
    lower, upper = _cube_bounds(mean_coords)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=mean_coords[:, 0],
            y=mean_coords[:, 1],
            z=mean_coords[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.9)
        )
    )

    for j1, j2 in skel_conns:
        i1 = joint_names.index(j1) if isinstance(j1, str) else j1
        i2 = joint_names.index(j2) if isinstance(j2, str) else j2
        if i1 < len(mean_coords) and i2 < len(mean_coords):
            pts = mean_coords[[i1, i2], :]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode='lines',
                    line=dict(color='blue', width=4)
                )
            )

    fig.update_layout(
        title='Mean Skeleton',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[lower[0], upper[0]]),
            yaxis=dict(range=[lower[1], upper[1]]),
            zaxis=dict(range=[lower[2], upper[2]]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
    return


def visualize_projected_2d(self, frame_range, step):
    """
    TODO: Project 3D coordinates to 2D plane and visualize.
    """
    print("2D projection visualization not yet implemented.")
    return []


def _assign_colors(c_maps, joint_used):
    colors = []
    for j in joint_used:
        for joints, color in c_maps.values():
            if j in joints:
                colors.append(color)
                break
    return colors


def _draw_skeleton(ax, coord_array, frame_idx, joint_used, skel_conn, colors):
    coords = {}
    joint_indices = {j: i for i, j in enumerate(joint_used)}
    for joint in joint_used:
        idx = joint_indices[joint]
        coords[joint] = coord_array[frame_idx, idx, :] 
    xs = [coords[j][0] for j in joint_used]
    ys = [coords[j][1] for j in joint_used]
    zs = [coords[j][2] for j in joint_used]
    ax.scatter(xs, ys, zs, c=colors, s=5)
    for j1, j2 in skel_conn:
        if j1 in coords and j2 in coords:
            ax.plot([coords[j1][0], coords[j2][0]],
                    [coords[j1][1], coords[j2][1]],
                    [coords[j1][2], coords[j2][2]], c='gray')
    return ax

def _draw_arena(ax, nest=True, looming=True):
    # Draw arena rect.
    arena_coord = [[[0, 0, 0], [0, 50, 0], [50, 50, 0], [50, 0, 0]]]
    arena = Poly3DCollection(arena_coord, edgecolors='black', linewidths=1, facecolors=(0,0,0,0))
    ax.add_collection3d(arena)
    # Draw nest rect.
    if nest:
        nest_coord = [[[0, 0, 0], [0, 22.65, 0], [16.23, 22.65, 0], [16.23, 0, 0]]]
        nest = Poly3DCollection(nest_coord, edgecolors='none', linewidths=0, facecolors='green', alpha=0.2)
        ax.add_collection3d(nest)
    # Draw looming disk.
    if looming:
        looming_radius = 8.2
        looming_center = [25.0, 50-14.25, 0]
        num_points = 100
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = looming_center[0] + looming_radius * np.cos(theta)
        y = looming_center[1] + looming_radius * np.sin(theta)
        z = np.zeros_like(x)
        faces = [[(x[i], y[i], z[i]) for i in range(num_points)]]
        looming = Poly3DCollection(faces, edgecolors='none', facecolors='red', alpha=0.2)
        ax.add_collection3d(looming)
    return ax

def visualize_3d(
    coord_array: np.ndarray,
    frame_range: tuple,
    joint_used: list,
    skel_conns,
    c_maps,
    save_path: str,
    viz_fps=20,
    video_path=None,
    bhvr_seq=None,
    mode='global',
    format=None,
    writer='opencv'
):
    """
    Visualize 3D skeleton with raw recording video and save output.
    Supports: mp4 / avi / gif / png / jpg

    Args:
        writer: 'opencv' or 'moviepy'
            - 'opencv': Stream write with cv2.VideoWriter (memory efficient, fast)
            - 'moviepy': Collect all frames then write (better quality, needs more memory)
    
    
    """
    start_frame, end_frame, step = frame_range
    colors = _assign_colors(c_maps, joint_used)

    # ---------- format ----------
    if format is not None:
        fmt = format.lower().lstrip('.')
    else:
        fmt = os.path.splitext(save_path)[1].lower().lstrip('.')
    if fmt == 'jpeg':
        fmt = 'jpg'

    # ---------- video reader ----------
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frame)

    # ---------- egocentric alignment ----------
    if mode == 'egocentric':
        anterior_idxs = [joint_used.index(j) for j in ['EarL', 'EarR', 'Snout']]
        posterior_idxs = [joint_used.index(j) for j in ['SpineM', 'TailB']]
        coord_array, _, _ = align_egocentric(coord_array, anterior_idxs, posterior_idxs)

    # ---------- setup output ----------
    frames = []  # Only for GIF format or moviepy backend
    video_writer = None  # For opencv streaming (cv2.VideoWriter object)
    
    if fmt in ['png', 'jpg']:
        os.makedirs(os.path.splitext(save_path)[0], exist_ok=True)
    elif fmt in ['mp4', 'avi'] and writer == 'opencv':
        # Will initialize opencv writer after first frame to get dimensions
        pass
    elif fmt not in ['png', 'jpg', 'gif', 'mp4', 'avi']:
        raise ValueError(f"Unsupported format: {fmt}")

    print(f"Start rendering & saving...)")

    for i, frame_idx in enumerate(range(start_frame, end_frame, step)):

        # ---- read video frame ----
        frame_rgb = None
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- matplotlib render ----
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvas(fig)

        ax = _draw_skeleton(ax, coord_array, frame_idx,
                            joint_used, skel_conns, colors)

        if mode == 'global':
            ax = _draw_arena(ax)
            ax.set_xlim(50, 0)
            ax.set_ylim(0, 50)
            ax.set_zlim(0, 25)
            ax.set_box_aspect([1, 1, 0.5])
        elif mode == 'egocentric':
            ax.set_xlim(5, -5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-2, 8)
            ax.set_box_aspect([1, 1, 1])

        ax.view_init(elev=20, azim=-135)  # For camera 1.
        # ax.view_init(elev=-1, azim=180)  # For test.
        ax.set_title(f'Frame {frame_idx}')

        if bhvr_seq is not None:
            bhv = bhvr_seq[frame_idx] if frame_idx < len(bhvr_seq) else "Unknown"
            ax.text2D(0.75, 0.9, f"{bhv}", transform=ax.transAxes,
                      fontsize=10, color='red')

        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # ---- concat raw video ----
        if frame_rgb is not None:
            frame_rgb = cv2.resize(frame_rgb, (img.shape[1], img.shape[0]))
            img = np.hstack((frame_rgb, img))

        # ---- initialize video writer on first frame ----
        if video_writer is None and fmt in ['mp4', 'avi'] and writer == 'opencv':
            height, width = img.shape[:2]
            if fmt == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:  # avi
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(save_path, fourcc, viz_fps, (width, height))

        # ---- save ----
        if fmt == 'gif':
            frames.append(img)
        elif fmt in ['mp4', 'avi']:
            if writer == 'opencv':
                # Stream write directly (BGR format for cv2)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)
            elif writer == 'moviepy':  # moviepy
                # Collect frames for moviepy
                frames.append(img)
        elif fmt in ['png', 'jpg']:
            out_dir = os.path.splitext(save_path)[0]
            cv2.imwrite(
                os.path.join(out_dir, f"{i:04d}.{fmt}"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

    # ---------- finalize ----------
    if cap:
        cap.release()
    
    if video_writer is not None:
        video_writer.release()
        print(f"Saved to {save_path}")
    elif fmt == 'gif':
        # Use moviepy for GIF (requires all frames)
        print("Creating GIF...")
        clip = ImageSequenceClip(frames, fps=viz_fps)
        clip.write_gif(save_path, logger=None)
        print(f"Saved to {save_path}")
    elif fmt in ['mp4', 'avi'] and writer == 'moviepy':
        # Use moviepy for mp4/avi if requested
        print("Encoding video with moviepy...")
        clip = ImageSequenceClip(frames, fps=viz_fps)
        clip.write_videofile(save_path, codec='libx264', audio=False, logger=None)
        print(f"Saved to {save_path}")
    elif fmt in ['png', 'jpg']:
        print(f"Saved to {save_path}")

