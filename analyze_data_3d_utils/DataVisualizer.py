import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import os


def plot_lines(series_dict, draw_frames, title_id, xticks=None, save_dir=None, fps=20, lsf_list=None):
    """
    Plot multiple line series
    
    Args:
        series_dict: dict of {name: (data, color)}
        draw_frames: (start, end) frame range
        title_id: plot title
        xticks: custom xticks (in seconds if looming_starts provided, otherwise in frames)
        save_dir: directory to save plot
        fps: frames per second for time conversion (default: 20)
        looming_starts: list of looming start frames (optional, first one sets time 0, draws vertical lines for all)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    start, end = draw_frames
    
    # If looming_starts provided, use first looming as time 0, otherwise use frames
    if lsf_list and len(lsf_list) > 0:
        time_zero = lsf_list[0]
        x = (np.arange(start, end) - time_zero) / fps  # Time in seconds
        xlabel = 'Time (s)'
    else:
        time_zero = 0
        x = np.arange(start, end)  # Frames
        xlabel = 'Frame'
    
    for label, (data, color) in series_dict.items():
        ax.plot(x, data[start:end], label=label, color=color, linewidth=2)
    
    # Draw vertical lines for each looming
    if lsf_list:
        for i, lsf in enumerate(lsf_list):
            loom_x = (lsf - time_zero) / fps if time_zero else lsf
            if x[0] <= loom_x <= x[-1]:  # Only draw if in visible range
                if i == 0:
                    ax.axvline(loom_x, color='black', linestyle='--', linewidth=1, label='Looming')
                else:
                    ax.axvline(loom_x, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Value')
    ax.set_title(title_id)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{title_id}.png")
        fig.savefig(filepath, dpi=150)
        print(f"Saved: {filepath}")
    
    # Close the figure to free memory, especially important in loops
    plt.close(fig)


def plot_ethogram(bhvr_etho_dict, nest_etho_dict, bhvr_params=None,
                  draw_frames=None, lsf_list=None, title='ethogram', save_dir=None, fps=20):
    """
    Plot ethogram (behavior timeline)
    
    Args:
        bhvr_etho_dict: dict of behavior tuples
        nest_etho_dict: dict of nest presence tuples
        bhvr_params: dict of {behavior: (color, height)}, optional. If None, colors are auto-assigned
        draw_frames: (start, end) frame range
        lsf_list: list of looming start frames (draws vertical lines for each)
        title: plot title
        save_dir: directory to save plot
        fps: frames per second for time conversion (default: 20)
    """
    if draw_frames is None:
        # Auto-determine frame range from all behavior tuples
        all_starts = []
        all_ends = []
        for trial_id, bhvr_tuples in bhvr_etho_dict.items():
            for bhvr_name, tuples in bhvr_tuples.items():
                for t_start, t_end in tuples:
                    all_starts.append(t_start)
                    all_ends.append(t_end)
        
        if all_starts and all_ends:
            draw_frames = (min(all_starts), max(all_ends))
        else:
            draw_frames = (0, 2000)  # Fallback if no data
    
    start, end = draw_frames
    
    # Convert frames to seconds
    start_sec = start / fps
    end_sec = end / fps
    
    # Auto-generate bhvr_params if not provided
    if bhvr_params is None:
        # Collect all unique behavior names
        all_behaviors = set()
        for trial_id, bhvr_tuples in bhvr_etho_dict.items():
            all_behaviors.update(bhvr_tuples.keys())
        
        # Use matplotlib's tab20 colormap for diverse colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_behaviors)))
        bhvr_params = {bhvr: (colors[i], 1) for i, bhvr in enumerate(sorted(all_behaviors))}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    trial_idx = 0
    row_spacing = 0.1  # Space between rows
    row_height = 0.8   # Height of each behavior bar (1 - 2*row_spacing)
    nest_drawn = False  # Track if we've added nest to legend
    
    for trial_id, bhvr_tuples in bhvr_etho_dict.items():
        # Draw behavior rectangles
        for bhvr_name, tuples in bhvr_tuples.items():
            if bhvr_name not in bhvr_params:
                continue
            
            color, height = bhvr_params[bhvr_name]
            
            for t_start, t_end in tuples:
                if t_end > start and t_start < end:
                    t_start = max(t_start, start)
                    t_end = min(t_end, end)
                    
                    # Convert to seconds
                    t_start_sec = t_start / fps
                    t_end_sec = t_end / fps
                    
                    rect = Rectangle((t_start_sec, trial_idx + row_spacing), t_end_sec - t_start_sec, row_height,
                                   facecolor=color, edgecolor='none')
                    ax.add_patch(rect)
        
        # Draw nest presence as horizontal lines at the bottom of each row
        if nest_etho_dict and trial_id in nest_etho_dict:
            nest_data = nest_etho_dict[trial_id]
            row_center = trial_idx + 0.5  # Center of the row
            
            # nest_data is a dict like {'nest': [(start, end), ...]}
            nest_tuples = nest_data.get('nest', [])
            
            for t_start, t_end in nest_tuples:
                if t_end > start and t_start < end:
                    t_start = max(t_start, start)
                    t_end = min(t_end, end)
                    
                    # Convert to seconds
                    t_start_sec = t_start / fps
                    t_end_sec = t_end / fps
                    
                    # Draw thick horizontal line at bottom of row
                    line_label = 'Nest' if not nest_drawn else None
                    ax.plot([t_start_sec, t_end_sec], [row_center, row_center], 
                           color='gray', linewidth=2, solid_capstyle='butt', 
                           label=line_label, alpha=0.7)
                    nest_drawn = True
        
        trial_idx += 1
    
    # Draw vertical lines for each looming start frame
    if lsf_list:
        for i, lsf in enumerate(lsf_list):
            if start <= lsf <= end:  # Only draw if in visible range
                lsf_sec = lsf / fps
                if i == 0:
                    ax.axvline(lsf_sec, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Looming')
                else:
                    ax.axvline(lsf_sec, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(0, trial_idx)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.invert_yaxis()
    
    # Create legend
    legend_elements = [Patch(facecolor=color, edgecolor='none', label=bhvr_name) 
                      for bhvr_name, (color, _) in bhvr_params.items()]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             borderaxespad=0, frameon=True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{title}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    # Close the figure to free memory, especially important in loops
    plt.close(fig)