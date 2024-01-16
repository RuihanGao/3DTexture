import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from camera_pose_visualizer import CameraPoseVisualizer

def get_transform_matrix_from_quaternion(tvec, quat):
  """
  Calculate the 4x4 homogeneous transformation matrix.
  :param tvec: np.1darray (3,); translation vector.
  :param quat: np.1darray (4,): quaternion vector. (qx, qy, qz, qw)
  :return: np.2darray (4, 4); homogeneous transformation matrix.
  """
  T = np.eye(4)
  T[:3, :3] = R.from_quat(quat).as_matrix()
  T[:3, 3] = tvec
  return T


def get_quaternion_from_matrix(T):
  """
  Calculate the quaternion vector from the 4x4 homogeneous transformation matrix.
  :param T: np.2darray (4, 4); homogeneous transformation matrix.
  :return: np.1darray (4,); quaternion vector. Order: x, y, z, w.
  """
  quat = R.from_matrix(T[:3, :3]).as_quat()
  return quat


def camera_projection(points, mtx):
  """
  Get the pixel position of 3D points in the camera frame.
  :param points: np.2darray (N, 3); the points in camera frame.
  :param mtx: np.2darray (3, 3); the camera matrix.
  :return: np.2darray (N, 2); the pixel position of the points.
  """
  proj_points = (mtx @ points.T).T
  pxs = proj_points[:, :2] / proj_points[:, 2]
  return pxs


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

  

# Extract the camera poses from transforms.json file
def extract_camera_poses_from_json(json_path):
    """
    Input:
        json_path: path to the json file containing the camera poses
    Output:
        poses: a list of poses, each pose is a 4x4 numpy array 
    """
    import json
    with open(json_path) as f:
        data = json.load(f)

    # Note the camera poses are stored in the 'frames' field and each frame has file_path field to store the image name, e.g.   "file_path": "./images/00058.png",
    # Since json file is not sorted, we need to sort the frames by the file_path field
    sorted_frames = sorted(data['frames'], key=lambda k: k['file_path'].split('/')[-1])
    poses = []

    for i in range(len(sorted_frames)):
        pose = np.array(sorted_frames[i]['transform_matrix'])
        poses.append(pose)

    return np.array(poses)

  
def visualize_cam_poses_3D(T_camerainworld, plt_title="Visualize cameras in world frame", save_plot=False, show_plot=True, save_path=None, dpi=600, flip_transform=np.eye(4), focal_len_scaled=0.1, draw_idx=False, selected_idx=None):
    """
    
    """
    # Create 3D plot for visualization
    axis_margin = 0.01
    
    if len(T_camerainworld.shape) == 2:
        T_camerainworld = np.expand_dims(T_camerainworld, axis=0)

    T_camerainworld = T_camerainworld @ flip_transform
    print(f"Visualize {len(T_camerainworld)} camera poses in world frame")

    # obtain the min and max of all camera poses
    x_min = np.min(T_camerainworld[:, 0, 3]) - axis_margin
    x_max = np.max(T_camerainworld[:, 0, 3]) + axis_margin
    y_min = np.min(T_camerainworld[:, 1, 3]) - axis_margin
    y_max = np.max(T_camerainworld[:, 1, 3]) + axis_margin
    z_min = np.min(T_camerainworld[:, 2, 3]) - axis_margin
    z_max = np.max(T_camerainworld[:, 2, 3]) + axis_margin
    print(f"limit for all camera poses: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}")

    visualizer = CameraPoseVisualizer([x_min, x_max], [y_min, y_max], [z_min, z_max])
    set_axes_equal(visualizer.ax)

    # Generate a list of random colors
    colormap = plt.get_cmap("viridis")
    num_colors = len(T_camerainworld)
    colors = [colormap(i / num_colors) for i in range(num_colors)]
    # Add camera poses in the world
    for i, color in zip(range(len(T_camerainworld)), colors):
        if selected_idx is not None and i not in selected_idx:
            continue
        visualizer.extrinsic2pyramid(T_camerainworld[i], color, focal_len_scaled=focal_len_scaled, idx=i, draw_idx=draw_idx)
    visualizer.show(plt_title=plt_title, save_plot=save_plot, show_plot=show_plot, save_path=save_path, dpi=dpi)



def visualize_cam_gelsight_poses_3D(T_camerainworld=None, T_gelsightinworld=None, plt_title="Visualize cameras in world frame", save_plot=False, show_plot=True, save_path=None, dpi=600, flip_transform=np.eye(4), focal_len_scaled=0.1, draw_idx=False, selected_idx=None):
    """
    """
    # Create 3D plot for visualization
    axis_margin = 0.01
    
    if T_camerainworld is not None:
        if len(T_camerainworld.shape) == 2:
            T_camerainworld = np.expand_dims(T_camerainworld, axis=0)
        T_camerainworld = T_camerainworld @ flip_transform
        print(f"Visualize {len(T_camerainworld)} camera poses in world frame")
    
    if T_gelsightinworld is not None:
        if len(T_gelsightinworld.shape) == 2:
            T_gelsightinworld = np.expand_dims(T_gelsightinworld, axis=0)
        T_gelsightinworld = T_gelsightinworld @ flip_transform
        print(f"Visualize {len(T_gelsightinworld)} gelsight poses in world frame")
    
    # obtain the min and max of all camera poses
    # assume the camera poses occupy a larger space than the gelsight poses
    if T_camerainworld is not None:
        T_limit = T_camerainworld
    else:
        assert T_gelsightinworld is not None, "Please provide either T_camerainworld or T_gelsightinworld"
        T_limit = T_gelsightinworld

    x_min = np.min(T_limit[:, 0, 3]) - axis_margin
    x_max = np.max(T_limit[:, 0, 3]) + axis_margin
    y_min = np.min(T_limit[:, 1, 3]) - axis_margin
    y_max = np.max(T_limit[:, 1, 3]) + axis_margin
    z_min = np.min(T_limit[:, 2, 3]) - axis_margin
    z_max = np.max(T_limit[:, 2, 3]) + axis_margin
    print(f"limit for all sensor poses: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}")

    visualizer = CameraPoseVisualizer([x_min, x_max], [y_min, y_max], [z_min, z_max])
    set_axes_equal(visualizer.ax)

    if T_camerainworld is not None:
        # Add camera poses in the world
        cam_colormap = plt.get_cmap("autumn")
        num_colors = len(T_camerainworld)
        colors = [cam_colormap(i / num_colors) for i in range(num_colors)]
        # Add camera poses in the world
        for i, color in zip(range(len(T_camerainworld)), colors):
            if selected_idx is not None and i not in selected_idx:
                continue
            visualizer.extrinsic2pyramid(T_camerainworld[i], color, focal_len_scaled=focal_len_scaled, idx=i, draw_idx=draw_idx)
    
    if T_gelsightinworld is not None:
        # Add gelsight poses in the world
        gelsight_colormap = plt.get_cmap("winter")
        num_colors = len(T_gelsightinworld)
        colors = [gelsight_colormap(i / num_colors) for i in range(num_colors)]
        # Add camera poses in the world
        for i, color in zip(range(len(T_gelsightinworld)), colors):
            visualizer.extrinsic2pyramid(T_gelsightinworld[i], color, focal_len_scaled=focal_len_scaled/2, idx=i)
    
    visualizer.show(plt_title=plt_title, save_plot=save_plot, show_plot=show_plot, save_path=save_path, dpi=dpi)