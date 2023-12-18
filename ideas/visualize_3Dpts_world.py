import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytorch3d.transforms as transformations
import os
import cv2 
import matplotlib.pyplot as plt


SHOW_PLOT = False
def visualize_camera_poses(poses, plt_title="Camera Poses", arrow_length=0.1):
    """
    Input:
        poses: a list of poses, each pose is a 4x4 numpy array 
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pose in poses:
        # Extract camera position and rotatio
        translation = pose[:, 3][:3]
        rotation_matrix = pose[:3, :3]

        # convert rotation matrix to tensor
        rotation_matrix = torch.from_numpy(rotation_matrix).float()

        # Convert rotation matrix to Euler angles
        euler_angles = transformations.matrix_to_euler_angles(rotation_matrix, "XYZ")

        # Plot camera position
        ax.scatter(translation[0], translation[1], translation[2], c='b', marker='o')

        # Plot camera orientation as a line
        ax.quiver(translation[0], translation[1], translation[2],
                  np.cos(euler_angles[2]), np.sin(euler_angles[2]), 0,
                  length=arrow_length, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(plt_title)
    plt.savefig(f"logs/{plt_title.replace(' ', '_')}.png", dpi=600)
    if SHOW_PLOT:
        plt.show()


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



sample_json_path = os.path.join('test_data', 'my_purple_apple_1107_visual', 'transforms.json')
T_marker2world = extract_camera_poses_from_json(sample_json_path)
print(f"extract {len(T_marker2world)} poses from {sample_json_path} {T_marker2world.shape}")
visualize_camera_poses(T_marker2world, plt_title="11.07 Optitrack Marker Poses", arrow_length=0.05)

camera_matrix = np.load(os.path.join('test_data', 'calib_result', 'camera_matrix.npy'))
print(f"check camera_matrix:\n {camera_matrix.shape} \n {camera_matrix}")

T_marker2camera = np.load(os.path.join('test_data', 'calib_result', 'marker2camera_transform.npy'))
print(f"check T_marker2camera:\n {T_marker2camera.shape} \n {T_marker2camera}")

# Obtain camera poses in the world frame
T_camera2world = np.linalg.inv(T_marker2camera) @ T_marker2world
print(f"check T_camera2world:\n {T_camera2world.shape}")
visualize_camera_poses(T_marker2world, plt_title="11.07 Camera Poses", arrow_length=0.05)


# # For each masked image, retrieve its T_marker2world and convert the masked pixels to world coordinates
# masked_image_dir = os.path.join('test_data', 'my_purple_apple_1107_visual', 'masked_images')
# objectInworld_list = []
# counter = 0 
# for masked_img_path in sorted(os.listdir(masked_image_dir), key=lambda x: int(x.split('.')[0])):
#     # Load a masked image and obtain the pixel coordinates of the masked pixels
#     sample_masked_img = cv2.imread(os.path.join(masked_image_dir, masked_img_path)) # (480, 848, 3)
#     object_pixels = np.where(np.all(sample_masked_img != [0, 0, 0], axis=-1)) # masked pixels are black [0, 0, 0]
#     # convert object_pixels to homogeneous coordinates
#     object_pixels = np.vstack((object_pixels, np.ones(object_pixels[0].shape))) # [3, N]


#     # Convert the pixel coordinates to world coordinates
#     objectIncamera = np.linalg.inv(camera_matrix) @ object_pixels
#     # convert objectIncamera to homogeneous coordinates
#     objectIncamera = np.vstack((objectIncamera, np.ones(objectIncamera[0].shape))) # [4, N]
#     img_idx = int(masked_img_path.split('.')[0])
#     sample_T_marker2world = T_marker2world[img_idx]
#     objectInworld = sample_T_marker2world @ np.linalg.inv(T_marker2camera) @ objectIncamera # [4, N]

#     objectInworld = objectInworld[:3, :] / objectInworld[3, :] # convert to 3D coordinates [3, N]
#     objectInworld_list.append(objectInworld)

#     counter += 1
#     if counter > 10:
#         break


# # Plot the 3D points, for each set of points in the list, plot the points with one color
# # Generate a list of random colors
# colormap = plt.get_cmap("viridis")
# num_colors = len(objectInworld_list)
# colors = [colormap(i / num_colors) for i in range(num_colors)]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for objectInworld, color in zip(objectInworld_list, colors):
#     ax.scatter(objectInworld[0, :], objectInworld[1, :], objectInworld[2, :], color=color, marker='o')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# set_axes_equal(ax)
# ax.set_title("3D points in world frame")
# plt.savefig("logs/20231107_3Dpts_in_world.png", dpi=200)
# if SHOW_PLOT:
#     plt.show()
