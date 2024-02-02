import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
"""
Create a pair of coordinates with a known but random transform adn try to recover the transform
"""

def random_rotation_matrix():
    # Generate random Euler angles (in radians)
    roll = np.random.uniform(0, 2 * np.pi)
    pitch = np.random.uniform(0, 2 * np.pi)
    yaw = np.random.uniform(0, 2 * np.pi)

    # Create rotation matrix
    rotation_matrix = np.array(
        [
            [
                np.cos(yaw) * np.cos(pitch),
                -np.sin(yaw) * np.cos(roll)
                + np.cos(yaw) * np.sin(pitch) * np.sin(roll),
                np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(roll) + np.sin(yaw) * np.sin(pitch) * np.sin(roll),
                -np.cos(yaw) * np.sin(roll)
                + np.sin(yaw) * np.sin(pitch) * np.cos(roll),
            ],
            [
                -np.sin(pitch),
                np.cos(pitch) * np.sin(roll),
                np.cos(pitch) * np.cos(roll),
            ],
        ]
    )

    return rotation_matrix


def random_translation_vector():
    translation_vector = np.random.uniform(-1.0, 1.0, 3)  # Adjust the range as needed
    return translation_vector


def random_scaling_matrix(scale_factor):
    scaling_matrix = np.diag([scale_factor, scale_factor, scale_factor, 1.0])
    return scaling_matrix


def random_homogeneous_transform(scale_factor=1):
    rotation = random_rotation_matrix()
    translation = random_translation_vector()
    scaling = random_scaling_matrix(scale_factor)  # Optional

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation
    homogeneous_matrix[:3, 3] = translation
    homogeneous_matrix = np.dot(homogeneous_matrix, scaling)  # Optional

    return homogeneous_matrix


def transform_point(point, transformation_matrix):
    # Add a homogeneous coordinate (w = 1.0) to the point
    point_homogeneous = np.append(point, 1.0)

    # Apply the transformation matrix
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)

    # Remove the homogeneous coordinate (w) and return the transformed point
    return transformed_point_homogeneous[:3]



def estimate_transform(points_original, points_transformed, verbose=False):
    """
    Given the coordinates of a set of points in two coordinate frames, estimate the homogeneous transform between two frames.
    Input:
        points_original: a list of points in the original coordinate frame, shape (N, 3)
        points_transformed: a list of points in the transformed coordinate frame, shape (N, 3)
    """
    # Convert the points to numpy arrays
    points_original = np.array(points_original)
    points_transformed = np.array(points_transformed)
    if verbose:
        print(
            f"check shape points_original: {points_original.shape} points_transformed: {points_transformed.shape}"
        )
    # Compute the centroid
    centroid_original = np.mean(points_original, axis=0)
    centroid_transformed = np.mean(points_transformed, axis=0)
    
    if verbose: 
        print(f"centroid_original: {centroid_original.shape} {centered_original}")
    # Center the points
    centered_original = points_original - centroid_original
    centered_transformed = points_transformed - centroid_transformed
    # Compute the covariance matrix
    H = np.dot(centered_original.T, centered_transformed)
    # Use SVD to compute the rotation matrix
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # Handle the reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    # Compute the translation
    t = centroid_transformed.T - np.dot(R, centroid_original.T)
    # Assemble the transformation matrix
    Transform = np.eye(4)
    Transform[:3, :3] = R
    Transform[:3, 3] = t
    return Transform


def evaluate_transform(points_original, points_transformed, Transform, alpha=0.5, plt_title=None, save_plot=False, show_plot=False, save_path="evaulate_transform.png", dpi=600, draw_idx=False, start_idx=0, idx_list=None, rank_error=False, error_threshold=1.0):
    """
    Given the coordinates of a set of points in two coordinate frames, evaluate the quality of the estimated transform and plot the transformed and remapped points for visual inspection.
    """
    remapped_points = (
        np.dot(Transform[:3, :3], points_original.T).T + Transform[:3, 3]
    )  # Apply the estimated transform to the original points, shape [N, 3]
    if rank_error:
        errors = np.linalg.norm(points_transformed - remapped_points, axis=1) # shape [N,]
       
        print(f"Check errors shape {errors.shape}, min {np.min(errors)}, max {np.max(errors)}")
        # # plot histogram of errors
        # plt.hist(errors, bins=20)
        # plt.xlabel("Error")
        # plt.ylabel("Occurence")
        # today  = datetime.now().strftime("%Y%m%d")
        # plt.savefig(f"logs/{today}_error_hist.png", dpi=300)
        # print(f"Save the histogram of the remapping error to logs/{today}_error_hist.png")
        # plt.close()
        
        # Based on the histogram plot, set a threshold of 1.0
        # find the index of points where the error exceeds the threshold
        filtered_pts_index = np.where(errors < error_threshold)[0]
        filted_out_pts_index = np.where(errors >= error_threshold)[0]
        print(f"check shape filtered_pts_index {filtered_pts_index.shape}, filted_out_pts_index {filted_out_pts_index.shape}")
        print(f"Find {len(filtered_pts_index)} points have error smaller than threshold, filter out {len(filted_out_pts_index)} pts, index: {filted_out_pts_index} ")


    squared_error = np.sum(np.square(points_transformed - remapped_points), axis=1)
    mse = np.mean(np.square(points_transformed - remapped_points))
    print(f"In function evaluate_transform, \nMean Squared Error: {mse} \nTotal squared error {squared_error.sum()}")

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot original points in blue
    # ax.scatter(points_original[:, 0], points_original[:, 1], points_original[:, 2], c='blue', label='Original Points')

    # Plot transformed points in green
    ax.scatter(
        points_transformed[:, 0],
        points_transformed[:, 1],
        points_transformed[:, 2],
        c="green",
        marker="o",
        alpha=alpha,
        label="Transformed Points",
    )

    # Plot remapped points in red
    ax.scatter(
        remapped_points[:, 0],
        remapped_points[:, 1],
        remapped_points[:, 2],
        c="red",
        marker="s",
        alpha=alpha,
        label="Remapped Points",
    )

    if draw_idx:
        if idx_list is None:
            idx_list = list(range(remapped_points.shape[0]))
        for i in range(remapped_points.shape[0]):
            if i in idx_list:
                ax.text(remapped_points[i, 0], remapped_points[i, 1], remapped_points[i, 2], str(i+start_idx))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    if plt_title is not None:
        plt.title(plt_title)
    if save_plot:
        plt.savefig(save_path, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()
    

def test_homogeneous_transform():
    """
        Step 1: Create a random homogeneous transform and a list of random points. Compute the transformed points
        Step 2. Invert the problem. Given the points_original and points_transformed, compute the homogeneous transformation between them
    """
    ## Step 1.
    # create a random homogeneous transformation matrix
    homogeneous_matrix = random_homogeneous_transform()
    # create a list of 3D points in the original coordinate frame
    points_original = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
        np.array([1.0, 4.0, 2.0]),
        np.array([2.0, 7.0, 3.0]),
        np.array([5.0, 4.0, 8.0]),
        np.array([4.0, 7.0, 2.0]),
        np.array([10.0, 6.0, 5.0]),
    ]
    # apply the transformation to each point in the list
    points_transformed = [
        transform_point(point, homogeneous_matrix) for point in points_original
    ]

    for i, point in enumerate(points_transformed):
        print(f"Point {i + 1} (Transformed): {point}")
    points_original = np.array(points_original)
    points_transformed = np.array(points_transformed)

    ## Step 2.
    Transform = estimate_transform(points_original, points_transformed, verbose=False)
    evaluate_transform(points_original, points_transformed, Transform, save_plot=True, show_plot=False, save_path="logs/20231125_verify_homogeneous_transform.png", dpi=600)
    # MSE 7.6e-29

if __name__ == "__main__":
    test_homogeneous_transform()
