import numpy as np
from scipy.spatial.transform import Rotation as R

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