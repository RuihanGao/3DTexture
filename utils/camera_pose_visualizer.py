import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim, figsize=(6, 6)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3, idx=0, draw_idx=False):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        if draw_idx:
            self.ax.text(vertex_transformed[0, 0], vertex_transformed[0, 1], vertex_transformed[0, 2], str(idx)) 
        

    def center2cube(self, center=[0, 0, 0],  side_length=0.1, color='r',):
        """
        Implemented by Ruihan. A quick modification to viusalize an object as a cube given pose and sidelength

        :param center: np.1darray (3,); the center of the cube.
        :param color: str; the color of the cube.
        :param side_length: float; the side length of the cube.
        """

        half_side = side_length / 2

        vertices = np.array([
            [center[0] - half_side, center[1] - half_side, center[2] - half_side],
            [center[0] + half_side, center[1] - half_side, center[2] - half_side],
            [center[0] + half_side, center[1] + half_side, center[2] - half_side],
            [center[0] - half_side, center[1] + half_side, center[2] - half_side],
            [center[0] - half_side, center[1] - half_side, center[2] + half_side],
            [center[0] + half_side, center[1] - half_side, center[2] + half_side],
            [center[0] + half_side, center[1] + half_side, center[2] + half_side],
            [center[0] - half_side, center[1] + half_side, center[2] + half_side]
        ])

        faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]]

        cube = Poly3DCollection(faces, linewidths=1, edgecolors=color, alpha=.25)
        self.ax.add_collection3d(cube)


    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self, plt_title="Extrinsic Parameters", save_plot=False, save_path=None, dpi=300, show_plot=True):
        plt.title(plt_title)
        if save_plot:
            assert save_path is not None, 'Please specify the path to save the plot'
            print(f"Save plot to {save_path}")
            plt.savefig(save_path, dpi=dpi)
        if show_plot:
            plt.show()
