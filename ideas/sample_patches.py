import os
import sys

import frnn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pytorch3d
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch3d import _C
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from RayTracer import RayTracer
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from tqdm import tqdm
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from tools.map import MeshProjector
from tools.encoding import get_encoder
from tools.shape_tools import write_ply



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
verbose = False

def subdivide_mesh(mesh_path, min_vnum=5e5, obj_suffix="fine", replace=False):
    obj_path = ".".join(mesh_path.split(".")[:-1]) + "_" + obj_suffix + ".obj"
    if not os.path.exists(obj_path) or replace:
        mesh = trimesh.load_mesh(mesh_path)
        while mesh.vertices.shape[0] < min_vnum:
            v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
            mesh = trimesh.Trimesh(vertices=v, faces=f)
        mesh.export(obj_path)
    return obj_path


def sample_patches(
    mesh_path=None,
    patch_size=128,
    max_patch_num=2000,
    scan_pcl_path=None,
    sample_on_template=False,
    sample_on_picked_faces=True,
    picked_faces_path=None,
    use_trimesh_raycast=False,
    sample_poisson_disk=True,
    record_rays=False,
    work_space="./test_data/",
    cast_on_mfs=True,
):
    """
    Modified from the function sample_patches in map.py
    Isolate the sampling part from the class to test out sampling process for GelSight depth images.
    """

    # set up parameters that are defined in meshfield class
    pattern_rate = 1 / 50.0
    normal_net = None
    h_threshold = 20
    bound = 1 # extreme value of the mesh coordinate. e.g. bound = 1, then mesh coordinates should be [-1, 1]
    K = 8
    min_h = 1.0 # 1e-1
    max_depth = 10.0
    # TODO: encoder takes input [0,1], need to check whether we should normalize the mesh input
    # hash
    grid_name = "hashgrid"
    num_level = 8
    encoder, encoder_f_out_dim = get_encoder(
        grid_name,
        desired_resolution=1024,
        input_dim=3,
        num_levels=num_level,
        level_dim=2,
        base_resolution=512,
        log2_hashmap_size=19,
        align_corners=True,
    )
    encoder = encoder.to(device) # RH: added it so that embeddings in GridEncoder is GPU tensor. but didn't find corresponding implementation in original code

    current_level = 0
    meshprojector = MeshProjector(device=device, mesh_path=mesh_path, store_f=True, store_uv=True)
    fea_mesh_path = subdivide_mesh(mesh_path=mesh_path, min_vnum=128**2, obj_suffix=f"fea_level{current_level}")
    meshprojector_fea = MeshProjector(device=device, mesh_path=fea_mesh_path, store_f=True)

    # Build kd tree of scan point cloud for nn query
    scan_pcl = trimesh.load_mesh(scan_pcl_path).vertices if scan_pcl_path is not None else None
    scan_tree = KDTree(scan_pcl, leaf_size=2)

    # Load template mesh and determine grid gap
    mesh = trimesh.load_mesh(mesh_path)
    # Sample on picked faces or the whole mesh
    if sample_on_picked_faces and picked_faces_path is not None and os.path.exists(picked_faces_path):
        print("Sampling on ", picked_faces_path)
        mesh_for_sample = trimesh.load(picked_faces_path)
    else:
        print("Sampling on the whole mesh")
        mesh_for_sample = mesh
    edges = mesh_for_sample.vertices[mesh_for_sample.edges_unique]
    edges = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=-1)
    mean_edge_length = edges.mean()
    grid_gap = mean_edge_length * pattern_rate
    patch_edge_length = patch_size * grid_gap
    print(
        f"In function sample_patches, mean_edge_length: {mean_edge_length}, pattern_rate {pattern_rate}, patch_size: {patch_size}, patch_edge_length: {patch_edge_length}"
    )

    # Sampled patch align with the first component
    pca = PCA(n_components=3)
    if os.path.exists(work_space + "/meshes/direction.obj"):
        print("Use direction.obj for PCA")
        vertices_dir = trimesh.load_mesh(work_space + "/meshes/direction.obj").vertices
    else:
        print(f"Use {mesh_path} for PCA")
        vertices_dir = mesh.vertices
    pca.fit(vertices_dir)
    first_component = pca.components_[0]  # np.array([1., 0., 0.])
    print("First component: ", first_component)

    # Initialize patch coordinates
    calibration = np.linspace(-patch_size * grid_gap / 2, patch_size * grid_gap / 2, patch_size)
    x, y = np.meshgrid(calibration, calibration, indexing="ij")
    patch_coor = np.stack([x, y], axis=-1).reshape([-1, 2])
    patch_coor = np.concatenate([patch_coor, np.zeros_like(patch_coor)], axis=-1)
    patch_coor[..., -1] = 1
    if not os.path.exists(work_space + "/meshes/"):
        os.makedirs(work_space + "/meshes/")
    write_ply(patch_coor, work_space + "/meshes/sample_patch.ply")
    if not use_trimesh_raycast:
        patch_coor = torch.from_numpy(patch_coor).float().to(device)
        # cast_on_mfs determines whether cast on mesh_for_sample or not
        raytracer = (
            RayTracer(mesh_for_sample.vertices, mesh_for_sample.faces)
            if cast_on_mfs
            else RayTracer(mesh.vertices, mesh.faces)
        )

    # Estimate template normals for projection
    mesh_for_sample.as_open3d.compute_vertex_normals()
    if sample_on_template:
        v_normals = np.asarray(mesh_for_sample.as_open3d.vertex_normals)
        vertices = np.asarray(mesh_for_sample.vertices)
    elif sample_poisson_disk:
        vertices = np.asarray(mesh_for_sample.as_open3d.sample_points_poisson_disk(max_patch_num).points)
        _, _, face_idx = trimesh.proximity.closest_point(mesh_for_sample, vertices)
        v_normals = mesh_for_sample.face_normals[face_idx]
    else:
        vertices, face_idx = trimesh.sample.sample_surface_even(mesh_for_sample, 20000, radius=patch_edge_length / 16)
        vertices = vertices[: min(vertices.shape[0], max_patch_num)]
        face_idx = face_idx[: min(face_idx.shape[0], max_patch_num)]
        v_normals = mesh_for_sample.face_normals[face_idx]

    # Sample patches along the surface of template mesh
    print("Getting patches from curved surface ...")
    patches = np.zeros([max_patch_num, patch_size, patch_size, encoder_f_out_dim])
    patch_coors = np.zeros([max_patch_num, patch_size, patch_size, 3])
    patch_norms = np.zeros([max_patch_num, 3])
    patch_sample_tbn = np.zeros([max_patch_num, 9])
    picked_vertices = np.zeros([max_patch_num, 3])
    patch_phi_embed = (
        np.zeros([max_patch_num, patch_size, patch_size, normal_net.encoder_out_dim])
        if normal_net is not None
        else None
    )
    patch_local_tbn = np.zeros([max_patch_num, patch_size, patch_size, 9])
    patch_rays = np.zeros([max_patch_num, patch_size, patch_size, 6])
    count = 0

    # pdb.set_trace()

    # Sample one patch for each selected vertices
    for i in tqdm(range(vertices.shape[0])):
        print(f"*** sample patch {i}, vertices {vertices[i]} ***")

        # Discard parts below y=0 when no scan_pcl
        if scan_pcl is None and vertices[i, 1] < 0:
            continue

        # Determine the transform matrix by sample vertex
        z_axis = v_normals[i]
        y_axis = np.cross(z_axis, first_component)
        if verbose: print(f"check axis z_axis {z_axis}, y_axis {y_axis}, first_component {first_component}")
        if y_axis.sum() == 0:
            y_axis = np.cross(z_axis, np.array([1.0, 1.0, 1.01]) * first_component)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        T = np.eye(4)
        T[:3, :3] = np.stack([x_axis, y_axis, z_axis], -1)
        T[:3, 3] = vertices[i]

        if verbose: print(f"transformation matrix T \n {T}")

        # Find the intersections of the square scan rays and the coarse mesh
        if use_trimesh_raycast:
            # Initialize intersections and mask of intersection
            mask = np.zeros((patch_coor.shape[0]), dtype=bool)
            intersections = np.zeros((patch_coor.shape[0], 3), dtype=np.float32)
            # Transform the patch coordinates to ray origins
            ray_origins = np.einsum("ab,nb->na", T, patch_coor)[..., :3]
            # Discard patches far from the template surface
            nn_dist, _ = scan_tree.query(ray_origins, k=1)
            if nn_dist.max() > 3 * h_threshold:
                continue
            # Move ray origins away from template along the z_axis
            ray_origins += 1e3 * z_axis
            # Ray casting
            ray_directions = np.broadcast_to(-z_axis[None], ray_origins.shape)
            # cast_on_mfs determines whether cast on mesh_for_sample or not
            mesh_for_cast = mesh_for_sample if cast_on_mfs else mesh
            locations, index_ray, _ = mesh_for_cast.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )
            for ray_idx in np.unique(index_ray):
                idx = np.where(index_ray == ray_idx)[0]
                location = locations[idx]
                zs = ((ray_origins[ray_idx] - location) ** 2).sum(axis=-1)
                zmin_idx = np.argmin(zs)
                location = location[zmin_idx]
                intersections[ray_idx] = location
                mask[ray_idx] = True
            intersections = torch.from_numpy(intersections).to(device)
            # Discard those patches with un-intersected pixels
            if not mask.all():
                continue
        else:
            if verbose: print(f"use nn_query for raytracer")
            T = torch.from_numpy(T).float().to(device)
            ray_origins = torch.einsum("ab,nb->na", T, patch_coor)[..., :3]
            if ray_origins.isnan().any():
                print(f"Continue! find nan in ray_origins {ray_origins}")
                continue
            # Discard patches far from the template surface
            nn_dist, _ = scan_tree.query(ray_origins.cpu().numpy(), k=1)
            if nn_dist.max() > min(min_h, 3 * h_threshold):
                print(f"Continue! nn_dist.max() {nn_dist.max()} > min(min_h, 3 * h_threshold) {min(min_h, 3 * h_threshold)}")
                continue
            ray_origins += 0.1 * torch.from_numpy(z_axis).float().to(device)
            ray_directions = (
                torch.from_numpy(np.copy(np.broadcast_to(-z_axis[None], ray_origins.shape))).float().to(device)
            )
            if verbose: print(f"check shape: ray_origins {ray_origins.shape}, ray_directions {ray_directions.shape},    nn_dist {nn_dist.shape}, min {nn_dist.min()}, max {nn_dist.max()}")
            # check shape: ray_origins torch.Size([16384, 3]), ray_directions torch.Size([16384, 3]), nn_dist (16384, 1), min 0.09871222017896335, max 0.10286183578196903

            # intersections are the intersection of the scan rays (patch_size x patch_size) with the coarse mesh
            intersections, _, depth, _ = raytracer.trace(ray_origins, ray_directions)
            if verbose: print(f"check shape intersections {intersections.shape}, depth {depth.shape}, min {depth.min()}, max {depth.max()}")
            # check shape intersections torch.Size([16384, 3]), depth torch.Size([16384]), min 0.10000213980674744, max 0.10000213980674744
            if depth.max().item() > max_depth:
                # if the point from the scan grid is too far from the mesh, discard it
                print(f"Continue: depth.max().item() {depth.max().item()} > max_depth {max_depth}")
                continue

        # Gater patches
        p_sur, _, _, normal, local_tbn = meshprojector.project(
            intersections, K=K, h_threshold=h_threshold, requires_grad_xyz=False
        )
        if verbose: print(f"check p_sur shape {p_sur.shape}, p_sur[0] {p_sur[0]}")
        # check p_sur shape torch.Size([16384, 3]), p_sur[0] tensor([0.9532, 0.5477, 0.0100], device='cuda:0')
        if hash:
            # Embed surface coordinates with hash grid
            patch = encoder(p_sur, bound=bound)
            if verbose: print(f"check range of {count} patch: min {patch.min()}, max {patch.max()}, shape {patch.shape}")
            # check range of 1 patch: min -8.994223026093096e-05, max 9.519934246782213e-05, shape torch.Size([16384, 16])
        else:
            # # Embed surface points with barycentric features weighting
            # vertex_idx, barycentric, _, _ = meshprojector_fea.barycentric_mapping(
            #     intersections, normal, h_threshold=h_threshold, requires_grad_xyz=False
            # )
            # patch = (features[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
            raise NotImplementedError("Not implemented yet")
        patches[count] = patch.detach().cpu().numpy().reshape([patch_size, patch_size, -1])
        if verbose: print(f"check patch count {count} shape {patches[count].shape}")
        patch_local_tbn[count] = local_tbn.cpu().numpy().reshape([patch_size, patch_size, 9])
        patch_coors[count] = intersections.cpu().numpy().reshape([patch_size, patch_size, 3])
        patch_norms[count] = z_axis
        if patch_phi_embed is not None:
            phi_embed = normal_net.phi_embedding(p_sur)
            patch_phi_embed[count] = phi_embed.detach().cpu().numpy().reshape([patch_size, patch_size, -1])
        if not type(T) == np.ndarray:
            T = T.cpu().numpy()
        patch_sample_tbn[count] = T[:3, :3].T.reshape([9])
        picked_vertices[count] = vertices[i]
        if record_rays:
            if not type(ray_origins) == np.ndarray:
                ray_origins = ray_origins.cpu().numpy()
                ray_directions = ray_directions.cpu().numpy()
            rays = np.concatenate([ray_origins, ray_directions], axis=-1)
            patch_rays[count] = rays.reshape([patch_size, patch_size, 6])
        count += 1
        if max_patch_num is not None and count == max_patch_num:
            break

    # Stack and return
    patches = patches[:count]
    patch_coors = patch_coors[:count]
    patch_norms = patch_norms[:count]
    patch_sample_tbn = patch_sample_tbn[:count]
    patch_local_tbn = patch_local_tbn[:count]
    picked_vertices = picked_vertices[:count]
    if patch_phi_embed is not None:
        patch_phi_embed = patch_phi_embed[:count]
    if record_rays:
        patch_rays = patch_rays[:count]
    print("Get patches: ", patches.shape, " Grid Gap: ", grid_gap)
    return (
        patches,
        grid_gap,
        patch_coors,
        patch_norms,
        patch_sample_tbn,
        patch_local_tbn,
        picked_vertices,
        patch_phi_embed,
        patch_rays,
    )


if __name__ == "__main__":

    # Step 1. Run sample_patches
    scan_pcl_path = "test_data/my_brown_box_plane_tesellation_normalize.ply"  # the combined point cloud of GelSight data
    mesh_path = "test_data/plane_mesh_subdiv_10.obj"
    patches, grid_gap, patch_coors, patch_norms, patch_sample_tbn, patch_local_tbn, picked_vertices, patch_phi_embed, patch_rays = sample_patches(mesh_path=mesh_path, scan_pcl_path=scan_pcl_path, record_rays=True, work_space="./test_data", max_patch_num=2000)
    print(f"check patches range before saving: {patches.min()}, {patches.max()}")
    # Get patches:  (1956, 128, 128, 16)  Grid Gap:  3.226698626424242e-05
    # check patches range before saving: -9.95844166027382e-05, 9.927829523803666e-05

    # Ref: function export_field() in network_curvefield.py
    field_dict = {'mesh': trimesh.load_mesh(mesh_path), 'patches': patches, 'grid_gap': grid_gap, 'patch_coors': patch_coors, 'patch_norms': patch_norms, 'patch_sample_tbn': patch_sample_tbn, 'patch_local_tbn': patch_local_tbn, 'picked_vertices': picked_vertices, 'patch_phi_embed': patch_phi_embed, 'patch_rays': patch_rays}
    # Ref: function save_field() in utils.py
    save_dir = "./test_data/field"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "curved_grid_hash.npz")
    np.savez(save_path, **field_dict)
    print('Save_field Done!')

    # # Step 2. Run patch_matching_and_quilting
    # # run patch_matching_and_quilting. Changed the path and done
    
    # # TODO: Step 3:  run load_synthesis
    # load_path = "test_data/field/texture.npz"
    # # Ref: function load_field in utils.py
    # field_dict = np.load(load_path, allow_pickle=True)
    # print(f"check field_dict: {field_dict.files}")
    # features = field_dict['features']
    # print(f"check field features: shape {features.shape}, dtype {features.dtype}, range {np.min(features)} - {np.max(features)}")
    # # self.model.import_field(field_dict, fp16=self.fp16)
    # # self.update_optimizer_scheduler()