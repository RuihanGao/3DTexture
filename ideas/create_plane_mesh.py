import pytorch3d
import torch
from pytorch3d.io import save_obj

# Define device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

color = [0.7, 0.7, 1]
shape = "plane"
verbose = False

# Define the vertices, faces, and textures.
plane_depth = 0.1
plane_width_x = 1
plane_width_y = 1
plane_origin_x = 0
plane_origin_y = 0

vertices = torch.FloatTensor(
    [[plane_origin_x, plane_origin_y, 0], [plane_origin_x+plane_width_x, plane_origin_y, 0], [plane_origin_x+plane_width_x, plane_origin_y+plane_width_y, 0], [plane_origin_x, plane_origin_y+plane_width_y, 0], [plane_origin_x, plane_origin_y, plane_depth], [plane_origin_x+plane_width_x, plane_origin_y, plane_depth], [plane_origin_x+plane_width_x, plane_origin_y+plane_width_y, plane_depth], [plane_origin_x, plane_origin_y+plane_width_y, plane_depth]]
)
faces = torch.FloatTensor(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 4, 5],
        [1, 2, 5],
        [2, 5, 6],
        [2, 3, 6],
        [3, 6, 7],
        [3, 0, 7],
        [0, 7, 4],
        [4, 5, 6],
        [4, 6, 7],
    ]
)

vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
textures = torch.ones_like(vertices)  # (1, N_v, 3)
textures = textures * torch.tensor(color)  # (1, N_v, 3)

mesh = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=pytorch3d.renderer.TexturesVertex(textures),
)

mesh = mesh.to(device)

if verbose: print(f"check shape: vertices {vertices.shape}, faces {faces.shape}, textures {textures.shape}")
obj_path = f"{shape}_mesh.obj"
save_obj(obj_path, vertices[0], faces[0])
