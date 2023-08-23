import pytorch3d
import torch
from pytorch3d.io import save_obj
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
import warnings
warnings.filterwarnings("ignore")

# Define device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

color = [0.7, 0.7, 1]
shape = "plane"
verbose = True

# Define the vertices, faces, and textures.
plane_depth = 0.01
plane_width_x = 1 # 1280
plane_width_y = 1 # 1280
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

if verbose: print(f"level 0 shape: vertices {vertices.shape}, faces {faces.shape}, textures {textures.shape}")
# [1, 8, 3], [1, 12, 3]
obj_path = f"test_data/{shape}_mesh.obj"
save_obj(obj_path, vertices[0], faces[0])

# Try to subdivide the mesh
# Ref: https://stackoverflow.com/questions/72862446/how-to-define-pytorch3d-plane-geometry
level = 10
for i in range(1, level):
    subdiv = SubdivideMeshes()
    mesh = subdiv(mesh)

    verts_subdiv = mesh.verts_list()[0]
    faces_subdiv = mesh.faces_list()[0]
    if verbose: print(f"level {i} shape after subdiv: vertices {verts_subdiv.shape}, faces {faces_subdiv.shape}")
    
# save the subdivided mesh
obj_path = f"test_data/{shape}_mesh_subdiv_{level}.obj"
save_obj(obj_path, verts_subdiv, faces_subdiv)


# for grid 1280x1280, the number of grid intersections = 1638400.
# I set the level to 10, so that the number of vertices is 1572866, which is close to 1638400.

# level 0 shape: vertices torch.Size([1, 8, 3]), faces torch.Size([1, 12, 3]), textures torch.Size([1, 8, 3])
# level 1 shape after subdiv: vertices torch.Size([26, 3]), faces torch.Size([48, 3])
# level 2 shape after subdiv: vertices torch.Size([98, 3]), faces torch.Size([192, 3])
# level 3 shape after subdiv: vertices torch.Size([386, 3]), faces torch.Size([768, 3])
# level 4 shape after subdiv: vertices torch.Size([1538, 3]), faces torch.Size([3072, 3])
# level 5 shape after subdiv: vertices torch.Size([6146, 3]), faces torch.Size([12288, 3])
# level 6 shape after subdiv: vertices torch.Size([24578, 3]), faces torch.Size([49152, 3])
# level 7 shape after subdiv: vertices torch.Size([98306, 3]), faces torch.Size([196608, 3])
# level 8 shape after subdiv: vertices torch.Size([393218, 3]), faces torch.Size([786432, 3])
# level 9 shape after subdiv: vertices torch.Size([1572866, 3]), faces torch.Size([3145728, 3])