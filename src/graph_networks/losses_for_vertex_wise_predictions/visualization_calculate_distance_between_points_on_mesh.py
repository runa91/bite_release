
"""
code adapted from: https://github.com/mikedh/trimesh/blob/main/examples/shortest.py
shortest.py
----------------
Given a mesh and two vertex indices find the shortest path
between the two vertices while only traveling along edges
of the mesh.
"""

# python src/graph_networks/losses_for_vertex_wise_predictions/calculate_distance_between_points_on_mesh.py



import trimesh

import networkx as nx


ROOT_PATH_MESH = ......
ROOT_OUT_PATH = .....

path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'


import pdb; pdb.set_trace()

my_mesh = trimesh.load_mesh(path_mesh, process=False,  maintain_order=True)

# edges without duplication
edges = my_mesh.edges_unique

# the actual length of each unique edge
length = my_mesh.edges_unique_length

# create the graph with edge attributes for length (option A)
#   g = nx.Graph()
#   for edge, L in zip(edges, length): g.add_edge(*edge, length=L)
# you can create the graph with from_edgelist and
# a list comprehension (option B)
ga = nx.from_edgelist([(e[0], e[1], {'length': L}) for e, L in zip(edges, length)])

# arbitrary indices of mesh.vertices to test with
start = 0
end = int(len(my_mesh.vertices) / 2.0)

# run the shortest path query using length for edge weight
path = nx.shortest_path(ga, source=start, target=end, weight='length')

# VISUALIZE RESULT
# make the sphere transparent-ish
my_mesh.visual.face_colors = [100, 100, 100, 100]
# Path3D with the path between the points
path_visual = trimesh.load_path(my_mesh.vertices[path])
# visualizable two points
points_visual = trimesh.points.PointCloud(my_mesh.vertices[[start, end]])

# create a scene with the mesh, path, and points
my_scene = trimesh.Scene([points_visual, path_visual, my_mesh])

my_scene.export(ROOT_OUT_PATH + 'shortest_path.stl')


scene.show(smooth=False)





