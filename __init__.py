from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Quaternion, Matrix

from .pig import Node, load_pig
from ...common_api import *


def pig_init():
    pass


def plugin_init():
    pass


def _create_skeleton(model_name: str, nodes: list[Node]):
    arm_data = bpy.data.armatures.new(model_name + "_ARMDATA")
    arm_obj = bpy.data.objects.new(model_name + "_ARM", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    arm_obj.show_in_front = True
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for bone in nodes:
        bl_bone = arm_data.edit_bones.new(bone.name)
        bl_bone.tail = Vector([0, 0, 0.5 * bone.unk0]) + bl_bone.head
    for bone in nodes:
        x, y, z, w = bone.rotation
        matrix = Matrix.LocRotScale(Vector(bone.position), Quaternion((w, x, y, z)), Vector(bone.scale))
        if bone.parent_id != -1:
            bl_bone.parent = arm_data.edit_bones[nodes[bone.parent_id].name]

        if bl_bone.parent:
            bl_bone.matrix = bl_bone.parent.matrix @ matrix
        else:
            bl_bone.matrix = matrix
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def pig_load(operator, filepath: str, files: list[str]):
    collection = get_or_create_collection("Main", bpy.context.scene.collection)
    base_path = Path(filepath).parent
    for file in files:
        filepath = base_path / file
        with FileBuffer(filepath, "rb") as f:
            nodes, objects = load_pig(filepath.stem, f)
        _create_skeleton(filepath.stem, nodes)

        for object in objects:
            node = nodes[object.node_id]
            for lod in object.lods:
                for mesh in lod.meshes:
                    mesh_data = bpy.data.meshes.new(node.name + f"_MESH")
                    mesh_obj = bpy.data.objects.new(node.name, mesh_data)
                    mesh_data.from_pydata(mesh.vertices["POSITION"], [], mesh.faces)
                    mesh_data.update(calc_edges=True, calc_edges_loose=True)
                    # mesh_obj.location = mesh.pivot_point
                    if "NORMAL" in mesh.vertices.dtype.names:
                        add_custom_normals(mesh.vertices["NORMAL"], mesh_data)
                    if "UV1" in mesh.vertices.dtype.names:
                        add_uv_layer("UV1", mesh.vertices["UV1"], mesh_data, flip_uv=True)
                    if "UV2" in mesh.vertices.dtype.names:
                        add_uv_layer("UV2", mesh.vertices["UV2"], mesh_data, flip_uv=True)
                    if "VCOLOR" in mesh.vertices.dtype.names:
                        add_vertex_color_layer("VCOLOR", mesh.vertices["VCOLOR"].astype(np.float32) / 255, mesh_data)
                    collection.objects.link(mesh_obj)
    return {"FINISHED"}


plugin_info = {
    "name": "Gameloft engine",
    "id": "GameloftEngineLoader",
    "description": "Import Gameloft engine assets",
    "version": (0, 1, 1),
    "loaders": [
        {
            "name": "Load .pig file",
            "id": "gle_pig",
            "exts": ("*.pig",),
            "init_fn": pig_init,
            "import_fn": pig_load,
            "properties": [
            ]
        },
    ],
    "init_fn": plugin_init
}
