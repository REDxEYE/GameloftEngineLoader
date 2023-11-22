from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Quaternion, Matrix

from .pig import Node, load_pig
from ...common_api import *


def pig_init():
    pass


def pvr_init():
    pass


def plugin_init():
    pass


def pvr_load(operator, filepath: str, files: list[str]):
    base_path = Path(filepath).parent
    for file in files:
        texture = Texture.from_pvr(base_path / file)
        if not texture:
            operator.report({"ERROR"}, f"Failed to load {base_path / file}")
            continue
        if create_image_from_texture(file, texture, True) is None:
            operator.report({"ERROR"}, f"Failed to create texture from {base_path / file}")
    return {"FINISHED"}


def _create_skeleton(model_name: str, nodes: list[Node]):
    arm_data = bpy.data.armatures.new(model_name + "_ARMDATA")
    arm_obj = bpy.data.objects.new(model_name + "_ARM", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    arm_obj.show_in_front = True
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for n, bone in enumerate(nodes):
        if bone.type == 0 and n != 0:
            continue
        bl_bone = arm_data.edit_bones.new(bone.name)
        bl_bone.tail = Vector([0, 0, 0.5 * max(0.01, bone.unk0)]) + bl_bone.head
    for n, bone in enumerate(nodes):
        if bone.type == 0 and n != 0:
            continue
        bl_bone = arm_data.edit_bones[bone.name]
        if bone.parent_id != -1:
            bl_bone.parent = arm_data.edit_bones[nodes[bone.parent_id].name]
    bpy.ops.object.mode_set(mode='POSE')
    for n, bone in enumerate(nodes):
        if bone.type == 0 and n != 0:
            continue
        z, y, x, w = bone.rotation
        matrix = Matrix.LocRotScale(Vector(bone.position), Quaternion((w, x, y, z)), Vector(bone.scale))
        bl_bone = arm_obj.pose.bones[bone.name]

        if bl_bone.parent:
            bl_bone.matrix = bl_bone.parent.matrix @ matrix
        else:
            bl_bone.matrix = matrix
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def pig_load(operator, filepath: str, files: list[str]):
    collection = get_or_create_collection("Main", bpy.context.scene.collection)
    base_path = Path(filepath).parent
    for file in files:
        filepath = base_path / file
        with FileBuffer(filepath, "rb") as f:
            nodes, objects = load_pig(filepath.stem, f)
        skeleton_object = _create_skeleton(filepath.stem, nodes)

        for object in objects:
            node = nodes[object.node_id]
            for lod in object.lods:
                for mesh in lod.meshes:
                    mesh_data = bpy.data.meshes.new(node.name + f"_MESH")
                    mesh_obj = bpy.data.objects.new(node.name, mesh_data)
                    mesh_data.from_pydata(mesh.vertices["POSITION"], [], mesh.faces)
                    mesh_data.update(calc_edges=True, calc_edges_loose=True)
                    x, y, z, w = node.rotation
                    mesh_obj.matrix_local = Matrix.LocRotScale(Vector(node.position),
                                                               Quaternion((-w, x, y, z)),
                                                               Vector(node.scale))
                    if "NORMAL" in mesh.vertices.dtype.names:
                        add_custom_normals(mesh.vertices["NORMAL"], mesh_data)
                    if "UV1" in mesh.vertices.dtype.names:
                        add_uv_layer("UV1", mesh.vertices["UV1"], mesh_data, flip_uv=True)
                    if "UV2" in mesh.vertices.dtype.names:
                        add_uv_layer("UV2", mesh.vertices["UV2"], mesh_data, flip_uv=True)
                    if "VCOLOR" in mesh.vertices.dtype.names:
                        add_vertex_color_layer("VCOLOR", mesh.vertices["VCOLOR"].astype(np.float32) / 255, mesh_data)

                    if lod.have_skeleton:
                        remap = mesh.unk_buffer
                        remap_table = np.zeros((len(remap)), np.uint8)
                        for n, r in enumerate(remap):
                            remap_table[n] = r[7]
                        add_weights(remap_table[mesh.vertices["INDICES"]].reshape((-1, 4)), mesh.vertices["WEIGHTS"],
                                    [node.name for node in nodes],
                                    mesh_obj)
                        modifier = mesh_obj.modifiers.new(type="ARMATURE", name="Armature")
                        modifier.object = skeleton_object

                    collection.objects.link(mesh_obj)
    return {"FINISHED"}


plugin_info = {
    "name": "Gameloft engine",
    "id": "GameloftEngineLoader",
    "description": "Import Gameloft engine assets",
    "version": (0, 1, 2),
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
        {
            "name": "Load .pvr(or any other name) file",
            "id": "gle_pvr",
            "exts": ("*.*",),
            "init_fn": pvr_init,
            "import_fn": pvr_load,
            "properties": [
            ]
        },
    ],
    "init_fn": plugin_init
}
