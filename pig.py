from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pprint import pprint
from typing import Optional

import numpy as np

from ...common_api import *


@dataclass(slots=True)
class Node:
    name: str
    type: int
    parent_id: int
    position: Vector3
    rotation: Vector4
    scale: Vector3
    unk0: float
    unk1: tuple[int, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        assert buffer.read_uint32() == 100, "Invalid ident"
        name = buffer.read_ascii_string(buffer.read_uint16())
        node_type, parent_id = buffer.read_fmt("Bh")
        position = buffer.read_fmt("3f")
        rotation = buffer.read_fmt("4f")
        scale = buffer.read_fmt("3f")
        unk0, *unk1 = buffer.read_fmt("f2B")
        return cls(name, node_type, parent_id, position, rotation, scale, unk0, unk1)


@dataclass(slots=True)
class TextureEntry:
    name: str
    unk0: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name_len = buffer.read_uint16()
        if name_len == 0:
            return None
        return cls(buffer.read_ascii_string(name_len), buffer.read_uint16())


class CompressionMode(IntEnum):
    LZ4 = 1
    ZSTD = 2


class VertexFormat(IntFlag):
    POSITION = 0x1
    NORMALS = 0x2
    TANGENT = 0x4
    UNK0x8 = 0x8
    UNK0x10 = 0x10
    UNK0x20 = 0x20
    UNK0x40 = 0x40
    UV1 = 0x80
    UV2 = 0x100


class MeshFlags(IntFlag):
    PACKED_VERTEX_DATA = 0x1
    UNK0x2 = 0x2
    UNK0x4 = 0x4
    UNK0x8 = 0x8
    UNK0x10 = 0x10
    UNK0x20 = 0x20
    UNK0x40 = 0x40
    UNK0x80 = 0x80
    UNK0x100 = 0x100
    UNK0x200 = 0x200
    UNK0x400 = 0x400
    UNK0x800 = 0x800
    UNK0x1000 = 0x1000
    UNK0x2000 = 0x2000
    UNK0x4000 = 0x4000
    UNK0x8000 = 0x8000
    UNK0x10000 = 0x10000
    UNK0x20000 = 0x20000
    UNK0x40000 = 0x40000
    UNK0x80000 = 0x80000
    UNK0x100000 = 0x100000
    UNK0x200000 = 0x200000
    UNK0x400000 = 0x400000
    UNK0x800000 = 0x800000
    UNK0x1000000 = 0x1000000
    UNK0x2000000 = 0x2000000
    UNK0x4000000 = 0x4000000
    UNK0x8000000 = 0x8000000
    UNK0x10000000 = 0x10000000
    UNK0x20000000 = 0x20000000
    UNK0x40000000 = 0x40000000
    UNK0x80000000 = 0x80000000
    UNK0x100000000 = 0x100000000


@dataclass(slots=True)
class Mesh:
    flags: int
    vertex_format: VertexFormat
    pivot_point: Vector3
    position: Optional[Vector3]
    size: Optional[Vector3]

    vertex_count: int
    face_count: int
    material_name: str
    unk0: int

    textures: list[TextureEntry]
    vertices: np.ndarray
    faces: np.ndarray
    unk_buffer: Optional[list]
    inv_bind_matrices: Optional[np.ndarray]

    @classmethod
    def from_buffer(cls, buffer: Buffer, have_skeleton: bool):
        assert buffer.read_uint32() == 100, "Invalid ident"
        flags, vertex_format = buffer.read_fmt("2I")
        flags = MeshFlags(flags)
        pivot_point = buffer.read_fmt("3f")
        offset = (0, 0, 0)
        size = (1, 1, 1)
        if flags & MeshFlags.PACKED_VERTEX_DATA:
            offset = buffer.read_fmt("3f")
            size = buffer.read_fmt("3f")
        vertex_count, face_count = buffer.read_fmt("HI")
        material_name = buffer.read_ascii_string(buffer.read_uint16())
        unk0 = buffer.read_uint16()
        textures = []
        for _ in range(8):
            texture = TextureEntry.from_buffer(buffer)
            textures.append(texture)
        buffer.align(4)
        while buffer.peek_uint32() == 0:
            buffer.skip(4)
        compression_mode = CompressionMode(buffer.read_uint8())
        compressed_size, decompressed_size = buffer.read_fmt("2I")
        if compression_mode == CompressionMode.ZSTD:
            geo_data = MemoryBuffer(zstd_decompress(buffer.read(compressed_size), decompressed_size))
        elif compression_mode == CompressionMode.LZ4:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        if have_skeleton:
            unk1, unk2, unk3 = buffer.read_fmt("2HI")
            compression_mode = CompressionMode(buffer.read_uint8())
            compressed_size, decompressed_size = buffer.read_fmt("2I")
            if compression_mode == CompressionMode.ZSTD:
                unk_data = zstd_decompress(buffer.read(compressed_size), decompressed_size)
            elif compression_mode == CompressionMode.LZ4:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            tmp = MemoryBuffer(unk_data)
            unk0_data = [tmp.read_fmt("6f2H") for _ in range(unk2)]
            bind_matrices = np.asarray([tmp.read_fmt("16f") for _ in range(unk2)]).reshape((-1, 4, 4))
        else:
            unk0_data = None
            bind_matrices = None

        vertex_attributes = []
        if vertex_format & VertexFormat.POSITION:
            vertex_attributes.append(("POSITION", np.float32, (3,)))
        if vertex_format & VertexFormat.NORMALS:
            vertex_attributes.append(("NORMAL", np.float32, (3,)))
        if vertex_format & VertexFormat.TANGENT:
            vertex_attributes.append(("TANGENT", np.float32, (4,)))
        if vertex_format & VertexFormat.UNK0x8:
            vertex_attributes.append(("UNK0x8", np.uint8, (4,)))
        if vertex_format & VertexFormat.UNK0x40:
            vertex_attributes.append(("UNK0x40", np.uint8, (4,)))
        if vertex_format & VertexFormat.UV1:
            vertex_attributes.append(("UV1", np.float32, (2,)))
        if vertex_format & VertexFormat.UV2:
            vertex_attributes.append(("UV1", np.float32, (2,)))
        if have_skeleton:
            vertex_attributes.append(("RIG_RELATED", np.uint8, (8,)))
        pprint(vertex_attributes)
        vertex_type = np.dtype(vertex_attributes)
        vertices = np.zeros(vertex_count, vertex_type)
        if flags & MeshFlags.PACKED_VERTEX_DATA:
            if vertex_format & VertexFormat.POSITION:
                positions = np.frombuffer(geo_data.read(vertex_count * 8), np.int16).reshape((-1, 4))[:, :3].astype(
                    np.float32)
                positions /= 32767
                positions *= size
                positions += offset
                vertices["POSITION"][:] = positions
            if vertex_format & VertexFormat.NORMALS:
                normals = np.frombuffer(geo_data.read(vertex_count * 4), np.int8)
                normals = normals.reshape((-1, 4))[:, :3].astype(np.float32)
                normals /= 127
                vertices["NORMAL"][:] = normals
            if vertex_format & VertexFormat.TANGENT:
                tangent = np.frombuffer(geo_data.read(vertex_count * 4), np.int8)
                tangent = tangent.reshape((-1, 4)).astype(np.float32)
                tangent /= 127
                vertices["TANGENT"][:] = tangent
            if vertex_format & VertexFormat.UNK0x8:
                vertices["UNK0x8"][:] = np.frombuffer(geo_data.read(vertex_count * 4), np.uint8).reshape((-1, 4))
            if vertex_format & VertexFormat.UNK0x40:
                vertices["UNK0x40"][:] = np.frombuffer(geo_data.read(vertex_count * 4), np.uint8).reshape((-1, 4))
            if vertex_format & VertexFormat.UV1:
                vertices["UV1"][:] = np.frombuffer(geo_data.read(vertex_count * 8), np.float32).reshape((-1, 2))
            if vertex_format & VertexFormat.UV2:
                vertices["UV2"][:] = np.frombuffer(geo_data.read(vertex_count * 8), np.float32).reshape((-1, 2))
            if have_skeleton:
                geo_data.skip(8 * vertex_count)
        else:
            if vertex_format & VertexFormat.POSITION:
                vertices["POSITION"][:] = np.frombuffer(geo_data.read(vertex_count * 12), np.float32).reshape((-1, 3))
            if vertex_format & VertexFormat.NORMALS:
                normals = np.frombuffer(geo_data.read(vertex_count * 4), np.int8)
                normals = normals.reshape((-1, 4))[:, :3].astype(np.float32)
                normals /= 127
                vertices["NORMAL"][:] = normals
            if vertex_format & VertexFormat.TANGENT:
                tangent = np.frombuffer(geo_data.read(vertex_count * 4), np.int8)
                tangent = tangent.reshape((-1, 4)).astype(np.float32)
                tangent /= 127
                vertices["TANGENT"][:] = tangent
            if vertex_format & VertexFormat.UNK0x8:
                vertices["UNK0x8"][:] = np.frombuffer(geo_data.read(vertex_count * 4), np.uint8).reshape((-1, 4))
            if vertex_format & VertexFormat.UNK0x40:
                vertices["UNK0x40"][:] = np.frombuffer(geo_data.read(vertex_count * 4), np.uint8).reshape((-1, 4))
            if vertex_format & VertexFormat.UV1:
                vertices["UV1"][:] = np.frombuffer(geo_data.read(vertex_count * 8), np.float32).reshape((-1, 2))
            if vertex_format & VertexFormat.UV2:
                vertices["UV2"][:] = np.frombuffer(geo_data.read(vertex_count * 8), np.float32).reshape((-1, 2))
            if have_skeleton:
                geo_data.skip(8 * vertex_count)
        _faces = np.frombuffer(geo_data.read(face_count * 2), np.int16)
        faces = []
        z = 0
        for i in range(0, len(_faces), 3):
            x = _faces[i] + z
            y = _faces[i + 1] + x
            z = _faces[i + 2] + y
            faces.append((x, y, z))
        faces = np.asarray(faces, np.uint32)

        return cls(flags, VertexFormat(vertex_format), pivot_point, offset, size, vertex_count, face_count,
                   material_name, unk0,
                   textures, vertices, faces, unk0_data, bind_matrices)


@dataclass(slots=True)
class Lod:
    id: int
    have_skeleton: bool
    bbox: tuple[Vector3, Vector3]
    meshes: list[Mesh]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        lod_id, ident, have_skeleton = buffer.read_fmt("BIH")
        assert ident == 100, "Invalid ident"
        bbox = buffer.read_fmt("3f"), buffer.read_fmt("3f")
        meshes = [Mesh.from_buffer(buffer, have_skeleton) for _ in range(buffer.read_uint16())]
        return cls(lod_id, have_skeleton == 1, bbox, meshes)


@dataclass(slots=True)
class Object:
    node_id: int
    lods: list[Lod]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        assert buffer.read_uint32() == 100, "Invalid ident"
        node_id, lod_count = buffer.read_fmt("IH")
        lods = [Lod.from_buffer(buffer) for _ in range(lod_count)]
        return cls(node_id, lods)


def load_pig(filename: str, buffer: Buffer):
    assert buffer.read_uint32() == 100, "Invalid ident"
    nodes = [Node.from_buffer(buffer) for _ in range(buffer.read_uint16())]
    buffer.skip(1)
    objects = [Object.from_buffer(buffer) for _ in range(buffer.read_uint16())]

    return nodes, objects
