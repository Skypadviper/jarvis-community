"""
Sculpts JARVIS's brain in Blender and exports it as a cloud of glowing particles
for the ops floor hub.

    blender -b -P blender/build_brain.py                 # particles + preview
    blender -b -P blender/build_brain.py -- --no-render

Writes:
    public/models/brain-points.bin   particles, float32 little-endian, 8 per point:
                                     x, y, z, nx, ny, nz, fold, region
    blender/renders/brain.png        cavity-shaded preview of the sculpt

The brain is sculpted procedurally on dense spheres: two hemispheres with a
fissure between them, a temporal lobe under a lateral fissure, gyri and sulci
everywhere, a cerebellum with fine folia, and a brain stem. Particles are then
scattered over the surface by area, so the folds read as light and shadow in
the points themselves. Each particle carries its surface normal (for lighting)
and its fold value (1 on a crest, 0 at the bottom of a groove).

Blender is Z-up with the brain facing +X; the file is written Y-up (x, z, -y),
matching what the glTF exporter does for the agents.
"""

import math
import os
import random
import struct
import sys

import bpy
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "public", "models", "brain-points.bin")
RENDERS = os.path.join(HERE, "renders")

POINTS = 110000
SEED = 7

FRONTAL, PARIETAL, TEMPORAL, OCCIPITAL, CEREBELLUM, STEM = range(6)

# Fold sharpness: how wide a groove is, as a fraction of the noise range.
GROOVE = 0.34
GROOVE_DEPTH = 0.04


def smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def gyri(p, side):
    """0 at the bottom of a sulcus, 1 on top of a gyrus.

    The zero-set of smooth noise is a web of thin wandering lines. Taking abs()
    and keeping only values near zero turns those lines into narrow grooves,
    and everything else into broad rounded ridges, which is what cortex looks
    like. A little domain warp makes the grooves meander like the real thing.
    """
    q = Vector((p.x * 11.0, p.y * 11.0 + side * 5.0, p.z * 11.0))
    q += noise.noise_vector(q * 0.3) * 1.1
    # Two overlapping webs of grooves: denser, and fewer closed loops.
    n = min(abs(noise.noise(q)), abs(noise.noise(q * 1.37 + Vector((9.1, 3.3, 7.7)))) * 1.25)
    g = smoothstep(0.0, GROOVE, n)
    # Round the crowns of the gyri instead of leaving flat plateaus.
    return math.sin(g * math.pi / 2) ** 0.8


def lateral_fissure(p, side):
    """The Sylvian fissure: a deep groove running back and up along the side,
    separating the temporal lobe below from the frontal and parietal lobes."""
    if p.y * side < 0.05:
        return 1.0
    line_z = -0.06 + (0.12 - p.x) * 0.32
    d = abs(p.z - line_z)
    along = smoothstep(-0.32, -0.12, p.x) * (1 - smoothstep(0.16, 0.26, p.x))
    return 1.0 - along * (1.0 - smoothstep(0.0, 0.035, d))


def sphere(name, center, scale, subdivisions=7):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def hemisphere(side):
    obj = sphere(f"hemisphere_{'L' if side < 0 else 'R'}", (0.0, 0.0, 0.0), (0.5, 0.205, 0.34))
    mesh = obj.data
    fold = mesh.attributes.new("fold", "FLOAT", "POINT")

    for v in mesh.vertices:
        p = v.co.copy()
        # Shape the egg into a hemisphere: flatter below, narrower behind,
        # medial face flattened against the fissure.
        if p.z < 0:
            p.z *= 0.74
        if p.x < 0:
            p.y *= 1.0 + p.x * 0.22
        medial = (p.y * side) < 0
        if medial:
            p.y *= 0.55
        else:
            # Temporal lobe swelling out low at the side.
            t = math.exp(-((p.x - 0.02) ** 2) / 0.035 - ((p.z + 0.13) ** 2) / 0.010)
            p.y += side * 0.055 * t
        p.y += side * 0.215
        p.z += 0.04
        v.co = p

    mesh.update()
    normals = [v.normal.copy() for v in mesh.vertices]
    for v, n in zip(mesh.vertices, normals):
        g = gyri(v.co, side) * lateral_fissure(v.co, side)
        # The medial wall is mostly hidden: fold it less.
        weight = 1.0 if (v.co.y * side) > 0.05 else 0.45
        v.co += n * GROOVE_DEPTH * (g - 1.0) * weight
        fold.data[v.index].value = g
    mesh.update()
    return obj


def cerebellum():
    obj = sphere("cerebellum", (-0.34, 0.0, -0.23), (0.19, 0.31, 0.125), subdivisions=6)
    mesh = obj.data
    fold = mesh.attributes.new("fold", "FLOAT", "POINT")
    normals = [v.normal.copy() for v in mesh.vertices]
    for v, n in zip(mesh.vertices, normals):
        # Folia: fine parallel ridges wrapping round it.
        f = 0.5 + 0.5 * math.sin((v.co.z + 0.23) * 150.0 + v.co.x * 20.0)
        g = smoothstep(0.15, 0.6, f)
        v.co += n * 0.012 * (g - 1.0)
        fold.data[v.index].value = 0.35 + 0.65 * g
    mesh.update()
    return obj


def stem():
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.07, depth=0.32, location=(-0.16, 0.0, -0.37))
    obj = bpy.context.active_object
    obj.name = "brain_stem"
    obj.rotation_euler = (0.0, math.radians(-16), 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=6)
    bpy.ops.object.mode_set(mode="OBJECT")
    fold = obj.data.attributes.new("fold", "FLOAT", "POINT")
    for d in fold.data:
        d.value = 0.6
    return obj


def region(c):
    if c.z < -0.3:
        return STEM
    if c.x < -0.2 and c.z < -0.13:
        return CEREBELLUM
    if c.z < -0.05 and abs(c.y) > 0.18 and c.x > -0.25:
        return TEMPORAL
    if c.x > 0.12:
        return FRONTAL
    if c.x < -0.2:
        return OCCIPITAL
    return PARIETAL


def scatter(objs, count):
    """Area-weighted random points over every triangle of every part."""
    rng = random.Random(SEED)
    tris = []
    for obj in objs:
        mesh = obj.data
        mesh.calc_loop_triangles()
        fold = mesh.attributes["fold"].data
        verts = mesh.vertices
        for t in mesh.loop_triangles:
            i, j, k = t.vertices
            if t.area > 0:
                tris.append((t.area, verts[i].co.copy(), verts[j].co.copy(), verts[k].co.copy(),
                             verts[i].normal.copy(), verts[j].normal.copy(), verts[k].normal.copy(),
                             fold[i].value, fold[j].value, fold[k].value))
    picks = rng.choices(tris, weights=[t[0] for t in tris], k=count)
    out = []
    for t in picks:
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        p = t[1] * w + t[2] * u + t[3] * v
        n = (t[4] * w + t[5] * u + t[6] * v).normalized()
        f = t[7] * w + t[8] * u + t[9] * v
        out.append((p, n, f))
    return out


def write(points):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        for p, n, fold in points:
            # Blender (x, y, z) Z-up -> three.js (x, z, -y) Y-up.
            f.write(struct.pack("<8f", p.x, p.z, -p.y, n.x, n.z, -n.y, fold, float(region(p))))
    print(f"[brain] {len(points)} particles -> {OUT} ({os.path.getsize(OUT) // 1024} KB)")


def preview():
    os.makedirs(RENDERS, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.45, 0.85, 0.95)
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.curvature_ridge_factor = 1.5
    shading.curvature_valley_factor = 2.0
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.004, 0.024, 0.047)
    scene.render.resolution_x, scene.render.resolution_y = 1100, 720
    bpy.ops.object.camera_add(location=(0.55, -1.9, 0.75))
    cam = bpy.context.active_object
    cam.rotation_euler = (Vector((0, 0, -0.04)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    scene.render.filepath = os.path.join(RENDERS, "brain.png")
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = [hemisphere(-1), hemisphere(1), cerebellum(), stem()]
    write(scatter(parts, POINTS))
    if "--no-render" not in argv:
        preview()


main()
