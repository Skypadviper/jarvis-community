"""
Builds the JARVIS operations floor assets in Blender and exports them as glTF.

    python3 blender/build_agents.py            # with the `bpy` wheel installed
    blender -b -P blender/build_agents.py      # or with a real Blender

Writes:
    public/models/agent.glb        one rigged humanoid, five animation clips
    public/models/workstation.glb  desk, monitor, keyboard and chair
    blender/renders/*.png          preview renders (pass --no-render to skip)

Everything is procedural, so the look is a code change rather than a binary
edit: tweak a number here, run it again, and the web app picks up the new GLBs.

Conventions, because they decide every sign below. Blender is Z-up and the
character faces -Y; the glTF exporter turns that into Y-up facing +Z, which is
what three.js expects. Rotations are written about *world* axes in the rest
pose (see `turn`) so that a positive angle about X always swings a limb
backwards, whatever roll Blender happened to give the bone.
"""

import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "public", "models")
RENDERS = os.path.join(HERE, "renders")
FPS = 30

X = Vector((1, 0, 0))
Y = Vector((0, 1, 0))
Z = Vector((0, 0, 1))


# -- scene helpers -------------------------------------------------------------


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


def material(name, color, emission=None, strength=0.0, metallic=0.0, rough=0.5, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    return m


def finish(obj, mat, group=None, smooth=True):
    """Assign a material, optionally put every vertex in one bone's group."""
    obj.data.materials.append(mat)
    if smooth:
        for p in obj.data.polygons:
            p.use_smooth = True
    if group:
        vg = obj.vertex_groups.new(name=group)
        vg.add([v.index for v in obj.data.vertices], 1.0, "REPLACE")
    return obj


def ellipsoid(center, scale, mat, group=None, segments=24, rings=14):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=center)
    o = bpy.context.object
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    return finish(o, mat, group)


def limb(a, b, radius, mat, group, taper=0.85):
    """An elongated, slightly tapered ellipsoid from joint a to joint b."""
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=14, location=(0, 0, 0))
    o = bpy.context.object
    half = d.length / 2 + radius * 0.35
    for v in o.data.vertices:
        # Taper toward b: the far end of a forearm is thinner than the elbow.
        t = (1 - v.co.z) / 2
        k = 1 - (1 - taper) * t
        v.co = Vector((v.co.x * radius * k, v.co.y * radius * k, v.co.z * half))
    # Built along -Z so the tapered end is at the -Z pole; aim -Z at b.
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (-Z).rotation_difference(d.normalized())
    o.location = (a + b) / 2
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return finish(o, mat, group)


def box(center, size, mat, group=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=center)
    o = bpy.context.object
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return finish(o, mat, group, smooth=bevel > 0)


def cylinder(center, radius, depth, mat, group=None, axis=Z, verts=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center)
    o = bpy.context.object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Z.rotation_difference(axis)
    bpy.ops.object.transform_apply(location=True, rotation=True)
    return finish(o, mat, group)


def ring(center, major, minor, mat, group=None, axis=Z):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major, minor_radius=minor, major_segments=32, minor_segments=8, location=center
    )
    o = bpy.context.object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Z.rotation_difference(axis)
    bpy.ops.object.transform_apply(location=True, rotation=True)
    return finish(o, mat, group)


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    o.data.name = name
    return o


# -- the agent -----------------------------------------------------------------

# Joint positions, in metres, standing. One table so the mesh and the skeleton
# can never disagree about where an elbow is.
J = {
    "hips": (0, 0, 0.95),
    "spine": (0, 0, 1.06),
    "chest": (0, 0, 1.28),
    "neck": (0, 0, 1.49),
    "head": (0, 0, 1.58),
    "head_end": (0, 0, 1.80),
}
for side, s in (("L", 1), ("R", -1)):
    J[f"shoulder.{side}"] = (0.21 * s, 0, 1.42)
    J[f"elbow.{side}"] = (0.25 * s, 0.01, 1.14)
    J[f"wrist.{side}"] = (0.27 * s, 0, 0.88)
    J[f"hand_end.{side}"] = (0.275 * s, -0.01, 0.78)
    J[f"hip.{side}"] = (0.1 * s, 0, 0.92)
    J[f"knee.{side}"] = (0.105 * s, -0.01, 0.5)
    J[f"ankle.{side}"] = (0.105 * s, 0.02, 0.09)
    J[f"toe.{side}"] = (0.11 * s, -0.13, 0.03)

BONES = [
    # name, head, tail, parent
    ("hips", "hips", "spine", None),
    ("spine", "spine", "chest", "hips"),
    ("chest", "chest", "neck", "spine"),
    ("neck", "neck", "head", "chest"),
    ("head", "head", "head_end", "neck"),
]
for side in ("L", "R"):
    BONES += [
        (f"upperarm.{side}", f"shoulder.{side}", f"elbow.{side}", "chest"),
        (f"forearm.{side}", f"elbow.{side}", f"wrist.{side}", f"upperarm.{side}"),
        (f"hand.{side}", f"wrist.{side}", f"hand_end.{side}", f"forearm.{side}"),
        (f"thigh.{side}", f"hip.{side}", f"knee.{side}", "hips"),
        (f"shin.{side}", f"knee.{side}", f"ankle.{side}", f"thigh.{side}"),
        (f"foot.{side}", f"ankle.{side}", f"toe.{side}", f"shin.{side}"),
    ]


def build_agent():
    # Body is the one material the web app recolours per agent; Glow is the
    # cyan circuitry; Visor is the face plate. Names are the contract.
    body = material("Body", (0.06, 0.09, 0.13), metallic=0.7, rough=0.32)
    shell = material("Shell", (0.75, 0.82, 0.88), metallic=0.4, rough=0.28)
    glow = material("Glow", (0.0, 0.9, 1.0), emission=(0.0, 0.9, 1.0), strength=6.0)
    visor = material("Visor", (0.0, 0.5, 0.6), emission=(0.0, 0.85, 1.0), strength=3.5, metallic=0.2, rough=0.1)

    parts = []
    p = parts.append

    # Torso: pelvis, abdomen, a broad chest, and shell plates over it.
    p(ellipsoid((0, 0, 0.96), (0.165, 0.11, 0.1), body, "hips"))
    p(ellipsoid((0, 0.005, 1.1), (0.14, 0.1, 0.12), body, "spine"))
    p(ellipsoid((0, 0, 1.31), (0.2, 0.125, 0.17), body, "chest"))
    p(ellipsoid((0, -0.035, 1.33), (0.175, 0.1, 0.13), shell, "chest"))
    p(ellipsoid((0.17, 0, 1.43), (0.075, 0.075, 0.06), shell, "chest"))
    p(ellipsoid((-0.17, 0, 1.43), (0.075, 0.075, 0.06), shell, "chest"))
    # The arc reactor — the one detail that says whose agents these are.
    p(cylinder((0, -0.133, 1.33), 0.032, 0.012, glow, "chest", axis=Y))
    p(ring((0, -0.13, 1.33), 0.045, 0.006, glow, "chest", axis=Y))
    p(ring((0, 0, 1.0), 0.15, 0.007, glow, "hips"))  # belt line

    # Neck and head.
    p(limb(J["neck"], (0, 0, 1.57), 0.045, body, "neck"))
    p(ellipsoid((0, 0.005, 1.67), (0.095, 0.11, 0.125), shell, "head"))
    p(ellipsoid((0, -0.045, 1.675), (0.08, 0.075, 0.045), visor, "head"))
    p(ring((0, 0.0, 1.67), 0.1, 0.005, glow, "head", axis=Vector((0, 0.25, 1)).normalized()))

    for side in ("L", "R"):
        s = 1 if side == "L" else -1
        p(limb(J[f"shoulder.{side}"], J[f"elbow.{side}"], 0.052, body, f"upperarm.{side}"))
        p(limb(J[f"elbow.{side}"], J[f"wrist.{side}"], 0.045, shell, f"forearm.{side}", taper=0.75))
        p(ring(J[f"elbow.{side}"], 0.045, 0.006, glow, f"forearm.{side}"))
        p(ellipsoid((0.272 * s, -0.005, 0.83), (0.03, 0.045, 0.06), body, f"hand.{side}"))
        p(limb(J[f"hip.{side}"], J[f"knee.{side}"], 0.075, body, f"thigh.{side}"))
        p(limb(J[f"knee.{side}"], J[f"ankle.{side}"], 0.058, shell, f"shin.{side}", taper=0.7))
        p(ring(J[f"knee.{side}"], 0.06, 0.007, glow, f"shin.{side}"))
        p(ellipsoid((0.108 * s, -0.05, 0.045), (0.05, 0.11, 0.045), body, f"foot.{side}"))

    mesh = join(parts, "Agent")

    # Skeleton.
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    rig = bpy.context.object
    rig.name = "AgentRig"
    rig.data.name = "AgentRig"
    eb = rig.data.edit_bones
    eb.remove(eb[0])
    for name, h, t, parent in BONES:
        b = eb.new(name)
        b.head, b.tail = J[h], J[t]
        if parent:
            b.parent = eb[parent]
            b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh.parent = rig
    mod = mesh.modifiers.new("rig", "ARMATURE")
    mod.object = rig
    return rig


# -- animation -----------------------------------------------------------------


def turn(rig, bone, axis, degrees):
    """Pose `bone` rotated about a world axis, measured in its rest frame."""
    pb = rig.pose.bones[bone]
    pb.rotation_mode = "QUATERNION"
    local = rig.data.bones[bone].matrix_local.to_3x3().inverted() @ axis
    pb.rotation_quaternion = Quaternion(local.normalized(), math.radians(degrees))


def shift(rig, bone, offset):
    pb = rig.pose.bones[bone]
    pb.location = rig.data.bones[bone].matrix_local.to_3x3().inverted() @ Vector(offset)


def clear(rig):
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)


def key(rig, frame):
    for pb in rig.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb.keyframe_insert("location", frame=frame)


def action(rig, name, frames, pose):
    """Key `pose(rig, phase)` across `frames`, phase running 0..1 and looping."""
    rig.animation_data_create()
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    rig.animation_data.action = act
    steps = 8 if frames > 8 else frames
    for i in range(steps + 1):
        f = 1 + round(i * frames / steps)
        clear(rig)
        pose(rig, (i / steps) % 1.0)
        key(rig, f)
    # Make the loop seam exact rather than "probably close".
    for fc in act.fcurves:
        fc.modifiers.new("CYCLES")
    act.frame_range = (1, frames + 1)
    return act


def arms_relaxed(rig, wobble=0.0):
    turn(rig, "upperarm.L", Y, -6)
    turn(rig, "upperarm.R", Y, 6)
    turn(rig, "forearm.L", X, -8 - wobble)
    turn(rig, "forearm.R", X, -8 + wobble)


def pose_idle(rig, t):
    w = math.sin(t * 2 * math.pi)
    turn(rig, "chest", X, -1.5 * w)
    turn(rig, "head", Z, 6 * math.sin(t * 2 * math.pi + 1))
    arms_relaxed(rig, 2 * w)
    shift(rig, "hips", (0, 0, 0.004 * w))


def pose_walk(rig, t):
    a = math.sin(t * 2 * math.pi)
    lift = abs(math.cos(t * 2 * math.pi))
    shift(rig, "hips", (0, 0, 0.025 * lift - 0.015))
    turn(rig, "hips", Z, 5 * a)
    turn(rig, "chest", Z, -8 * a)
    turn(rig, "chest", X, -4)
    turn(rig, "thigh.L", X, -26 * a)
    turn(rig, "thigh.R", X, 26 * a)
    # The knee bends on the swing, which is the half where the thigh comes forward.
    turn(rig, "shin.L", X, 8 + 32 * max(0, math.sin(t * 2 * math.pi + 1.4)))
    turn(rig, "shin.R", X, 8 + 32 * max(0, math.sin(t * 2 * math.pi + math.pi + 1.4)))
    turn(rig, "upperarm.L", X, 22 * a)
    turn(rig, "upperarm.R", X, -22 * a)
    turn(rig, "upperarm.L", Y, -5)
    turn(rig, "upperarm.R", Y, 5)
    turn(rig, "forearm.L", X, -18 - 10 * max(0, -a))
    turn(rig, "forearm.R", X, -18 - 10 * max(0, a))


def pose_seated(rig):
    # Drop the pelvis onto the seat: knees at seat height, feet on the floor.
    shift(rig, "hips", (0, 0.02, -0.43))
    for side in ("L", "R"):
        turn(rig, f"thigh.{side}", X, -88)
        turn(rig, f"shin.{side}", X, 0)
        pb = rig.pose.bones[f"shin.{side}"]
        # Shin rotates relative to the thigh, so undo the thigh to point down.
        pb.rotation_quaternion = Quaternion(
            (rig.data.bones[f"shin.{side}"].matrix_local.to_3x3().inverted() @ X).normalized(),
            math.radians(88),
        )
    turn(rig, "spine", X, -4)


def pose_sit(rig, t):
    pose_seated(rig)
    w = math.sin(t * 2 * math.pi)
    turn(rig, "chest", X, -1.5 * w)
    turn(rig, "upperarm.L", X, -22)
    turn(rig, "upperarm.R", X, -22)
    turn(rig, "forearm.L", X, -45)
    turn(rig, "forearm.R", X, -45)


def pose_type(rig, t):
    pose_seated(rig)
    a = math.sin(t * 2 * math.pi * 2)
    b = math.sin(t * 2 * math.pi * 3 + 1)
    turn(rig, "chest", X, -6)
    turn(rig, "head", X, -10 + 1.5 * math.sin(t * 2 * math.pi))
    turn(rig, "head", Z, 4 * math.sin(t * 2 * math.pi))
    for side, s, k in (("L", 1, a), ("R", -1, b)):
        pb = rig.pose.bones[f"upperarm.{side}"]
        q1 = Quaternion((rig.data.bones[f"upperarm.{side}"].matrix_local.to_3x3().inverted() @ X).normalized(), math.radians(-40))
        q2 = Quaternion((rig.data.bones[f"upperarm.{side}"].matrix_local.to_3x3().inverted() @ Y).normalized(), math.radians(10 * s))
        pb.rotation_quaternion = q2 @ q1
        turn(rig, f"forearm.{side}", X, -52 + 3 * k)
        turn(rig, f"hand.{side}", X, 12 + 10 * k)


def pose_wave(rig, t):
    a = math.sin(t * 2 * math.pi)
    turn(rig, "chest", Y, -3)
    turn(rig, "head", Z, -8)
    turn(rig, "upperarm.L", Y, -6)
    turn(rig, "forearm.L", X, -10)
    pb = rig.pose.bones["upperarm.R"]
    q1 = Quaternion((rig.data.bones["upperarm.R"].matrix_local.to_3x3().inverted() @ Y).normalized(), math.radians(150))
    pb.rotation_quaternion = q1
    turn(rig, "forearm.R", Y, 25 * a)
    turn(rig, "hand.R", Y, 10 * a)


def animate(rig):
    action(rig, "Idle", 60, pose_idle)
    action(rig, "Walk", 30, pose_walk)
    action(rig, "Sit", 60, pose_sit)
    action(rig, "Type", 36, pose_type)
    action(rig, "Wave", 30, pose_wave)


# -- the workstation -----------------------------------------------------------

# Laid out so the chair sits at +Y and the screen faces it: after the glTF axis
# change that puts the seat at z = -0.6 and an agent sitting in it facing +Z,
# towards the desk — the same way the agent faces at rest.
CHAIR_Y = 0.6


def build_workstation():
    frame = material("Frame", (0.05, 0.07, 0.1), metallic=0.8, rough=0.35)
    glass = material("Glass", (0.0, 0.5, 0.65), emission=(0.0, 0.6, 0.8), strength=0.6, alpha=0.35, rough=0.05)
    edge = material("Edge", (0.0, 0.9, 1.0), emission=(0.0, 0.9, 1.0), strength=5.0)
    screen = material("Screen", (0.0, 0.1, 0.14), emission=(0.0, 0.75, 1.0), strength=2.5, rough=0.1)
    seat = material("Seat", (0.1, 0.12, 0.16), metallic=0.1, rough=0.7)

    parts = []
    p = parts.append

    # Desk: a glass top on a dark frame, cyan strip along the front edge.
    p(box((0, 0, 0.74), (1.4, 0.7, 0.025), glass))
    p(box((0, 0.35, 0.735), (1.4, 0.012, 0.012), edge))
    for x in (-0.66, 0.66):
        p(box((x, 0, 0.36), (0.05, 0.62, 0.72), frame, bevel=0.01))
    p(box((0, -0.3, 0.55), (1.3, 0.03, 0.3), frame))

    # Monitor, facing the chair.
    p(box((0, -0.18, 0.76), (0.26, 0.18, 0.012), frame, bevel=0.004))
    p(box((0, -0.2, 0.9), (0.04, 0.03, 0.28), frame))
    p(box((0, -0.19, 1.07), (0.82, 0.035, 0.48), frame, bevel=0.008))
    scr = box((0, -0.17, 1.07), (0.78, 0.004, 0.44), screen)
    scr.name = "Screen"
    scr.data.name = "Screen"

    # Keyboard and mouse.
    p(box((0, 0.17, 0.765), (0.46, 0.15, 0.018), frame, bevel=0.004))
    p(box((0, 0.17, 0.776), (0.43, 0.12, 0.004), edge))
    p(ellipsoid((0.34, 0.18, 0.765), (0.03, 0.05, 0.015), frame))

    desk = join(parts, "Desk")

    # Chair, separate so the app can roll it back when someone sits down.
    cp = []
    cp.append(box((0, CHAIR_Y, 0.47), (0.48, 0.46, 0.07), seat, bevel=0.025))
    cp.append(box((0, CHAIR_Y + 0.25, 0.82), (0.46, 0.06, 0.55), seat, bevel=0.025))
    cp.append(box((0, CHAIR_Y + 0.215, 0.62), (0.05, 0.03, 0.3), frame))
    cp.append(cylinder((0, CHAIR_Y, 0.25), 0.025, 0.4, frame))
    cp.append(ring((0, CHAIR_Y, 0.06), 0.22, 0.012, edge))
    for i in range(5):
        a = i * 2 * math.pi / 5
        cp.append(limb((0, CHAIR_Y, 0.06), (0.25 * math.cos(a), CHAIR_Y + 0.25 * math.sin(a), 0.035), 0.018, frame, None))
    chair = join(cp, "Chair")

    return [desk, scr, chair]


# -- export & preview ----------------------------------------------------------


def export(path, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
        for c in o.children:
            c.select_set(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_frame_step=1,
        export_lights=False,
        export_cameras=False,
    )
    print(f"[blender] wrote {os.path.relpath(path, ROOT)}")


def preview(name, look_at, distance, height, angle=25, res=(960, 540)):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"

    if not sc.world:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.012, 0.022, 1)

    cam = bpy.data.objects.get("PreviewCam")
    if cam is None:
        cam = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
        sc.collection.objects.link(cam)
        key_light = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
        key_light.data.energy = 600
        key_light.data.size = 3
        key_light.data.color = (0.75, 0.9, 1.0)
        key_light.location = (2.5, -3, 4)
        key_light.rotation_euler = (math.radians(50), 0, math.radians(40))
        sc.collection.objects.link(key_light)
        rim = bpy.data.objects.new("Rim", bpy.data.lights.new("Rim", "AREA"))
        rim.data.energy = 400
        rim.data.size = 2
        rim.data.color = (0.0, 0.8, 1.0)
        rim.location = (-2.5, 3, 3)
        rim.rotation_euler = (math.radians(-50), 0, math.radians(-140))
        sc.collection.objects.link(rim)
        bpy.ops.mesh.primitive_plane_add(size=20)
        floor = bpy.context.object
        floor.name = "PreviewFloor"
        floor.data.materials.append(material("Floor", (0.01, 0.025, 0.04), metallic=0.6, rough=0.25))
    sc.camera = cam
    a = math.radians(angle)
    target = Vector(look_at)
    cam.location = target + Vector((math.sin(a) * distance, -math.cos(a) * distance, height))
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - cam.location).to_track_quat("-Z", "Y")
    cam.data.lens = 50

    os.makedirs(RENDERS, exist_ok=True)
    sc.render.filepath = os.path.join(RENDERS, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] rendered {os.path.relpath(sc.render.filepath, ROOT)}")


def pose_still(rig, act, frame):
    rig.animation_data.action = bpy.data.actions[act]
    bpy.context.scene.frame_set(frame)


def main():
    render = "--no-render" not in sys.argv

    reset()
    rig = build_agent()
    animate(rig)
    rig.animation_data.action = bpy.data.actions["Idle"]
    export(os.path.join(MODELS, "agent.glb"), [rig])

    reset()
    station = build_workstation()
    export(os.path.join(MODELS, "workstation.glb"), station)

    if not render:
        return

    # One staged shot of the whole idea: a desk with an agent typing at it,
    # another walking up, a third waving hello.
    reset()
    build_workstation()
    typing = build_agent()
    animate(typing)
    typing.location = (0, CHAIR_Y, 0)
    pose_still(typing, "Type", 10)

    walker = build_agent()
    animate(walker)
    walker.location = (1.3, -0.9, 0)
    walker.rotation_euler = (0, 0, math.radians(120))
    pose_still(walker, "Walk", 8)

    waver = build_agent()
    animate(waver)
    waver.location = (-1.4, -0.7, 0)
    waver.rotation_euler = (0, 0, math.radians(-30))
    pose_still(waver, "Wave", 8)
    bpy.context.view_layer.update()

    preview("ops_floor", (0, 0, 0.9), 6.2, 2.0, angle=200)
    preview("agent_typing", (0, CHAIR_Y - 0.1, 0.9), 2.4, 0.6, angle=130)

    # A character sheet: the standing clips side by side, facing the camera.
    reset()
    for i, clip in enumerate(("Idle", "Walk", "Wave")):
        r = build_agent()
        animate(r)
        r.location = ((i - 1) * 0.9, 0, 0)
        pose_still(r, clip, 8)
    bpy.context.view_layer.update()
    preview("lineup", (0, 0, 0.95), 4.4, 0.4, angle=0)


if __name__ == "__main__":
    main()
