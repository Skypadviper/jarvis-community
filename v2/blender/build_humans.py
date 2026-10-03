"""
Builds the ops-floor agents as realistic humans with MPFB (MakeHuman for Blender)
and exports them in place of the procedural agent.

    blender -b -P blender/build_humans.py                 # agent.glb + preview
    blender -b -P blender/build_humans.py -- --no-render

Needs, once per machine:
    - the MPFB extension (Blender > Get Extensions > MPFB), and
    - the CC0 MakeHuman asset packs "makehuman_system_assets" and "suits01",
      unpacked into MPFB's user data folder.

Writes:
    public/models/agent.glb       the same contract as build_agents.py: a rig with
                                  the bone names below and the clips Idle, Walk,
                                  Sit, Type and Wave; materials named Glow (lapel
                                  pin, smartwatch) and Visor (AR glasses lenses),
                                  which the web app tints per agent
    blender/renders/humans.png    preview

The human gets MPFB's game-engine rig, whose skin weights are painted properly.
Its main bones are renamed to the names build_agents.py animates, and its arms
are lowered from MakeHuman's A-pose to hang at the sides, then that is made the
rest pose. The animation code in build_agents.py then drives it unchanged.
"""

import glob
import json
import math
import os
import sys

import addon_utils
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_agents as base  # noqa: E402  (reuse its helpers)
import guns  # noqa: E402  (Quaternius Ultimate Gun Pack loadouts)
import retarget_ual  # noqa: E402  (real motion from the Universal Animation Library)

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "public", "models", "agent.glb")
TEXTURE_SIZE = 1024

SKIN = "skins/young_caucasian_male/young_caucasian_male.mhmat"
SUIT = "clothes/toigo_male_suit_tie_and_jacket/toigo_male_suit_tie_and_jacket.mhclo"
SHOES = "clothes/shoes02/shoes02.mhclo"
HAIR = "hair/short02/short02.mhclo"
EYEBROWS = "eyebrows/eyebrow001/eyebrow001.mhclo"
EYELASHES = "eyelashes/eyelashes01/eyelashes01.mhclo"
EYES = "eyes/high-poly/high-poly.mhclo"
EYE_MATERIAL = "eyes/materials/brown.mhmat"

# MPFB game-engine bone -> the name build_agents.py animates.
RENAME = {
    "pelvis": "hips", "spine_01": "spine", "spine_03": "chest", "neck_01": "neck", "head": "head",
    "upperarm_l": "upperarm.L", "lowerarm_l": "forearm.L", "hand_l": "hand.L",
    "upperarm_r": "upperarm.R", "lowerarm_r": "forearm.R", "hand_r": "hand.R",
    "thigh_l": "thigh.L", "calf_l": "shin.L", "foot_l": "foot.L",
    "thigh_r": "thigh.R", "calf_r": "shin.R", "foot_r": "foot.R",
}


def mpfb():
    """Enable MPFB in this session and return its services."""
    for mod in addon_utils.modules():
        if mod.__name__.endswith(".mpfb"):
            addon_utils.enable(mod.__name__, default_set=True)
            pkg = mod.__name__
            break
    else:
        sys.exit("MPFB is not installed: Blender > Edit > Preferences > Get Extensions > MPFB")
    import importlib
    hs = importlib.import_module(pkg + ".services.humanservice").HumanService
    ls = importlib.import_module(pkg + ".services.locationservice").LocationService
    return hs, ls


def asset(ls, rel):
    path = ls.get_user_data(rel)
    if not os.path.exists(path):
        # Some packs name their folders slightly differently; take the first match.
        hits = glob.glob(ls.get_user_data(os.path.join(os.path.dirname(os.path.dirname(rel)), "*", os.path.basename(rel).split(".")[-1].join(["*.", ""]))))
        if not hits:
            sys.exit(f"Missing MakeHuman asset: {rel} (is the asset pack installed?)")
        path = hits[0]
    return path


def find(ls, folder, ext):
    hits = sorted(glob.glob(os.path.join(ls.get_user_data(folder), "**", f"*.{ext}"), recursive=True))
    if not hits:
        sys.exit(f"No .{ext} under {folder} (is the asset pack installed?)")
    return hits[0]


def build_human(hs, ls):
    basemesh = hs.create_human(feet_on_ground=True, scale=0.1)
    hs.add_builtin_rig(basemesh, "game_engine")
    rig = basemesh.parent
    hs.set_character_skin(asset(ls, SKIN), basemesh, skin_type="GAMEENGINE")

    def wear(rel, kind, folder=None, ext="mhclo"):
        path = ls.get_user_data(rel)
        if not os.path.exists(path):
            path = find(ls, folder or os.path.dirname(os.path.dirname(rel)), ext)
        hs.add_mhclo_asset(path, basemesh, asset_type=kind, subdiv_levels=0, material_type="GAMEENGINE")

    wear(EYES, "Eyes", "eyes/high-poly")
    wear(EYEBROWS, "Eyebrows", "eyebrows")
    wear(EYELASHES, "Eyelashes", "eyelashes")
    wear(HAIR, "Hair", "hair/short02")
    wear(SUIT, "Clothes", "clothes/toigo_male_suit_tie_and_jacket")
    wear(SHOES, "Clothes", "clothes/shoes02")
    return rig, basemesh


def meshes_of(rig):
    return [o for o in rig.children_recursive if o.type == "MESH"]


def bake(obj):
    """Freeze shape keys and every modifier except the armature, so the rest
    pose can be changed underneath it."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if obj.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    for m in list(obj.modifiers):
        if m.type == "ARMATURE":
            continue
        if m.type == "SUBSURF":
            obj.modifiers.remove(m)
            continue
        try:
            bpy.ops.object.modifier_apply(modifier=m.name)
        except RuntimeError:
            obj.modifiers.remove(m)


def arms_down(rig):
    """Swing the upper arms from MakeHuman's A-pose to hang at the sides, and
    make that the rest pose, which is what the animations assume."""
    base.clear(rig)
    for side, s in (("L", 1), ("R", -1)):
        b = rig.data.bones[f"upperarm.{side}"]
        d = (b.tail_local - b.head_local).normalized()
        # Angle out from straight down, in the front plane.
        angle = math.degrees(math.atan2(abs(d.x), -d.z))
        base.turn(rig, f"upperarm.{side}", base.Y, s * (angle - 6))
    bpy.context.view_layer.update()

    for obj in meshes_of(rig):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for m in list(obj.modifiers):
            if m.type == "ARMATURE":
                name = m.name
                bpy.ops.object.modifier_apply(modifier=name)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    for obj in meshes_of(rig):
        mod = obj.modifiers.new("rig", "ARMATURE")
        mod.object = rig


def accessories(rig, body):
    """Lapel pin, smartwatch and AR glasses, rigid on their bones."""
    glow = base.material("Glow", (0.0, 0.9, 1.0), emission=(0.0, 0.9, 1.0), strength=4.0)
    visor = base.material("Visor", (0.02, 0.06, 0.08), emission=(0.0, 0.85, 1.0), strength=0.35, metallic=0.4, rough=0.05)
    frame = base.material("Frame", (0.02, 0.02, 0.025), metallic=0.8, rough=0.3)

    world = [body.matrix_world @ v.co for v in body.data.vertices]
    bones = rig.data.bones

    def front_at(x, z, tol=0.025):
        ys = [p.y for p in world if abs(p.x - x) < tol and abs(p.z - z) < tol]
        return min(ys) if ys else -0.12

    chest = bones["chest"].head_local
    pin_z = chest.z + 0.06
    pin = base.ellipsoid((0.085, front_at(0.085, pin_z) - 0.03, pin_z), (0.011, 0.005, 0.011), glow, "chest", segments=12, rings=8)

    wrist = bones["hand.L"].head_local
    watch = base.ring(wrist + Vector((0, 0, 0.03)), 0.034, 0.007, glow, "forearm.L")

    # Glasses sit in front of the eyes.
    def is_eyes(o):
        n = o.name.lower()
        return ("eye" in n or "high-poly" in n or "low-poly" in n) and "brow" not in n and "lash" not in n
    eyes = [o for o in meshes_of(rig) if is_eyes(o)]
    print("[humans] eyes:", [o.name for o in eyes], "all:", [o.name for o in meshes_of(rig)])
    pts = [e.matrix_world @ v.co for e in eyes for v in e.data.vertices]
    parts = [pin, watch]
    if pts:
        for s in (1, -1):
            side = [p for p in pts if p.x * s > 0]
            c = sum(side, Vector()) / len(side)
            front = min(p.y for p in side)
            parts.append(base.ellipsoid((c.x, front - 0.012, c.z), (0.021, 0.003, 0.014), visor, "head", segments=16, rings=8))
            parts.append(base.limb((c.x + 0.022 * s, front - 0.01, c.z + 0.004), (c.x + 0.048 * s, front + 0.07, c.z + 0.008), 0.0025, frame, "head"))
        y0 = min(p.y for p in pts) - 0.012
        zc = sum(p.z for p in pts) / len(pts)
        parts.append(base.limb((-0.012, y0, zc + 0.004), (0.012, y0, zc + 0.004), 0.0025, frame, "head"))

    kit = base.join(parts, "Accessories")
    kit.parent = rig
    mod = kit.modifiers.new("rig", "ARMATURE")
    mod.object = rig
    return kit


def shrink_textures():
    for img in bpy.data.images:
        if img.size[0] > TEXTURE_SIZE or img.size[1] > TEXTURE_SIZE:
            img.scale(TEXTURE_SIZE, TEXTURE_SIZE)


def export(rig):
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    for o in rig.children_recursive:
        o.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=OUT,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_frame_step=1,
        export_lights=False,
        export_cameras=False,
        export_image_format="JPEG",
    )
    print(f"[humans] wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) // 1024} KB)")


def build(hs, ls):
    rig, body = build_human(hs, ls)
    rig.name = "AgentRig"
    for old, new in RENAME.items():
        rig.data.bones[old].name = new
    for obj in meshes_of(rig):
        bake(obj)
    arms_down(rig)
    accessories(rig, body)
    guns.arm(rig)
    return rig


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    base.reset()
    hs, ls = mpfb()
    rig = build(hs, ls)
    speed = retarget_ual.animate(rig)
    with open(os.path.join(ROOT, 'public', 'models', 'agent.walk.json'), 'w') as f:
        f.write(json.dumps({'walkSpeed': round(speed, 3)}))
    rig.animation_data.action = bpy.data.actions["Idle"]
    shrink_textures()
    export(rig)

    if "--no-render" in argv:
        return
    # Preview: the standing clips side by side, facing the camera.
    base.reset()
    hs, ls = mpfb()
    for i, clip in enumerate(("Idle", "Walk", "Wave")):
        r = build(hs, ls)
        retarget_ual.animate(r)
        r.location = ((i - 1) * 0.9, 0, 0)
        base.pose_still(r, clip, 8)
    bpy.context.view_layer.update()
    base.preview("humans", (0, 0, 0.95), 4.4, 0.4, angle=0)


main()
