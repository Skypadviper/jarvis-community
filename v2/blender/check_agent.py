"""Numeric sanity check of public/models/agent.glb's clips (no rendering).

    blender -b -P blender/check_agent.py
"""
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
bpy.ops.import_scene.gltf(filepath=os.path.join(os.path.dirname(HERE), "public", "models", "agent.glb"))
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mw = rig.matrix_world


def at(name):
    return mw @ rig.pose.bones[name].matrix.translation


def fmt(v):
    return "(" + ", ".join(f"{c:+.2f}" for c in v) + ")"


for clip in ("Idle", "Walk", "Sit", "Type", "Wave"):
    act = next((a for a in bpy.data.actions if a.name.startswith(clip)), None)
    if not act:
        print(f"[check] {clip}: MISSING")
        continue
    rig.animation_data.action = act
    if getattr(act, "slots", None) and rig.animation_data.action_slot is None:
        rig.animation_data.action_slot = act.slots[0]
    a, b = (int(v) for v in act.frame_range)
    rows = []
    for f in range(a, b + 1):
        bpy.context.scene.frame_set(f)
        rows.append({k: at(k) for k in ("hips", "head", "upperarm.L", "forearm.L", "hand.L", "upperarm.R", "forearm.R", "hand.R", "foot.L", "foot.R")})
    print(f"\n[check] {clip}  frames {a}-{b}")
    for r in rows[:: max(1, len(rows) // 4)][:4]:
        print(f"[check]   hips {fmt(r['hips'])}  handL {fmt(r['hand.L'])}  handR {fmt(r['hand.R'])}  elbowL {fmt(r['forearm.L'])}  footL {fmt(r['foot.L'])}  footR {fmt(r['foot.R'])}")
    ys_l = [r["hand.L"].y for r in rows]
    ys_r = [r["hand.R"].y for r in rows]
    print(f"[check]   hand swing front/back  L {min(ys_l):+.2f}..{max(ys_l):+.2f}  R {min(ys_r):+.2f}..{max(ys_r):+.2f}")
    print(f"[check]   hips height {min(r['hips'].z for r in rows):.2f}..{max(r['hips'].z for r in rows):.2f}  head height {min(r['head'].z for r in rows):.2f}")
