"""
Real motion for the human agents, from Quaternius's Universal Animation Library
(CC0, blender/assets/ual/UAL1_Standard.glb).

    import retarget_ual
    speed = retarget_ual.animate(rig)    # adds Idle, Walk, Sit, Type, Wave

Retargeting is done in world space. For every mapped bone the source's rotation
is carried over with a fixed offset that also aligns the two rest poses: the
library's skeleton rests in a T-pose and the agent's with its arms down, so each
target bone is first swung onto the source bone's rest direction. That is what
lets a library idle hang the arms naturally instead of adding its own drop on
top of an arm that already hangs.

Idle, Walk and Sit are the library's motion capture as is, fingers included.
The library has no typing or waving, so those keep its real sitting and idle
motion for the body and drive the arms with two-bone IK: hands on the keyboard
with palms down, the elbows out and back, irregular wrist and finger taps, and
a slow drift of the hands; or a raised arm swinging from the elbow.
"""

import math
import os

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
UAL = os.path.join(HERE, "assets", "ual", "UAL1_Standard.glb")

# Library bone -> agent bone. Finger names are shared.
MAP = {
    "pelvis": "hips", "spine_01": "spine", "spine_02": "spine_02", "spine_03": "chest",
    "neck_01": "neck", "Head": "head",
}
for _s, _S in (("l", "L"), ("r", "R")):
    MAP.update({
        f"clavicle_{_s}": f"clavicle_{_s}", f"upperarm_{_s}": f"upperarm.{_S}",
        f"lowerarm_{_s}": f"forearm.{_S}", f"hand_{_s}": f"hand.{_S}",
        f"thigh_{_s}": f"thigh.{_S}", f"calf_{_s}": f"shin.{_S}", f"foot_{_s}": f"foot.{_S}",
        f"ball_{_s}": f"ball_{_s}",
    })
    for _f in ("index", "middle", "ring", "pinky", "thumb"):
        for _i in (1, 2, 3):
            MAP[f"{_f}_0{_i}_{_s}"] = f"{_f}_0{_i}_{_s}"

# Where the hands go when typing, in the agent's own space while seated: the
# workstation keyboard sits 0.43 m in front of the chair centre at 0.78 m.
KEYS_Y = -0.34   # wrist, a little short of the keys so the fingers lie on them
KEYS_Z = 0.80
KEYS_X = 0.14


# -- helpers ------------------------------------------------------------------------

def ordered(rig):
    out = []

    def walk(b):
        out.append(b)
        for c in b.children:
            walk(c)

    for b in rig.data.bones:
        if b.parent is None:
            walk(b)
    return out


def rot(m):
    return m.to_3x3().normalized()


def frame_from(y_dir, normal_hint, y_local, n_local):
    """World rotation that puts the bone's local `y_local` along `y_dir` and its
    local `n_local` as close as possible to `normal_hint`."""
    y = y_dir.normalized()
    n = (normal_hint - y * normal_hint.dot(y)).normalized()
    x = y.cross(n)
    world = Matrix((x, y, n)).transposed()
    yl = y_local.normalized()
    nl = (n_local - yl * n_local.dot(yl)).normalized()
    xl = yl.cross(nl)
    local = Matrix((xl, yl, nl)).transposed()
    return world @ local.inverted()


def two_bone(s, t, l1, l2, pole):
    d = t - s
    dist = min(d.length, (l1 + l2) * 0.999)
    u = d.normalized()
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    v = (pole - u * pole.dot(u)).normalized()
    return s + u * a + v * h, s + u * dist


def axis_rot(axis, deg):
    return Matrix.Rotation(math.radians(deg), 3, axis)


# -- retargeting -----------------------------------------------------------------------

class Retarget:
    def __init__(self, rig):
        self.rig = rig
        before = set(bpy.data.objects)
        acts_before = set(bpy.data.actions)
        bpy.ops.import_scene.gltf(filepath=UAL)
        self.imported = [o for o in bpy.data.objects if o not in before]
        self.src_actions = [a for a in bpy.data.actions if a not in acts_before]
        self.src = next(o for o in self.imported if o.type == "ARMATURE")
        for o in self.imported:
            if o.type == "MESH":
                o.hide_render = True

        self.bones = ordered(rig)
        self.rest = {b.name: rig.matrix_world @ b.matrix_local for b in rig.data.bones}
        self.rel = {b.name: (self.rest[b.parent.name].inverted() @ self.rest[b.name]) if b.parent else self.rest[b.name]
                    for b in rig.data.bones}
        srest = {b.name: self.src.matrix_world @ b.matrix_local for b in self.src.data.bones}
        self.offset = {}
        for s, t in MAP.items():
            if s not in srest or t not in self.rest:
                continue
            rs, rt = rot(srest[s]), rot(self.rest[t])
            q = (rt @ Vector((0, 1, 0))).rotation_difference(rs @ Vector((0, 1, 0))).to_matrix()
            self.offset[t] = (s, rs.inverted() @ q @ rt)
        self.src_hips = srest["pelvis"].translation.copy()
        self.tgt_hips = self.rest["hips"].translation.copy()
        self.scale = self.tgt_hips.z / self.src_hips.z
        # The agent's own palm direction: whatever local axis faces down when the
        # hand is swung onto the library's T-pose (palms down).
        self.palm = {}
        for side in ("L", "R"):
            for b in (f"forearm.{side}", f"hand.{side}"):
                s, c = self.offset[b]
                aligned = rot(srest[s]) @ c
                self.palm[b] = aligned.inverted() @ Vector((0, 0, -1))

    def sample(self, action, frame):
        src = self.src
        src.animation_data_create()
        src.animation_data.action = action
        if getattr(src.animation_data, "action_slot", None) is None and getattr(action, "slots", None):
            src.animation_data.action_slot = action.slots[0]
        bpy.context.scene.frame_set(frame)
        mw = src.matrix_world
        return {pb.name: mw @ pb.matrix for pb in src.pose.bones}

    def solve(self, pose, anchor=None, override=None):
        """World matrices for every agent bone from one library frame. `override`
        may replace any bone's world rotation after its parent is solved."""
        world = {}
        for b in self.bones:
            base = (world[b.parent.name] @ self.rel[b.name]) if b.parent else self.rest[b.name]
            pos = base.translation
            r = rot(base)
            if b.name in self.offset:
                s, c = self.offset[b.name]
                if s in pose:
                    r = rot(pose[s]) @ c
            if b.name == "hips":
                delta = (pose["pelvis"].translation - self.src_hips) * self.scale
                if anchor is not None:
                    delta.x -= anchor.x
                    delta.y -= anchor.y
                pos = self.tgt_hips + delta
            m = Matrix.Translation(pos) @ r.to_4x4()
            if override:
                m = override(b.name, m, world) or m
            world[b.name] = m
        return world

    def key(self, world, frame, prev):
        rig = self.rig
        for b in self.bones:
            pb = rig.pose.bones[b.name]
            parent = world[b.parent.name] if b.parent else Matrix.Identity(4)
            basis = self.rel[b.name].inverted() @ parent.inverted() @ world[b.name]
            q = basis.to_quaternion()
            if b.name in prev and prev[b.name].dot(q) < 0:
                q.negate()
            prev[b.name] = q
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=frame)
            if b.name == "hips":
                pb.location = basis.translation
                pb.keyframe_insert("location", frame=frame)

    def clip(self, name, source, frames=None, anchored=False, override=None):
        act_src = bpy.data.actions[source]
        start, end = (int(v) for v in act_src.frame_range)
        n = frames or (end - start)
        rig = self.rig
        rig.animation_data_create()
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
        rig.animation_data.action = act
        anchor = None
        if anchored:
            first = self.sample(act_src, start)
            anchor = (first["pelvis"].translation - self.src_hips) * self.scale
        prev = {}
        for i in range(n + 1):
            f = start + (i % (end - start + 1) if frames else i)
            pose = self.sample(act_src, min(f, end))
            t = i / n
            world = self.solve(pose, anchor, (lambda b, m, w, t=t: override(self, b, m, w, t)) if override else None)
            self.key(world, 1 + i, prev)
        act.frame_range = (1, n + 1)
        return act

    def walk_speed(self):
        """Ground speed the walk cycle implies: how fast a planted foot slides
        backwards in this in-place clip, in the agent's scale."""
        act = bpy.data.actions["Walk_Loop"]
        start, end = (int(v) for v in act.frame_range)
        ys, zs = [], []
        for f in range(start, end + 1):
            p = self.sample(act, f)["foot_l"].translation
            ys.append(p.y)
            zs.append(p.z)
        floor = min(zs)
        fps = bpy.context.scene.render.fps
        v = [abs(ys[i + 1] - ys[i]) * fps for i in range(len(ys) - 1) if zs[i] < floor + 0.015 and zs[i + 1] < floor + 0.015]
        v.sort()
        return (v[len(v) // 2] if v else 1.3) * self.scale

    def cleanup(self):
        for o in self.imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        for a in self.src_actions:
            if a.name in bpy.data.actions:
                bpy.data.actions.remove(a)


# -- typing and waving --------------------------------------------------------------------

def _lengths(rt, side):
    r = rt.rest
    up = (r[f"forearm.{side}"].translation - r[f"upperarm.{side}"].translation).length
    lo = (r[f"hand.{side}"].translation - r[f"forearm.{side}"].translation).length
    return up, lo


def typing(rt, name, m, world, t):
    """Arms to the keyboard by IK; wrists and fingers tap on uneven rhythms."""
    two_pi = 2 * math.pi
    for side, s, ph in (("L", 1, 0.0), ("R", -1, 1.7)):
        if name == f"upperarm.{side}":
            shoulder = m.translation
            l1, l2 = _lengths(rt, side)
            # Whole-number cycles over the loop so it repeats seamlessly.
            drift = Vector((0.012 * math.sin(two_pi * t + ph), 0.01 * math.sin(two_pi * 2 * t + ph), 0.0))
            target = Vector((KEYS_X * s, KEYS_Y, KEYS_Z)) + drift
            pole = Vector((0.55 * s, 0.45, -0.7))
            elbow, wrist = two_bone(shoulder, target, l1, l2, pole)
            rt._ik = getattr(rt, "_ik", {})
            rt._ik[side] = (elbow, wrist)
            up = rot(m)
            dir0 = up @ Vector((0, 1, 0))
            r = (dir0.rotation_difference(elbow - shoulder)).to_matrix() @ up
            return Matrix.Translation(shoulder) @ r.to_4x4()
        if name == f"forearm.{side}":
            elbow, wrist = rt._ik[side]
            down = Vector((0, -0.25, -1)).normalized()
            r = frame_from(wrist - elbow, down, Vector((0, 1, 0)), rt.palm[name])
            return Matrix.Translation(m.translation) @ r.to_4x4()
        if name == f"hand.{side}":
            tap = 5 * max(0.0, math.sin(two_pi * 7 * t + ph)) + 3 * max(0.0, math.sin(two_pi * 11 * t + 2 * ph))
            ahead = Vector((0.05 * s, -1.0, -0.12)).normalized()
            r = frame_from(ahead, Vector((0, 0, -1)), Vector((0, 1, 0)), rt.palm[name])
            r = r @ axis_rot(Vector((1, 0, 0)), -tap)
            return Matrix.Translation(m.translation) @ r.to_4x4()
        for f, k in (("index", 0.0), ("middle", 0.9), ("ring", 1.8), ("pinky", 2.6), ("thumb", 3.4)):
            for i in (1, 2, 3):
                if name == f"{f}_0{i}_{side.lower()}":
                    curl = (18 if f != "thumb" else 8) + 10 * max(0.0, math.sin(two_pi * 9 * t + ph + k))
                    parent = world[rt.rig.data.bones[name].parent.name]
                    base = parent @ rt.rel[name]
                    r = rot(base) @ axis_rot(Vector((1, 0, 0)), curl * (1.0 if i == 1 else 0.8))
                    return Matrix.Translation(base.translation) @ r.to_4x4()
    if name == "head":
        nod = 1.5 * math.sin(2 * math.pi * 2 * t)
        r = axis_rot(Vector((1, 0, 0)), -(6 + nod)) @ rot(m)
        return Matrix.Translation(m.translation) @ r.to_4x4()
    return None


def walking(rt, name, m, world, t):
    """The library walk swings the hands only in front of the hips. Tilting the
    upper arms back centres the swing, so each hand travels about as far behind
    the body as in front of it."""
    if name in ("upperarm.L", "upperarm.R"):
        r = axis_rot(Vector((1, 0, 0)), WALK_ARM_BACK) @ rot(m)
        return Matrix.Translation(m.translation) @ r.to_4x4()
    return None


# Degrees the upper arms tilt back in the walk (see walking()).
WALK_ARM_BACK = 36


def waving(rt, name, m, world, t):
    """Right arm raised, forearm swinging from the elbow, palm forward."""
    side, s = "R", -1
    if name == f"upperarm.{side}":
        shoulder = m.translation
        l1, l2 = _lengths(rt, side)
        elbow = shoulder + Vector((0.55 * s, -0.12, 0.45)).normalized() * l1
        swing = math.radians(25) * math.sin(2 * math.pi * 2 * t)
        up = Vector((math.sin(swing) * s, -0.15, math.cos(swing))).normalized()
        rt._wave = (elbow, elbow + up * l2)
        r0 = rot(m)
        r = ((r0 @ Vector((0, 1, 0))).rotation_difference(elbow - shoulder)).to_matrix() @ r0
        return Matrix.Translation(shoulder) @ r.to_4x4()
    if name == f"forearm.{side}":
        elbow, wrist = rt._wave
        r = frame_from(wrist - elbow, Vector((0, -1, 0)), Vector((0, 1, 0)), rt.palm[name])
        return Matrix.Translation(m.translation) @ r.to_4x4()
    if name == f"hand.{side}":
        elbow, wrist = rt._wave
        r = frame_from(wrist - elbow, Vector((0, -1, 0)), Vector((0, 1, 0)), rt.palm[name])
        return Matrix.Translation(m.translation) @ r.to_4x4()
    return None


def animate(rig):
    rt = Retarget(rig)
    speed = rt.walk_speed()
    rt.clip("Idle", "Idle_Loop")
    rt.clip("Walk", "Walk_Loop", override=walking)
    rt.clip("Sit", "Sitting_Idle_Loop", anchored=True)
    rt.clip("Type", "Sitting_Idle_Loop", anchored=True, override=typing)
    rt.clip("Wave", "Idle_Loop", frames=30, override=waving)
    rt.cleanup()
    rig.animation_data.action = bpy.data.actions["Idle"]
    print(f"[ual] retargeted Idle, Walk, Sit, Type, Wave; walk speed {speed:.2f} m/s")
    return speed
