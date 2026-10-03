"""
Arms the agents with Quaternius' CC0 Ultimate Gun Pack.

    blender -b -P blender/guns.py            # public/models/guns.glb (desk racks)

build_humans.py calls arm(rig) to add, per role, a long gun slung across the
back (Gun_<i>, rigid on the chest) and a pistol in a thigh holster
(Sidearm_<i>, rigid on the right thigh). The web app shows only the agent's own
pair. A chair back would cut through the slung gun, so while an agent is seated
the app hides it and shows the same gun propped against the desk instead
(Rack_<i> in guns.glb, in workstation coordinates).

Needs the pack (https://quaternius.com/packs/ultimategun.html) with its FBX
folder unpacked into blender/assets/guns/FBX.
"""

import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_agents as base  # noqa: E402

PACK = os.path.join(HERE, "assets", "guns", "FBX")
OUT = os.path.join(os.path.dirname(HERE), "public", "models", "guns.glb")

# One loadout per role, in ROLES order (research, creative, comms, devices,
# code, vision): (long gun, its length in metres), (sidearm, length).
LOADOUT = [
    (("SniperRifle_1", 1.05), ("Revolver_1", 0.26)),
    (("Bullpup_1", 0.72), ("Pistol_2", 0.19)),
    (("SubmachineGun_5", 0.6), ("Pistol_1", 0.19)),
    (("Shotgun_1", 0.95), ("Pistol_3", 0.2)),
    (("AssaultRifle_4", 0.9), ("Pistol_4", 0.19)),
    (("AssaultRifle2_1", 0.88), ("Pistol_5", 0.19)),
]
SLING_TILT = 22  # degrees from vertical; muzzle up over the right shoulder


def load(name, length):
    """Import one gun as a single mesh: barrel along +X (muzzle at +X), sights
    up +Z, centred on its bounding box and scaled to `length` metres."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=os.path.join(PACK, name + ".fbx"))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in new:
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.parent = None
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    gun = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    pts = [v.co for v in gun.data.vertices]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    s = length / (hi.x - lo.x)
    gun.data.transform(Matrix.Scale(s, 4) @ Matrix.Translation(-(lo + hi) / 2))
    for p in gun.data.polygons:
        p.use_smooth = False
    # Share materials between guns and keep them clear of the human's names.
    for slot in gun.material_slots:
        m = slot.material
        if m and not m.name.startswith("Gun_"):
            key = "Gun_" + m.name.split(".")[0]
            slot.material = bpy.data.materials.get(key) or m
            if slot.material is m:
                m.name = key
    gun.name = gun.data.name = name
    return gun, (hi - lo) * s


def rigid(obj, rig, bone):
    obj.vertex_groups.clear()
    vg = obj.vertex_groups.new(name=bone)
    vg.add([v.index for v in obj.data.vertices], 1.0, "REPLACE")
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    mod = obj.modifiers.new("rig", "ARMATURE")
    mod.object = rig


def place(obj, x_axis, y_axis, at):
    """Move a gun from load()'s frame: local X -> x_axis, Y -> y_axis, centre -> at."""
    x = x_axis.normalized()
    y = (y_axis - x * y_axis.dot(x)).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = at
    obj.data.transform(m)


def surface(rig):
    """World-space vertices of everything the human wears or is."""
    return [o.matrix_world @ v.co for o in rig.children_recursive if o.type == "MESH" and not o.name.startswith(("Gun_", "Sidearm_", "Accessories")) for v in o.data.vertices]


def back_gun(rig, pts, i, name, length):
    gun, size = load(name, length)
    chest = rig.data.bones["chest"].head_local
    t = math.radians(SLING_TILT)
    d = sling_dir()
    c = Vector((0, 0, chest.z - 0.04))
    # Rest it on the back along its whole length: the furthest-back point of
    # the body within its footprint, plus half its thickness.
    foot = [p for p in pts if abs(p.x) < 0.17 and abs((p - c).dot(d)) < length / 2 and (Vector((p.x, 0, p.z)) - c - d * (p - c).dot(d)).length < 0.07]
    c.y = max(p.y for p in foot) + size.y / 2 + 0.012
    place(gun, d, Vector((0, 1, 0)), c)
    # Centre the silhouette (magazine included) on the spine, clear of both arms.
    xs = [v.co.x for v in gun.data.vertices]
    gun.data.transform(Matrix.Translation((-(min(xs) + max(xs)) / 2, 0, 0)))
    rigid(gun, rig, "chest")
    gun.name = gun.data.name = f"Gun_{i}"
    return gun, c.y - size.y / 2


def sling_dir():
    t = math.radians(SLING_TILT)
    return Vector((-math.sin(t), 0, math.cos(t)))


def strap(rig, pts, back_y, mat):
    """The sling: from the gun's muzzle end over the right shoulder, diagonally
    down across the chest, round the left hip and back to the stock."""
    chest = rig.data.bones["chest"].head_local
    centre = Vector((0, back_y, chest.z - 0.04))
    d = sling_dir()

    def on_gun(x):
        return centre + d * (x / d.x)

    def near(q, tol=0.03):
        # The torso only: the hands hang close to the hips.
        return [p for p in pts if abs(p.x - q.x) < tol and abs(p.z - q.z) < tol and p.y > -0.22]

    def front(q):
        hits = near(q)
        return Vector((q.x, (min(p.y for p in hits) if hits else -0.12) - 0.008, q.z))

    a = Vector((-0.12, 0, chest.z + 0.2))
    b = Vector((0.13, 0, chest.z - 0.17))
    over = max(p.z for p in pts if abs(p.x - a.x) < 0.03 and abs(p.y) < 0.06) + 0.008
    side_hits = [p for p in pts if 0 < p.x < 0.2 and abs(p.y) < 0.05 and abs(p.z - b.z) < 0.02]
    side = Vector(((max(p.x for p in side_hits) if side_hits else 0.17) + 0.008, 0.02, b.z - 0.01))
    line = [on_gun(-0.12) + Vector((0, 0.004, 0)), Vector((a.x, back_y + 0.01, a.z - 0.02)), Vector((a.x, 0, over))]
    line += [front(a.lerp(b, k / 8)) for k in range(9)]
    line += [side, on_gun(0.13) + Vector((0, 0.004, 0))]
    parts = [base.limb(p, q, 0.009, mat, "chest", taper=1.0) for p, q in zip(line, line[1:])]
    s = base.join(parts, "Sling")
    rigid_keep(s, rig)
    return s


def rigid_keep(obj, rig):
    """Like rigid(), for parts built by build_agents' helpers, which already
    carry their bone's vertex group."""
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    mod = obj.modifiers.new("rig", "ARMATURE")
    mod.object = rig


def sidearm(rig, pts, i, name, length, leather):
    gun, size = load(name, length)
    hip = rig.data.bones["thigh.R"].head_local
    z = hip.z - 0.17
    ring = [p for p in pts if abs(p.x - hip.x) < 0.11 and abs(p.z - z) < 0.02]
    outer = min(p.x for p in ring)
    inner = max(p.x for p in ring)
    front = min(p.y for p in ring)
    rear = max(p.y for p in ring)
    leg = Vector(((outer + inner) / 2, (front + rear) / 2, z))
    w = size.y / 2 + 0.012
    at = Vector((outer - w, leg.y + 0.01, z))
    # Muzzle down and slightly back, slide facing forward, grip to the rear.
    place(gun, Vector((0, 0.12, -1)), Vector((1, 0, 0)), at)

    # Holster: a leather shell over the front two-thirds of the gun.
    x, y = Vector((0, 0.12, -1)).normalized(), Vector((1, 0, 0))
    z_ax = x.cross(y)
    shell = base.box(Vector(), (length * 0.7, size.y + 0.018, size.z * 0.62), leather, None, bevel=0.006)
    shell.data.transform(Matrix.Translation((length * 0.17, 0, size.z * 0.19)))
    m = Matrix((x, y, z_ax)).transposed().to_4x4()
    m.translation = at
    shell.data.transform(m)
    straps = []
    r = max(inner - outer, rear - front) / 2 + 0.01
    for dz in (-0.03, -0.12):
        straps.append(base.ring(Vector((leg.x, leg.y, z + dz)), r, 0.0085, leather, None))
    kit = base.join([gun, shell, *straps], f"Sidearm_{i}")
    rigid(kit, rig, "thigh.R")
    return kit


def arm(rig):
    """Every role's loadout, all on the one rig; the app shows the agent's own."""
    pts = surface(rig)
    leather = base.material("Gun_Leather", (0.035, 0.03, 0.028), rough=0.75)
    webbing = base.material("Gun_Webbing", (0.02, 0.022, 0.025), rough=0.9)
    backs = []
    for i, ((long_name, long_len), (side_name, side_len)) in enumerate(LOADOUT):
        backs.append(back_gun(rig, pts, i, long_name, long_len)[1])
        sidearm(rig, pts, i, side_name, side_len, leather)
    strap(rig, pts, max(backs), webbing)
    print(f"[guns] armed {len(LOADOUT)} loadouts")


def rack():
    """guns.glb: each long gun propped against the desk's left leg, butt on
    the floor, in workstation coordinates (build_agents.build_workstation)."""
    base.reset()
    objs = []
    for i, ((name, length), _) in enumerate(LOADOUT):
        gun, size = load(name, length)
        lean = math.radians(8)
        up = Vector((math.sin(lean), 0, math.cos(lean)))
        # On the desk's left leg (-X; agents come in from +X to sit), butt on
        # the floor, resting on the glass top's edge (x 0.7, z 0.727), or on
        # the leg's face (x 0.685) if the gun is shorter than the desk.
        touch = min(length * math.cos(lean), 0.727)
        face = 0.7 if touch >= 0.727 else 0.685
        butt = Vector((-(face + size.y / 2 + 0.004 + touch * math.tan(lean)), 0.3, 0.004))
        centre = butt + up * (length / 2)
        place(gun, up, Vector((1, 0, 0)), centre)
        gun.name = gun.data.name = f"Rack_{i}"
        objs.append(gun)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_apply=True, export_lights=False, export_cameras=False)
    print(f"[guns] wrote {os.path.relpath(OUT, os.path.dirname(HERE))} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    rack()
