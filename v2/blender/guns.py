"""
Arms the agents with Quaternius' CC0 Ultimate Gun Pack.

    blender -b -P blender/guns.py            # public/models/guns.glb (desk racks)

build_humans.py calls arm(rig) to holster a pistol on every agent's right thigh
(Sidearm_<i>), then, once the clips exist, hold() to put each role's long gun
in the right hand (Gun_<i>), where retarget_ual's carry holds it two-handed at
low ready. The web app shows only the agent's own pair. When an agent sits down
to work the app hides the gun in its hands and shows the same gun propped
against the desk (Rack_<i> in guns.glb, in workstation coordinates).

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
# Where the pistol grip (or a shotgun's wrist) sits, as a fraction of the
# length from the butt; read off each gun's underside profile.
GRIP = {"SniperRifle_1": 0.22, "Bullpup_1": 0.42, "SubmachineGun_5": 0.34,
        "Shotgun_1": 0.26, "AssaultRifle_4": 0.26, "AssaultRifle2_1": 0.26}
# How far ahead of the grip the support hand holds the handguard.
FORE_REACH = 0.23


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
                solid(m)
    gun.name = gun.data.name = name
    return gun, (hi - lo) * s


def solid(m):
    """The pack's FBX materials import with alpha 0, which glTF exports as a
    fully cut-out mask: invisible guns. Make them opaque, and let the metal
    catch the lab's light."""
    m.blend_method = "OPAQUE"
    bsdf = next((n for n in (m.node_tree.nodes if m.use_nodes else []) if n.type == "BSDF_PRINCIPLED"), None)
    if not bsdf:
        return
    alpha = bsdf.inputs["Alpha"]
    for link in list(alpha.links):
        m.node_tree.links.remove(link)
    alpha.default_value = 1.0
    col = bsdf.inputs["Base Color"]
    if not col.is_linked:
        c = col.default_value
        col.default_value = (c[0], c[1], c[2], 1.0)
    metal = "Metal" in m.name or m.name in ("Gun_Black", "Gun_Black2", "Gun_Grey", "Gun_Main", "Gun_MainDark", "Gun_MainLight")
    bsdf.inputs["Metallic"].default_value = 0.65 if metal else 0.0
    bsdf.inputs["Roughness"].default_value = 0.38 if metal else 0.6


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


def gripped(name, length):
    """load(), then moved so the firing hand's grip is the origin. Returns the
    gun and the support hand's point under the handguard, both in that frame."""
    gun, size = load(name, length)
    vs = [v.co for v in gun.data.vertices]
    lo = min(v.x for v in vs)

    def column(x, w=0.02):
        zs = [v.z for v in vs if abs(v.x - x) < w]
        return (min(zs), max(zs)) if zs else (0.0, 0.0)

    gx = lo + GRIP[name] * length
    z0, z1 = column(gx)
    grip = Vector((gx, 0, z0 + 0.4 * (z1 - z0)))
    fx = min(gx + FORE_REACH, lo + length - 0.04)
    fore = Vector((fx, 0, column(fx)[0] - 0.02)) - grip
    gun.data.transform(Matrix.Translation(-grip))
    return gun, fore


def fore_offset():
    """The support hand's point relative to the grip, averaged over the
    loadout: one animation serves every gun."""
    total = Vector()
    for (name, length), _ in LOADOUT:
        gun, fore = gripped(name, length)
        total += fore
        bpy.data.objects.remove(gun, do_unlink=True)
    return total / len(LOADOUT)


def hold(rig, carry, hand):
    """Put every long gun in the right hand: `carry` is the gun's world pose
    (grip at its origin) at the moment the hand bone's world matrix is `hand`;
    the gun is bound rigidly to the hand so it follows it in every clip."""
    rest = rig.matrix_world @ rig.data.bones["hand.R"].matrix_local
    m = rest @ hand.inverted() @ carry
    for i, ((name, length), _) in enumerate(LOADOUT):
        gun, _ = gripped(name, length)
        gun.data.transform(m)
        rigid(gun, rig, "hand.R")
        gun.name = gun.data.name = f"Gun_{i}"
    print(f"[guns] {len(LOADOUT)} long guns in hand")


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
    """Every role's holstered sidearm, all on the one rig; the app shows the
    agent's own. The long guns go in the hands after animating (hold())."""
    pts = surface(rig)
    leather = base.material("Gun_Leather", (0.035, 0.03, 0.028), rough=0.75)
    for i, (_, (side_name, side_len)) in enumerate(LOADOUT):
        sidearm(rig, pts, i, side_name, side_len, leather)
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
