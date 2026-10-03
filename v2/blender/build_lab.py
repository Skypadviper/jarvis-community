"""
Builds the ops-floor environment: a sci-fi command hall assembled from
Quaternius's Modular Sci-Fi MegaKit (CC0), with a containment capsule in the
middle for JARVIS's brain.

    blender -b -P blender/build_lab.py                 # lab.glb + preview
    blender -b -P blender/build_lab.py -- --no-render

Needs, once per machine: the free "Modular SciFi MegaKit[Standard]" from
https://quaternius.itch.io/modular-sci-fi-megakit, its glTF models and textures
gathered into blender/assets/megakit/flat/ (git-ignored; only the built GLB ships).

Writes:
    public/models/lab.glb        the hall and the capsule, one file
    blender/renders/lab.png      preview from the ops-floor camera

Layout is authored in the web app's coordinates (Y up, the camera on +Z looking
towards -Z) because that is where everything else lives: the hub at the origin,
agents on rings of 2.7 m and 3.5 m, desks on an arc of 5 m behind the hub. The
kit snaps to a 4 m grid, so the hall is 6 x 6 cells, 24 m across, 5 m to the
top of the cable trim.

Materials the web app treats specially, by name:
    CapsuleGlass    the containment tube (made see-through in the browser)
    CapsuleGlow     light rings on the capsule (tinted with JARVIS's colour)
"""

import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(HERE, "assets", "megakit", "flat")
OUT = os.path.join(ROOT, "public", "models", "lab.glb")
RENDERS = os.path.join(HERE, "renders")
TEXTURE_SIZE = 1024

CELL = 4.0
HALF = 12.0  # half the hall's width
CELLS = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]

placed = []
library = {}


# -- coordinates -----------------------------------------------------------------

def to_blender(x, y, z):
    """Web (Y up, camera on +Z) -> Blender (Z up). A turn about web Y is the
    same angle about Blender Z."""
    return Vector((x, -z, y))


# -- the kit -----------------------------------------------------------------------

def load(name):
    """Import one kit model once; every placement shares its mesh."""
    if name in library:
        return library[name]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(KIT, f"{name}.gltf"))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    keep = src.name
    # Bake the importer's root transform into the mesh, then drop the empties
    # and any rig (the aliens come rigged; here they are statues in a tank).
    world = src.matrix_world.copy()
    for m in list(src.modifiers):
        if m.type == "ARMATURE":
            src.modifiers.remove(m)
    src.parent = None
    src.matrix_world = world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(new):
        try:
            if o.name != keep:
                bpy.data.objects.remove(o, do_unlink=True)
        except ReferenceError:
            pass
    src = bpy.data.objects[keep]
    src.name = f"LIB_{name}"
    src.hide_render = True
    src.hide_set(True)
    library[name] = src
    return src


def put(name, x, y, z, turn=0.0, scale=1.0):
    src = load(name)
    obj = src.copy()  # linked duplicate: same mesh data
    bpy.context.collection.objects.link(obj)
    obj.hide_render = False
    obj.hide_set(False)
    obj.name = name
    obj.location = to_blender(x, y, z)
    # The glTF importer leaves objects in quaternion mode, where an Euler
    # rotation is silently ignored.
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, math.radians(turn))
    obj.scale = (scale, scale, scale)
    placed.append(obj)
    return obj


# -- the hall --------------------------------------------------------------------

# Each edge: where a wall piece sits for a cell, and the turn that faces it inwards.
EDGES = {
    "west": (lambda c: (-HALF + 2, c), 0),
    "east": (lambda c: (HALF - 2, c), 180),
    "north": (lambda c: (c, -HALF + 2), -90),
    "south": (lambda c: (c, HALF - 2), 90),
}

WALLS = {
    # Behind the desks, the wall the camera looks at: windows and heavy panels.
    "north": ["WallAstra_Straight_Window", "WallAstra_Straight", "WallWindow_Straight",
              "WallWindow_Straight", "WallAstra_Straight", "WallAstra_Straight_Window"],
    "west": ["WallBand_Straight", "WallAstra_Straight_Divided", "WallAstra_Straight_Flat_Window",
             "WallAstra_Straight", "WallAstra_Straight_Divided", "WallBand_Straight"],
    "east": ["WallBand_Straight", "WallAstra_Straight_Divided", "WallAstra_Straight_Flat_Window",
             "WallAstra_Straight", "WallAstra_Straight_Divided", "WallBand_Straight"],
    "south": ["WallAstra_Straight", "WallBand_Straight", "WallAstra_Straight",
              "WallAstra_Straight", "WallBand_Straight", "WallAstra_Straight"],
}
TOPS = ["TopCables_Straight", "TopCables_Straight_Hanging", "TopWindow_Straight",
        "TopWindow_Straight", "TopCables_Straight_Hanging", "TopCables_Straight"]


def hall():
    for edge, (where, turn) in EDGES.items():
        for i, c in enumerate(CELLS):
            x, z = where(c)
            put(WALLS[edge][i], x, 0, z, turn)
            put(TOPS[i], x, 0, z, turn)
            put("BottomMetal_Straight", x, 0, z, turn)
        # A column at every seam between wall cells.
        for s in (-8.0, -4.0, 0.0, 4.0, 8.0):
            x, z = where(s)
            put("Column_Astra", x - (2 if edge == "west" else -2 if edge == "east" else 0),
                0, z - (2 if edge == "north" else -2 if edge == "south" else 0), turn)
    # Tall pillars in the corners.
    for x in (-HALF + 0.7, HALF - 0.7):
        for z in (-HALF + 0.9, HALF - 0.9):
            put("Column_Large_Straight", x, 0, z, 0 if x < 0 else 180)


def floor():
    for x in CELLS:
        for z in CELLS:
            ring = max(abs(x), abs(z))
            if ring <= 2:
                name = "Platform_CenterPlate"
            elif ring <= 6:
                name = "Platform_DarkPlates"
            else:
                name = "Platform_Metal"
            put(name, x, 0, z, 90 * ((int(x) + int(z)) // 4 % 4))
    # A round dais under the capsule.
    put("Platform_Round1", 0, 0, 0, 0, scale=0.56)
    # Floor markings: a sweeping ring of lines and the logo in front of the hub.
    for q in range(4):
        put("Decal_Line_90_Round_Large", 0, 0.012, 0, 90 * q, scale=2.2)
    put("Decal_Logo", 0, 0.013, 6.4, 0, scale=1.6)


def stations():
    """Computer banks along the walls, access panels, lights and vents."""
    # Along the back wall, both sides of the windows.
    for x in [-10.9, -10.1, -9.3, -6.9, -6.1, -5.3, 5.3, 6.1, 6.9, 9.3, 10.1, 10.9]:
        put("Prop_Computer", x, 0, -HALF + 0.9, 0)
    # Down the side walls.
    for z in [-8.9, -8.1, -7.3, -0.9, -0.1, 0.7, 5.3, 6.1, 6.9]:
        put("Prop_Computer", -HALF + 0.9, 0, z, 90)
        put("Prop_Computer", HALF - 0.9, 0, z, -90)
    # Access panels on the walls between the computer banks.
    for z in (-4.0, 3.0):
        put("Prop_AccessPoint", -HALF + 2.0, 0, z, 0)
        put("Prop_AccessPoint", HALF - 2.0, 0, z, 180)
    for x in (-2.5, 2.5):
        put("Prop_AccessPoint", x, 0, -HALF + 2.0, -90)
    # Vents and fans up high.
    for x in (-8.0, 0.0, 8.0):
        put("Prop_Vent_Big", x, 3.6, -HALF + 0.4, 0)
    for z in (-6.0, 2.0, 8.0):
        put("Prop_Vent_Wide", -HALF + 0.4, 3.6, z, 90)
        put("Prop_Vent_Wide", HALF - 0.4, 3.6, z, -90)
    # Cables slung along the tops of the side walls.
    for z in (-6.0, 6.0):
        put("Prop_Cable_3", -HALF + 0.6, 4.2, z, 0)
        put("Prop_Cable_3", HALF - 0.6, 4.2, z, 180)
    # Floor lights ringing the dais, inside the agents' lounge ring.
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        put("Prop_Light_Floor", math.sin(a) * 1.95, 0, math.cos(a) * 1.95, math.degrees(a) + 90, scale=0.7)
    # Cargo in the front corners, out of the camera's way.
    for x, z, t in ((-10.5, 10.4, 10), (-9.4, 10.6, -15), (10.2, 10.3, 30)):
        put("Prop_Crate4", x, 0, z, t)
    put("Prop_Crate3", -10.4, 1.0, 10.4, 25)
    for x, z in ((9.2, 10.6), (9.8, 9.6), (10.6, 10.7)):
        put("Prop_Barrel_Large", x, 0, z, 0)


# -- the capsule ----------------------------------------------------------------------

def mat(name, color, emit=None, strength=0.0, metallic=0.0, rough=0.5, alpha=1.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


def cyl(name, r, h, y0, m, verts=64, caps=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=to_blender(0, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.location = to_blender(0, y0 + h / 2, 0)
    if not caps:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for p in o.data.polygons:
            p.select = len(p.vertices) > 4
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")
    o.data.materials.append(m)
    for p in o.data.polygons:
        p.use_smooth = True
    placed.append(o)
    return o


def torus(name, r, minor, y, m):
    bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=minor, major_segments=96, minor_segments=12,
                                     location=to_blender(0, y, 0))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(m)
    for p in o.data.polygons:
        p.use_smooth = True
    placed.append(o)
    return o


def capsule(scale=1.0, at=(0.0, 0.0), prefix="Capsule"):
    """A containment tube: armoured base, glass cylinder, cap, struts, cables.
    The brain floats in the main one at 2.05 m."""
    metal = mat("CapsuleMetal", (0.05, 0.06, 0.07), metallic=0.85, rough=0.32)
    trim = mat("CapsuleTrim", (0.45, 0.48, 0.52), metallic=0.9, rough=0.25)
    glow = mat("CapsuleGlow", (0.0, 0.6, 0.7), emit=(0.0, 0.9, 1.0), strength=8.0)
    glass = mat("CapsuleGlass", (0.6, 0.9, 1.0), metallic=0.0, rough=0.05, alpha=0.15)

    s = scale
    parts = [
        cyl(f"{prefix}Base", 1.45 * s, 0.25 * s, 0.0, metal),
        cyl(f"{prefix}Plinth", 1.2 * s, 0.55 * s, 0.25 * s, trim),
        torus(f"{prefix}RingLow", 1.21 * s, 0.03 * s, 0.8 * s, glow),
        cyl(f"{prefix}Floor", 1.02 * s, 0.05 * s, 0.8 * s, glow),
        cyl(f"{prefix}Glass", 1.05 * s, 2.45 * s, 0.85 * s, glass, caps=False),
        torus(f"{prefix}RingHigh", 1.08 * s, 0.025 * s, 3.3 * s, glow),
        cyl(f"{prefix}Cap", 1.25 * s, 0.35 * s, 3.3 * s, metal),
        cyl(f"{prefix}CapTrim", 0.8 * s, 0.25 * s, 3.65 * s, trim),
    ]
    # Struts round the glass.
    for k in range(6):
        a = math.radians(30 + 60 * k)
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.05 * s, depth=2.5 * s)
        o = bpy.context.active_object
        o.name = f"{prefix}Strut"
        o.location = to_blender(math.sin(a) * 1.13 * s, 2.05 * s, math.cos(a) * 1.13 * s)
        o.data.materials.append(trim)
        placed.append(o)
        parts.append(o)
    # Feed cables up into the ceiling dark.
    for k in range(4):
        a = math.radians(45 + 90 * k)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.07 * s, depth=2.2 * s)
        o = bpy.context.active_object
        o.name = f"{prefix}Feed"
        o.location = to_blender(math.sin(a) * 0.55 * s, 3.9 * s + 1.1 * s, math.cos(a) * 0.55 * s)
        o.data.materials.append(metal)
        placed.append(o)
        parts.append(o)
    for o in parts:
        o.location += to_blender(at[0], 0, at[1]) - to_blender(0, 0, 0)
    return parts


def specimens():
    """Two smaller tanks on the side walls, each holding one of the kit's aliens."""
    for x, alien in ((-HALF + 2.4, "Alien_Oculichrysalis"), (HALF - 2.4, "Alien_Cyclop")):
        capsule(scale=0.45, at=(x, -2.5), prefix=f"Tank{'W' if x < 0 else 'E'}")
        put(alien, x, 0.45 * 2.05, -2.5, 90 if x < 0 else -90, scale=0.55)


# -- output ----------------------------------------------------------------------------

def shrink_textures():
    for img in bpy.data.images:
        if img.size[0] > TEXTURE_SIZE or img.size[1] > TEXTURE_SIZE:
            img.scale(TEXTURE_SIZE, TEXTURE_SIZE)


def export():
    bpy.ops.object.select_all(action="DESELECT")
    for o in placed:
        o.select_set(True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=OUT,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_lights=False,
        export_cameras=False,
        export_animations=False,
    )
    print(f"[lab] {len(placed)} objects -> {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) // 1024} KB)")


def preview():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    world = bpy.data.worlds.new("World") if not scene.world else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.012, 0.02, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    for loc, energy in (((0, 0, 4.6), 900), ((-7, -6, 4.5), 500), ((7, -6, 4.5), 500), ((0, 7, 4.5), 400)):
        bpy.ops.object.light_add(type="POINT", location=loc)
        light = bpy.context.active_object
        light.data.energy = energy
        light.data.color = (0.6, 0.9, 1.0)
        light.data.shadow_soft_size = 1.0
    # The ops-floor camera: (0, 5.2, 8.6) looking at (0, 0.9, -1.2), 42 degrees.
    bpy.ops.object.camera_add(location=to_blender(0, 5.2, 8.6))
    cam = bpy.context.active_object
    cam.rotation_euler = (to_blender(0, 0.9, -1.2) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.angle = math.radians(56)
    scene.camera = cam
    scene.render.filepath = os.path.join(RENDERS, "lab.png")
    os.makedirs(RENDERS, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print("[lab] rendered blender/renders/lab.png")


def top_view():
    """Debug: an orthographic plan of the whole hall, to check the layout."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = 900
    bpy.ops.object.camera_add(location=(0, 0, 40))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 27
    scene.camera = cam
    scene.render.filepath = os.path.join(RENDERS, "lab_top.png")
    bpy.ops.render.render(write_still=True)
    print("[lab] rendered blender/renders/lab_top.png (north is up)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--top" in argv:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        hall()
        floor()
        stations()
        capsule()
        specimens()
        top_view()
        return
    if not os.path.isdir(KIT):
        sys.exit(f"MegaKit not found in {KIT}. See the note at the top of this file.")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    hall()
    floor()
    stations()
    capsule()
    specimens()
    shrink_textures()
    export()
    if "--no-render" not in argv:
        preview()


main()
