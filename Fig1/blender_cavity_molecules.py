import math
import random

import bpy
from mathutils import Vector


# ─── user controls ─────────────────────────────────────────────────────────────
SEED              = 7
MOLECULE_COUNT    = 55
MIRROR_SEPARATION = 6.4
MIRROR_RADIUS     = 2.15
MIRROR_THICKNESS  = 0.13
CAVITY_RADIUS     = 2.65
CAVITY_LENGTH     = 6.4
BACKGROUND_COLOR  = (1.0, 1.0, 1.0, 1.0)
RENDER_ENGINE     = "CYCLES"
MOLECULE_SCALE_RANGE = (0.38, 0.52)
WATER_BOND_LENGTH    = 0.44
WATER_BOND_ANGLE_DEG = 104.5
BOND_RADIUS          = 0.032


ATOM_STYLE = {
    "O": {"radius": 0.20,  "color": (0.95, 0.18, 0.12, 1.0)},   # clear pink-red
    "H": {"radius": 0.082, "color": (0.92, 0.93, 0.96, 1.0)},   # near-white, cool tint
}


# ─── scene helpers ──────────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for db in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
               bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for block in list(db):
            if block.users == 0:
                db.remove(block)


def set_socket(node, names, value):
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            try:
                sock.default_value = value
            except (TypeError, ValueError):
                continue
            return


def create_pbr_material(
    name, base_color, metallic=0.0, roughness=0.35,
    transmission=0.0, alpha=1.0, emission_strength=0.0,
    specular=0.5, coat=0.0, coat_roughness=0.03, anisotropic=0.0,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_socket(bsdf, ["Base Color"],                            base_color)
    set_socket(bsdf, ["Metallic"],                              metallic)
    set_socket(bsdf, ["Roughness"],                             roughness)
    set_socket(bsdf, ["Transmission", "Transmission Weight"],   transmission)
    set_socket(bsdf, ["Alpha"],                                 alpha)
    set_socket(bsdf, ["Emission Color", "Emission"],            base_color)
    set_socket(bsdf, ["Emission Strength"],                     emission_strength)
    set_socket(bsdf, ["Specular IOR Level", "Specular"],        specular)
    set_socket(bsdf, ["Coat Weight", "Clearcoat"],              coat)
    set_socket(bsdf, ["Coat Roughness", "Clearcoat Roughness"], coat_roughness)
    set_socket(bsdf, ["Anisotropic"],                           anisotropic)
    mat.blend_method = "BLEND" if (alpha < 1.0 or transmission > 0.0) else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = transmission > 0.0
    return mat


def make_materials():
    mats = {
        "bond": create_pbr_material(
            "Bond", (0.50, 0.52, 0.56, 1.0),
            roughness=0.26, specular=0.30, coat=0.04, coat_roughness=0.02,
        ),
        # Polished gold — roughness 0.14, anisotropic edge highlights
        "mirror": create_pbr_material(
            "MirrorGold", (0.831, 0.686, 0.215, 1.0),
            metallic=1.0, roughness=0.14, specular=0.55,
            coat=0.02, coat_roughness=0.04, anisotropic=0.12,
        ),
    }
    mats["O"] = create_pbr_material(
        "Atom_O", ATOM_STYLE["O"]["color"],
        roughness=0.12, specular=0.52, coat=0.12, coat_roughness=0.04,
    )
    mats["H"] = create_pbr_material(
        "Atom_H", ATOM_STYLE["H"]["color"],
        roughness=0.12, specular=0.55, coat=0.08, coat_roughness=0.03,
    )
    return mats


# ─── geometry primitives ────────────────────────────────────────────────────────

def add_uv_sphere(name, radius, location, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=location, segments=56, ring_count=32)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(material)
    return obj


def add_cylinder_between(name, start, end, radius, material):
    sv, ev = Vector(start), Vector(end)
    delta  = ev - sv
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=delta.length,
        location=(sv + ev) * 0.5, vertices=24,
    )
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(material)
    obj.rotation_euler = Vector((0, 0, 1)).rotation_difference(delta.normalized()).to_euler()
    return obj


def water_template():
    half = math.radians(WATER_BOND_ANGLE_DEG * 0.5)
    x = WATER_BOND_LENGTH * math.sin(half)
    y = WATER_BOND_LENGTH * math.cos(half)
    return [("O", (0., 0., 0.)), ("H", (-x, y, 0.)), ("H", (x, y, 0.))], [(0, 1), (0, 2)]


def add_molecule(name, atoms, bonds, location, rotation, scale, materials):
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    root.location       = location
    root.rotation_euler = rotation
    root.scale          = (scale, scale, scale)
    for idx, (elem, pos) in enumerate(atoms):
        atom = add_uv_sphere(
            f"{name}_atom_{idx}", ATOM_STYLE[elem]["radius"], pos, materials[elem])
        atom.parent = root
    for idx, (a, b) in enumerate(bonds):
        bond = add_cylinder_between(
            f"{name}_bond_{idx}", atoms[a][1], atoms[b][1],
            radius=BOND_RADIUS, material=materials["bond"])
        bond.parent = root
    return root


# ─── mirrors ────────────────────────────────────────────────────────────────────

def add_mirror(x_location, materials):
    """Thin beveled gold plate, strictly perpendicular to the cavity X-axis."""
    bpy.ops.mesh.primitive_cube_add(location=(x_location, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = "MirrorRight" if x_location > 0 else "MirrorLeft"
    obj.scale = (MIRROR_THICKNESS * 0.5, MIRROR_RADIUS * 0.52, MIRROR_RADIUS)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(scale=True)
    bev          = obj.modifiers.new("Bevel", "BEVEL")
    bev.width    = 0.035
    bev.segments = 5
    for poly in obj.data.polygons:
        poly.use_smooth = True
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(materials["mirror"])


# ─── cavity field: luminous ring + barely-perceptible haze ─────────────────────

def _make_vol_mat(name, color, density, emit_strength):
    """Uniform volume material — used only for the residual field haze."""
    mat   = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    pvol = nodes.new("ShaderNodeVolumePrincipled")
    pvol.inputs["Color"].default_value = (*color[:3], 1.0)
    ec = pvol.inputs.get("Emission Color") or pvol.inputs.get("Emission")
    if ec:
        ec.default_value = (*color[:3], 1.0)
    es = pvol.inputs.get("Emission Strength")
    if es:
        es.default_value = emit_strength
    pvol.inputs["Density"].default_value = density
    links.new(pvol.outputs["Volume"], out.inputs["Volume"])
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    return mat


def add_field_haze():
    """Barely-perceptible residual field atmosphere — just enough to suggest
    that the cavity interior has a slightly luminous quality without obscuring
    the molecules or competing with the ring."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius    = CAVITY_RADIUS * 0.58,
        depth     = CAVITY_LENGTH * 0.88,
        vertices  = 64,
        end_fill_type = "NGON",
        location  = (0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name           = "FieldHaze"
    obj.rotation_euler = (0.0, math.radians(90.0), 0.0)
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(
        _make_vol_mat("FieldHazeMat",
                      color=(0.25, 0.68, 1.0, 1.0),
                      density=0.0025,
                      emit_strength=0.025))
    obj.display_type = "WIRE"


# ─── molecule placement ─────────────────────────────────────────────────────────

def arranged_water_positions(count, rng):
    """Uniform-sphere packing with mild y-compression for 3-D visual depth.
    Some molecules fall inside the ring oval, others outside, giving depth cues."""
    cluster_radius = 2.05
    min_spacing    = 0.43
    y_compress     = 0.55
    positions      = []

    for _ in range(count * 80):
        if len(positions) >= count:
            break
        r     = cluster_radius * (rng.random() ** (1.0 / 3.0))
        cos_t = 2.0 * rng.random() - 1.0
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
        phi   = rng.uniform(0.0, math.tau)
        x     = r * sin_t * math.cos(phi)
        y     = r * sin_t * math.sin(phi) * y_compress
        z     = r * cos_t
        cand  = Vector((x, y, z))
        if all((cand - p).length > min_spacing for p in positions):
            positions.append(cand)
    return positions


def populate_cavity(materials):
    rng = random.Random(SEED)
    positions = arranged_water_positions(MOLECULE_COUNT, rng)
    atoms_tmpl, bonds_tmpl = water_template()

    for index, loc in enumerate(positions):
        rotation = (
            math.radians(90.0) + rng.uniform(math.radians(-35), math.radians(35)),
            rng.uniform(math.radians(-45), math.radians(45)),
            rng.uniform(0.0, math.tau),
        )
        scale = rng.uniform(*MOLECULE_SCALE_RANGE)
        mol = add_molecule(
            f"Molecule_{index:02d}", atoms_tmpl, bonds_tmpl,
            location=loc, rotation=rotation, scale=scale, materials=materials,
        )
        mol.rotation_euler.rotate_axis("Y", rng.uniform(-0.22, 0.22))


# ─── world / render ──────────────────────────────────────────────────────────────

def configure_world():
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out         = nodes.new(type="ShaderNodeOutputWorld")
    camera_bg   = nodes.new(type="ShaderNodeBackground")
    lighting_bg = nodes.new(type="ShaderNodeBackground")
    lpath       = nodes.new(type="ShaderNodeLightPath")
    mix         = nodes.new(type="ShaderNodeMixShader")
    camera_bg.inputs["Color"].default_value      = BACKGROUND_COLOR
    camera_bg.inputs["Strength"].default_value   = 1.0
    lighting_bg.inputs["Color"].default_value    = BACKGROUND_COLOR
    lighting_bg.inputs["Strength"].default_value = 0.0
    links.new(lpath.outputs["Is Camera Ray"],    mix.inputs["Fac"])
    links.new(lighting_bg.outputs["Background"], mix.inputs[1])
    links.new(camera_bg.outputs["Background"],   mix.inputs[2])
    links.new(mix.outputs["Shader"],             out.inputs["Surface"])
    if hasattr(world, "color"):
        world.color = (1.0, 1.0, 1.0)


def configure_render():
    scene = bpy.context.scene
    scene.render.engine           = RENDER_ENGINE
    scene.render.resolution_x     = 1600
    scene.render.resolution_y     = 900
    scene.render.film_transparent  = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.use_nodes = False
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.05
    scene.view_settings.gamma    = 1.0
    if hasattr(scene.display_settings, "display_device"):
        scene.display_settings.display_device = "sRGB"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples                 = 400
        scene.cycles.use_denoising           = True
        scene.cycles.max_bounces             = 10
        scene.cycles.transparent_max_bounces = 10
        scene.cycles.volume_bounces          = 2   # only faint haze remains


# ─── lighting ──────────────────────────────────────────────────────────────────

def add_lighting():
    R = math.radians

    def area(name, loc, energy, size, size_y, color, rot):
        bpy.ops.object.light_add(type="AREA", location=loc)
        obj = bpy.context.active_object
        obj.name           = name
        obj.data.energy    = energy
        obj.data.shape     = "RECTANGLE"
        obj.data.size      = size
        obj.data.size_y    = size_y
        obj.data.color     = color
        obj.rotation_euler = rot

    # Key: upper-left-front — soft warm main illumination
    area("KeyLight",      (-3.5, -6.0,  6.0),  850, 3.5, 2.8,
         (1.00, 0.980, 0.94), (R(50),  R(-14), R(-18)))

    # Cool fill from right — cross-illumination for molecule depth
    area("FillRight",     ( 5.0, -3.0,  1.0),  140, 5.5, 4.0,
         (0.88, 0.93, 1.00), (R(78),   R(12),  R( 30)))

    # Warm-neutral fill from left — balances FillRight
    area("FillLeft",      (-5.0, -3.0,  1.0),   85, 5.5, 4.0,
         (0.96, 0.97, 1.00), (R(78),  R(-12),  R(-30)))

    # Rim: behind-top, cool — separates molecules from background
    area("RimLight",      ( 0.0,  6.0,  3.5),  210, 7.0, 3.0,
         (0.82, 0.90, 1.00), (R(-72),  0.0,   R(180)))

    # Under-rim: lifts cluster from below for 3-D separation
    area("MoleculeRim",   ( 0.0,  4.0, -2.5),   65, 5.0, 3.0,
         (0.90, 0.92, 1.00), (R(-130), 0.0,   R(180)))

    # Ring accent: warm area light just in front of the ring, slightly camera-side.
    # Gives the luminous ring a soft warm atmosphere on the molecules nearest to it
    # and creates a gentle gold glow in the cavity interior.
    area("RingAccent",    ( 0.0, -2.8,  0.0),  100, 3.0, 3.0,
         (1.00, 0.97, 0.86), (R(90),   0.0,    0.0))

    # Mirror inner-face accents (back-positioned for specular toward camera)
    area("MirrorAccRight",( 3.5,  8.5,  0.0),  160, 2.5, 8.0,
         (1.00, 0.96, 0.78), (R(-20),  R(14),   0.0))
    area("MirrorAccLeft", (-2.0,  9.0,  0.0),  160, 2.5, 8.0,
         (1.00, 0.96, 0.78), (R(-20), R(-14),   0.0))

    # Narrow overhead strip — thin gold highlight on mirror top edges
    area("MirrorTopStrip",( 0.0, -2.8,  8.0),  170, 1.0, 12.0,
         (1.00, 0.98, 0.94), (R(22),   0.0,    0.0))


# ─── camera ───────────────────────────────────────────────────────────────────

def add_camera():
    R = math.radians
    # Looking along approximately +Y from (0.5, -15.8, 0.7):
    #   - Z rotation = 0  → no roll; perfectly level frame
    #   - X rotation 88°  → only 2° above horizontal; cavity axis reads horizontal
    #   - Y rotation -1.5° → slight off-axis to show mirror thickness/inner faces
    bpy.ops.object.camera_add(
        location=(0.5, -15.8, 0.7),
        rotation=(R(88), R(-1.5), R(0)),
    )
    cam = bpy.context.active_object
    cam.data.type         = "PERSP"
    cam.data.lens         = 65
    cam.data.sensor_width = 36
    # Gentle depth of field: focus on cluster centre, outer molecules softly blur
    cam.data.dof.use_dof         = True
    cam.data.dof.focus_distance  = 15.8
    cam.data.dof.aperture_fstop  = 6.3
    cam.data.dof.aperture_blades = 6
    bpy.context.scene.camera = cam


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    clear_scene()
    configure_world()
    configure_render()
    materials = make_materials()

    add_field_haze()           # barely-visible residual field atmosphere (rendered first)
    add_mirror(-(MIRROR_SEPARATION * 0.5), materials)
    add_mirror(  MIRROR_SEPARATION * 0.5,  materials)
    populate_cavity(materials)
    add_lighting()
    add_camera()


if __name__ == "__main__":
    main()
