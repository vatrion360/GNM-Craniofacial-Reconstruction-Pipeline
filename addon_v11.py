"""
GNM Markeri Craniofaciali (V11)
Eliminat Maxillary Notch si protectia restrictiva de lateralitate (Stanga/Dreapta).
"""

import csv
import bpy
import bmesh
from bpy.props import StringProperty, FloatProperty, IntProperty, PointerProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator, UIList
from bpy_extras import view3d_utils

bl_info = {
    "name": "GNM Markeri Stiintifici",
    "author": "VATRION",
    "version": (11, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > GNM Markeri",
    "category": "3D View",
}

# -----------------------------------------------------------------------
# BAZA DE DATE MARKERI
# -----------------------------------------------------------------------
LANDMARKS = [
    (12319, "Nasion", 6.0, 0, True), (12296, "Rhinion", 9.0, 0, True),
    (12337, "Glabella", 5.0, 0, True), (12284, "Pogonion", 10.0, 0, True),
    (12258, "Gnathion", 12.0, 0, True), (11165, "Gonion_Dr", 13.0, -1, True),
    (5037,  "Gonion_St", 13.0, 1, True), (7426,  "Orbita_Dr_Ext", 10.0, -1, True),
    (1298,  "Orbita_St_Ext", 10.0, 1, True), (11027, "Orbita_Dr_Int", 5.0, -1, True),
    (4899,  "Orbita_St_Int", 5.0, 1, True), (7566,  "Supraorbitale_Dr", 10.5, -1, True),
    (1438,  "Supraorbitale_St", 10.5, 1, True), (9903,  "Infraorbitale_Dr", 5.5, -1, True),
    (3775,  "Infraorbitale_St", 5.5, 1, True), (10603, "Zygion_Dr", 8.5, -1, True),
    (4475,  "Zygion_St", 8.5, 1, True), (9901,  "Alare_Dr", 4.5, -1, True),
    (3773,  "Alare_St", 4.5, 1, True), (8565,  "Eurion_Dr", 6.0, -1, True),
    (2437,  "Eurion_St", 6.0, 1, True), (12398, "Vertex_VarfCap", 5.5, 0, True),
    (33, "Nasospinale_BazaNas", 11.0, 0, False), (51, "Prosthion_BuzaSup", 12.0, 0, False),
]

def _encode_index(v_id: int, is_exact: bool) -> int:
    return -(v_id + 1) if is_exact else v_id

# -----------------------------------------------------------------------
# PROPRIETĂȚI
# -----------------------------------------------------------------------
class GNMSettings(PropertyGroup):
    marker_size_mm: FloatProperty(name="Raza Markeri (mm)", default=1.5, min=0.1)
    peg_thickness_mm: FloatProperty(name="Grosime Băț (mm)", default=0.5, min=0.1)

class GNMMarkerItem(PropertyGroup):
    gnm_index: IntProperty()
    label: StringProperty()
    tissue_depth_mm: FloatProperty(default=5.0, min=0.0)
    side: IntProperty(default=0)
    bone_empty: PointerProperty(type=bpy.types.Object)
    target_empty: PointerProperty(type=bpy.types.Object)
    peg_object: PointerProperty(type=bpy.types.Object)
    
    @property
    def is_placed(self) -> bool:
        return self.bone_empty is not None and self.target_empty is not None

# -----------------------------------------------------------------------
# OPERATORI
# -----------------------------------------------------------------------
class GNM_OT_import_setup(Operator):
    bl_idname = "gnm.import_setup"
    bl_label = "1. Importa & Calibreaza Craniul (.stl/.obj)"
    bl_options = {"REGISTER", "UNDO"}
    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        scene.unit_settings.system = 'METRIC'
        scene.unit_settings.scale_length = 0.001
        scene.unit_settings.length_unit = 'MILLIMETERS'

        ext = self.filepath.lower().split('.')[-1]
        try:
            if ext == 'stl':
                if hasattr(bpy.ops.wm, "stl_import"): bpy.ops.wm.stl_import(filepath=self.filepath)
                else: bpy.ops.import_mesh.stl(filepath=self.filepath)
            elif ext == 'obj':
                if hasattr(bpy.ops.wm, "obj_import"): bpy.ops.wm.obj_import(filepath=self.filepath)
                else: bpy.ops.import_scene.obj(filepath=self.filepath, split_mode='OFF')
            else:
                self.report({"ERROR"}, "Format neacceptat!")
                return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Eroare import: {e}")
            return {"CANCELLED"}
            
        obj = context.selected_objects[0]
        context.view_layer.objects.active = obj

        # Aliniere si Scalare
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0, 0, 0)
        
        max_dim = max(obj.dimensions)
        if max_dim < 5.0: 
            obj.scale = (1000.0, 1000.0, 1000.0)
            
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # CORECTAREA NORMALELOR
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        self.report({"INFO"}, "Craniul a fost importat, scalat si normalele corectate!")
        return {"FINISHED"}

class GNM_OT_init_markers(Operator):
    bl_idname = "gnm.init_markers"
    bl_label = "2. Incarca Lista Markeri"

    def execute(self, context):
        scene = context.scene
        scene.gnm_markers.clear()
        
        for v_id, lbl, depth, side, is_exact in LANDMARKS:
            item = scene.gnm_markers.add()
            item.gnm_index = _encode_index(v_id, is_exact)
            item.label = lbl
            item.tissue_depth_mm = depth
            item.side = side
            
        scene.gnm_marker_active_index = 0
        return {"FINISHED"}

class GNM_OT_place_marker(Operator):
    bl_idname = "gnm.place_marker"
    bl_label = "Plaseaza Marker"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        if not context.scene.gnm_markers: return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE": return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            return self._place(context, event)
        if event.type in {"RIGHTMOUSE", "ESC"}:
            return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def _place(self, context, event):
        scene = context.scene
        idx = scene.gnm_marker_active_index
        item = scene.gnm_markers[idx]

        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        depsgraph = context.evaluated_depsgraph_get()
        result, location, normal, _, _, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir)
        
        if not result: return {"RUNNING_MODAL"}

        depth = item.tissue_depth_mm
        target_location = location + normal.normalized() * depth
        m_size = scene.gnm_settings.marker_size_mm
        p_thick = scene.gnm_settings.peg_thickness_mm

        for old_obj in [item.bone_empty, item.target_empty, item.peg_object]:
            if old_obj: bpy.data.objects.remove(old_obj, do_unlink=True)
        
        bone = bpy.data.objects.new(f"GNM_OS_{item.label}", None)
        bone.empty_display_type = 'SPHERE'
        bone.empty_display_size = m_size
        bone.location = location
        context.collection.objects.link(bone)
        item.bone_empty = bone

        tinta = bpy.data.objects.new(f"GNM_PIELE_{item.label}", None)
        tinta.empty_display_type = 'SPHERE'
        tinta.empty_display_size = m_size
        tinta.location = target_location
        context.collection.objects.link(tinta)
        item.target_empty = tinta

        direction = target_location - location
        mesh = bpy.data.meshes.new("Peg")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12, radius1=1.0, radius2=1.0, depth=1.0)
        bm.to_mesh(mesh)
        bm.free()
        
        peg = bpy.data.objects.new(f"GNM_BAT_{item.label}", mesh)
        peg.location = (location + target_location) / 2
        peg.rotation_mode = "QUATERNION"
        peg.rotation_quaternion = direction.to_track_quat("Z", "Y")
        peg.scale = (p_thick, p_thick, direction.length)
        context.collection.objects.link(peg)
        item.peg_object = peg

        scene.gnm_marker_active_index = (idx + 1) % len(scene.gnm_markers)
        return {"FINISHED"}

class GNM_OT_export_csv(Operator):
    bl_idname = "gnm.export_csv"
    bl_label = "3. Exporta CSV Final"
    filepath: StringProperty(subtype="FILE_PATH", default="markeri_gnm_v11.csv")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        rows = []
        for item in context.scene.gnm_markers:
            if item.target_empty:
                loc = item.target_empty.matrix_world.translation
                rows.append((item.gnm_index, loc.x, loc.y, loc.z))
            else:
                rows.append((item.gnm_index, 0.0, 0.0, 0.0))

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gnm_landmark_index", "x", "y", "z"])
            writer.writerows(sorted(rows))
            
        self.report({"INFO"}, "Export complet!")
        return {"FINISHED"}

class GNM_UL_markers(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.label(text="", icon="CHECKMARK" if item.is_placed else "RADIOBUT_OFF")
        side_str = "[M]" if item.side == 0 else ("[Dr]" if item.side == -1 else "[St]")
        row.label(text=f"{side_str} {item.label}")
        row.prop(item, "tissue_depth_mm", text="mm")

class GNM_PT_panel(Panel):
    bl_label = "GNM Markeri V11"
    bl_idname = "GNM_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Markeri"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.gnm_settings

        layout.operator("gnm.import_setup", icon="IMPORT")
        layout.separator()
        
        box = layout.box()
        box.label(text="Dimensiuni Vizuale Markeri:")
        box.prop(settings, "marker_size_mm")
        box.prop(settings, "peg_thickness_mm")
        layout.separator()
        
        if not scene.gnm_markers:
            layout.operator("gnm.init_markers", icon="ADD")
            return

        layout.template_list("GNM_UL_markers", "", scene, "gnm_markers", scene, "gnm_marker_active_index")
        
        row = layout.row()
        row.operator("gnm.place_marker", icon="RESTRICT_SELECT_OFF")
        
        layout.separator()
        layout.operator("gnm.export_csv", icon="EXPORT")

_classes = (GNMSettings, GNMMarkerItem, GNM_OT_import_setup, GNM_OT_init_markers, GNM_OT_place_marker, GNM_OT_export_csv, GNM_UL_markers, GNM_PT_panel)

def register():
    for cls in _classes: bpy.utils.register_class(cls)
    bpy.types.Scene.gnm_settings = PointerProperty(type=GNMSettings)
    bpy.types.Scene.gnm_markers = CollectionProperty(type=GNMMarkerItem)
    bpy.types.Scene.gnm_marker_active_index = IntProperty(default=0)

def unregister():
    for cls in reversed(_classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gnm_settings
    del bpy.types.Scene.gnm_markers
    del bpy.types.Scene.gnm_marker_active_index

if __name__ == "__main__":
    register()