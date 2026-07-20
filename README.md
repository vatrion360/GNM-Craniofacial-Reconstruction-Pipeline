# GNM Craniofacial Reconstruction Pipeline

This project provides an end-to-end technical solution for **craniofacial reconstruction** based on osteological data. It bridges the gap between 3D skull scans (osteology) and the **GNM (GNM: Generative aNthropometric Model and Ecosystem)** parametric model, enabling the generation of a facial geometry (skin) that accurately adheres to the subject's anatomical landmarks.

## Overview
The pipeline consists of two main components:
1.  **Blender Add-on (`addon_v11.py`):** An intuitive tool for skull calibration (import, scaling, orientation), scientific landmark placement, and automated data export.
2.  **Python Processing Script (`gnm_skull_to_face.py`):** The mathematical engine that performs spatial alignment (Umeyama algorithm) and linear regression (regularized fitting) of the GNM model onto your anatomical data.

---

## Correlation Table: Markers & Scientific Landmarks

This table correlates the labels used in the Blender Add-on interface with standard anthropological/anatomical terminology.

| Add-on Label | Scientific Name | Anatomical Region | Marker Type |
| :--- | :--- | :--- | :--- |
| **Nasion** | Nasion | Mid-sagittal | Exact (Vertex) |
| **Rhinion** | Rhinion | Mid-sagittal | Exact (Vertex) |
| **Glabella** | Glabella | Mid-sagittal | Exact (Vertex) |
| **Pogonion** | Pogonion | Mid-sagittal | Exact (Vertex) |
| **Gnathion** | Gnathion | Mid-sagittal | Exact (Vertex) |
| **Vertex_VarfCap** | Vertex | Mid-sagittal | Exact (Vertex) |
| **Nasospinale** | Nasospinale | Mid-sagittal | Barycentric |
| **Prosthion** | Prosthion | Mid-sagittal | Barycentric |
| **Gonion_Dr / St** | Gonion | Mandibular | Exact (Vertex) |
| **Orbita_Dr / St_Ext** | Ectoconchion | Orbital | Exact (Vertex) |
| **Orbita_Dr / St_Int** | Dacryon / Endocanthion | Orbital | Exact (Vertex) |
| **Supraorbitale_Dr / St**| Supraorbitale | Orbital | Exact (Vertex) |
| **Infraorbitale_Dr / St**| Infraorbitale | Orbital | Exact (Vertex) |
| **Zygion_Dr / St** | Zygion | Zygomatic | Exact (Vertex) |
| **Alare_Dr / St** | Alare | Nasal | Exact (Vertex) |
| **Eurion_Dr / St** | Eurion | Cranial | Exact (Vertex) |
| **Frontotemporale_Dr / St**| Frontotemporale | Temporal | Exact (Vertex) |

---

## Workflow

### 1. Blender Setup
1.  **Install:** Install the add-on via *Edit > Preferences > Add-ons > Install*.
2.  **Import:** Use the "Import & Calibrate Skull" button. The script automatically handles metric units (mm), centers the skull, and corrects inverted normals.
3.  **Place Markers:** Use "Load Marker List" to populate the slots. Select a marker and click on the corresponding anatomical point on the skull mesh. (for markers, for the moment the units in addon are not mm, even if it says so, you need to manually convert them - will fix this bug)
4.  **Export:** Generate the `.csv` file using the "Export Final CSV" button.

### 2. Reconstruction Pipeline
Run the fitting script via your terminal to generate the facial mesh:
```bash
python gnm_skull_to_face.py --input markeri_gnm.csv --output reconstructie.obj
```

**Requirements**
**Blender 5+**

### **Python 3.x environment with the following dependencies:**
1. **numpy**
2. **trimesh**
3. **gnm (Model installation required per GNM official docs)**

### Disclaimer
### This tool is a personal project intended for anthropological research and 3D modeling purposes. Reconstruction accuracy is strictly dependent on the precise placement of anatomical markers and the biological quality of the input skull scan.

### Developed for digital anthropology and advanced craniofacial reconstruction workflows.
