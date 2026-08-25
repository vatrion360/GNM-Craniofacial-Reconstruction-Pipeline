# -*- coding: utf-8 -*-
"""Registru anatomic UNIC pentru landmarkurile craniofaciale.

Sursa de adevar partajata de addon-ul Blender (addon_v12.py) si de
pipeline-ul de reconstructie (gnm_reconstruct.py). Contine NUMAI date
anatomice independente de modelul statistic: etichete, latura, adancimi
de tesut, ponderi de incredere, perechi pentru verificari de consistenta,
hinturi de plasare, grupuri protejate si offseturi de regiuni dense.

Legatura dintre etichete si indecsii de vertex ai unui model concret
(ex. GNM Head v3.0) NU sta aici - este responsabilitatea backend-ului
(vezi cranio.backend.gnm_backend).

Adancimile de tesut urmeaza literatura: Rhine & Campbell 1980;
De Greef et al. 2006; Stephan & Simpson 2008 (corectate in addon V12).
"""

# Eticheta -> (adancime_tesut_mm, latura)
# latura: -1 = dreapta subiectului (_Dr), +1 = stanga (_St), 0 = median.
LANDMARK_INFO = {
    "Nasion": (6.0, 0), "Rhinion": (3.0, 0), "Glabella": (5.0, 0),
    "Pogonion": (10.0, 0), "Gnathion": (10.5, 0),
    "Gonion_Dr": (13.0, -1), "Gonion_St": (13.0, 1),
    "Orbita_Dr_Ext": (8.0, -1), "Orbita_St_Ext": (8.0, 1),
    "Orbita_Dr_Int": (6.0, -1), "Orbita_St_Int": (6.0, 1),
    "Supraorbitale_Dr": (7.0, -1), "Supraorbitale_St": (7.0, 1),
    "Infraorbitale_Dr": (7.0, -1), "Infraorbitale_St": (7.0, 1),
    "Zygion_Dr": (8.5, -1), "Zygion_St": (8.5, 1),
    "Alare_Dr": (4.5, -1), "Alare_St": (4.5, 1),
    "Eurion_Dr": (5.0, -1), "Eurion_St": (5.0, 1),
    "Vertex_VarfCap": (5.5, 0),
    "Nasospinale_BazaNas": (11.0, 0), "Prosthion_BuzaSup": (12.0, 0),
}

# Ordinea canonica (cea din addonul V12) - folosita la initializarea
# listei de markeri in Blender si la documentatie.
LANDMARK_ORDER = [
    "Nasion", "Rhinion", "Glabella", "Pogonion", "Gnathion",
    "Gonion_Dr", "Gonion_St", "Orbita_Dr_Ext", "Orbita_St_Ext",
    "Orbita_Dr_Int", "Orbita_St_Int", "Supraorbitale_Dr",
    "Supraorbitale_St", "Infraorbitale_Dr", "Infraorbitale_St",
    "Zygion_Dr", "Zygion_St", "Alare_Dr", "Alare_St",
    "Eurion_Dr", "Eurion_St", "Vertex_VarfCap",
    "Nasospinale_BazaNas", "Prosthion_BuzaSup",
]

# Ponderi de incredere per landmark (pentru alinierea/fitul ponderat).
# Markerii ososi precisi primesc 1.0; cei cu pozitie mai putin riguroasa
# sau cu tesut moale gros/variabil primesc mai putin.
CONFIDENCE_WEIGHTS = {
    "Nasion": 1.0, "Rhinion": 1.0, "Glabella": 1.0,
    "Pogonion": 1.0, "Gnathion": 1.0,
    "Gonion_Dr": 1.0, "Gonion_St": 1.0,
    "Orbita_Dr_Ext": 1.0, "Orbita_St_Ext": 1.0,
    "Orbita_Dr_Int": 1.0, "Orbita_St_Int": 1.0,
    "Supraorbitale_Dr": 0.5, "Supraorbitale_St": 0.5,
    "Infraorbitale_Dr": 0.7, "Infraorbitale_St": 0.7,
    "Zygion_Dr": 1.0, "Zygion_St": 1.0,
    "Alare_Dr": 0.7, "Alare_St": 0.7,
    "Eurion_Dr": 0.5, "Eurion_St": 0.5,
    "Vertex_VarfCap": 0.5, "Nasospinale_BazaNas": 1.0,
    "Prosthion_BuzaSup": 1.0,
}

DEFAULT_CONFIDENCE = 0.7

# Perechi de referinta pentru verificarea consistentei plasarii markerilor:
# bilaterale (latime) + lantul median facial (inaltime/profundime).
CONSISTENCY_PAIRS = [
    ("Orbita_Dr_Int", "Orbita_St_Int"),
    ("Orbita_Dr_Ext", "Orbita_St_Ext"),
    ("Infraorbitale_Dr", "Infraorbitale_St"),
    ("Supraorbitale_Dr", "Supraorbitale_St"),
    ("Zygion_Dr", "Zygion_St"),
    ("Gonion_Dr", "Gonion_St"),
    ("Eurion_Dr", "Eurion_St"),
    ("Alare_Dr", "Alare_St"),
    ("Nasion", "Rhinion"),
    ("Rhinion", "Nasospinale_BazaNas"),
    ("Nasospinale_BazaNas", "Prosthion_BuzaSup"),
    ("Prosthion_BuzaSup", "Pogonion"),
    ("Pogonion", "Gnathion"),
    ("Nasion", "Pogonion"),
    ("Orbita_Dr_Int", "Infraorbitale_Dr"),
    ("Orbita_St_Int", "Infraorbitale_St"),
]

# Indicatii de plasare corecta pentru markerii frecvent gresiti (apar in
# avertismente, ca ajutor pentru re-plasarea in Blender).
PLACEMENT_HINTS = {
    "Orbita_Dr_Int": "punctul osos dacryon (marginea mediala a orbitei), "
                     "nu centrul orbitei/globul ocular",
    "Orbita_St_Int": "punctul osos dacryon (marginea mediala a orbitei), "
                     "nu centrul orbitei/globul ocular",
    "Infraorbitale_Dr": "foramenul infraorbital (sub marginea orbitei, "
                        "pe verticala pupilei)",
    "Infraorbitale_St": "foramenul infraorbital (sub marginea orbitei, "
                        "pe verticala pupilei)",
    "Rhinion": "mijlocul oaselor nazale (~25 mm sub Nasion), pe os",
    "Pogonion": "proeminentea mentoniera anterioara (~35 mm deasupra "
                "Gnathion), nu marginea inferioara a mandibulei",
    "Prosthion_BuzaSup": "marginea inferioara a buzei superioare, pe median",
}

# Grupuri de vertecsi fara ancore anatomice (niciun landmark, nici os
# direct): campul TPS interpolat le deformeaza nerealist -> amortizate.
PROTECTED_GROUPS = ("eyes", "eye_interiors", "mouth_sock",
                    "upper_lip", "lower_lip")

# Offseturi per regiune faciala (mm) - preluate din tabelul de adancimi de
# tesut (aceeasi sursa ca LANDMARK_INFO), deci consistente cu baturile
# markerilor. Folosite de constrangerile dense faciale (--skull).
REGION_OFFSETS_MM = {
    "brow_median": 5.0,        # ~ Glabella
    "punte_nazala": 3.0,       # ~ Rhinion
    "zigomatic": 8.5,          # ~ Zygion
    "infraorbital": 7.0,       # ~ Infraorbitale
    "barbie": 10.0,            # ~ Pogonion
}

# Geometria regiunii dense "punte_nazala" (V13.3, corectata).
# Puntea osoasa se afla pe segmentul Nasion -> pronasale (vertex 12296),
# la ~45% din lungime (oasele nazale ocupa jumatatea superioara a nasului
# extern). Centrul patch-ului este interpolat acolo, cu raza de 14 mm, iar
# tot ce e sub 6 mm deasupra varfului este exclus (cartilaj/apertura).
# Istoric: pana in V13.3 patch-ul era centrat direct pe pronasale (sub
# numele eronat "Rhinion" - quirk corectat separat in V13.4 prin remaparea
# Rhinion -> 12310), deci constrangea apertura, nu puntea osoasa.
NOSE_BRIDGE_RADIUS_MM = 14.0    # raza patch-ului pe puntea osoasa
NOSE_BRIDGE_CENTER_FRAC = 0.45  # pozitia centrului pe Nasion -> pronasale
NOSE_BRIDGE_TIP_CUT_MM = 6.0    # exclude vertecsii sub varf+6 mm (cartilaj)


def side_of(label: str) -> int:
    """-1 pentru *_Dr (dreapta), +1 pentru *_St (stanga), 0 altfel."""
    if label.endswith("_Dr"):
        return -1
    if label.endswith("_St"):
        return 1
    return 0


def pair_label(label: str):
    """Eticheta perechii contralaterale (Dr<->St) sau None pentru mediani."""
    if "_Dr" in label:
        return label.replace("_Dr", "_St")
    if "_St" in label:
        return label.replace("_St", "_Dr")
    return None


def bilateral_pairs():
    """Lista de perechi (dr, st) ordonate, derivata din LANDMARK_INFO."""
    pairs = []
    for label in LANDMARK_ORDER:
        if label.endswith("_Dr"):
            st = pair_label(label)
            if st in LANDMARK_INFO:
                pairs.append((label, st))
    return pairs
