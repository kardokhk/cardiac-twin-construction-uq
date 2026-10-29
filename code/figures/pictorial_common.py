"""Segmentation-source styling shared by fig1_pipeline.py and graphical_abstract.py (one style per source)."""
from figcommon import OI

# one colour, line style and marker per segmentation source, used in Fig. 1 and the graphical abstract
SOURCES = ["expert", "nnunet_task027_acdc", "nnunet_task114_mnms", "nnunet_ukb_sax", "cinema_acdc_sax"]
SRC_LABEL = {"expert": "Expert", "nnunet_task027_acdc": "nnU-Net Task027",
             "nnunet_task114_mnms": "nnU-Net Task114", "nnunet_ukb_sax": "nnU-Net UKB SAX",
             "cinema_acdc_sax": "CineMA"}
SRC_STYLE = {  # colour, line style, marker
    "expert": ("#000000", "-", "o"),
    "nnunet_task027_acdc": (OI["sky"], (0, (4, 1.5)), "s"),
    "nnunet_task114_mnms": (OI["vermillion"], (0, (1, 1)), "^"),
    "nnunet_ukb_sax": (OI["purple"], (0, (4, 1, 1, 1)), "D"),
    "cinema_acdc_sax": (OI["yellow"], (0, (2, 1)), "v"),
}
