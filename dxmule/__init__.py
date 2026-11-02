"""DXMulE: multi-scale prototype explanations on top of IDEA.

Scales are MSExplainer-style edge-retention ratios {k_m} applied to one
shared edge-score vector. The IDEA codebook, HGTokenizer / RQ-VAE, and
Wasserstein prototype alignment stay in place. The added piece is a
cross-scale map between explanatory prototypes (C2).

Nothing in this package is a measured improvement over IDEA or MSExplainer.
See the README section "DXMulE multi-scale" for the hypothesis list.
"""

from dxmule.cross_scale import CrossScalePrototypeLink
from dxmule.explainer import DXMulE
from dxmule.scales import EdgeRetentionScales

__all__ = [
    "CrossScalePrototypeLink",
    "DXMulE",
    "EdgeRetentionScales",
]
