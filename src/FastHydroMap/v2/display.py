"""Recommended color saturation only; numerical predictions remain raw.

These are the original HydroMap/FastHydroMap visualization ranges, not
data-derived percentiles, training clipping thresholds or confidence bounds.
"""

PC_DISPLAY_RANGES = {
    "PC1": (-8.0, 8.0),
    "PC2": (-2.0, 8.0),
    "PC3": (-2.0, 2.0),
}
