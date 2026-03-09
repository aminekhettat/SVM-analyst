"""Visualization helpers for SVM and inverter switching states.

This module includes helpers to generate plots for the SVM hexagon and related
switching state visualizations.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import numpy as np


def svm_hexagon_vertices(vdc: float = 1.0) -> np.ndarray:
    """Return the 6 active space vector vertices in the alpha-beta plane.

    A three-phase inverter has six active vectors located at 60 degree steps on a
    circle with radius 2/3*Vdc. We normalize to +/-1 for display.

    Returns
    -------
    vertices:
        An array of shape (6, 2) containing (alpha, beta) pairs.
    """

    # The six active space vectors are spaced 60 degrees apart.
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    radius = 2.0 / 3.0 * vdc
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def svm_reference_vector(theta: float, vdc: float = 1.0) -> np.ndarray:
    """Compute a reference vector (alpha-beta) for a given electrical angle.

    Parameters
    ----------
    theta:
        Electrical angle in radians.
    vdc:
        DC link voltage normalization.

    Returns
    -------
    ref:
        2-element array containing (alpha, beta).
    """

    # Park transform (abc to alpha-beta) for a unit amplitude reference.
    alpha = np.cos(theta)
    beta = np.sin(theta)
    return np.array([alpha, beta]) * (2.0 / 3.0 * vdc)
