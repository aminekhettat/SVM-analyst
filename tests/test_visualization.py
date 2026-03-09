"""Unit tests for visualization helper functions.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.visualization import svm_hexagon_vertices, svm_reference_vector


def test_svm_hexagon_vertices_shape():
    verts = svm_hexagon_vertices()
    assert verts.shape == (6, 2)


def test_svm_reference_vector_magnitude():
    vec = svm_reference_vector(0.0)
    assert np.allclose(vec, [2.0 / 3.0, 0.0])
