"""Extended unit tests for the visualization module.

Atomic features covered:
- svm_hexagon_vertices: default vdc=1.0, custom vdc=2.0, shape, radius, closure
- svm_reference_vector: unit direction, scaling with vdc, multiple sector angles
"""

import numpy as np
import pytest

from svm_shaper.visualization import svm_hexagon_vertices, svm_reference_vector


# ---------------------------------------------------------------------------
# svm_hexagon_vertices
# ---------------------------------------------------------------------------


class TestSvmHexagonVertices:
    def test_default_shape(self):
        v = svm_hexagon_vertices()
        assert v.shape == (6, 2)

    def test_custom_shape(self):
        v = svm_hexagon_vertices(vdc=2.0)
        assert v.shape == (6, 2)

    def test_default_radius(self):
        """Vertices should lie on a circle of radius 2/3 * vdc."""
        vdc = 1.0
        v = svm_hexagon_vertices(vdc=vdc)
        radii = np.linalg.norm(v, axis=1)
        expected = 2.0 / 3.0 * vdc
        np.testing.assert_allclose(radii, expected, rtol=1e-6)

    def test_custom_vdc_radius(self):
        vdc = 2.0
        v = svm_hexagon_vertices(vdc=vdc)
        radii = np.linalg.norm(v, axis=1)
        expected = 2.0 / 3.0 * vdc
        np.testing.assert_allclose(radii, expected, rtol=1e-6)

    def test_vertices_evenly_spaced_60_degrees(self):
        """Adjacent vertices should be 60° apart (handle circular wrapping)."""
        v = svm_hexagon_vertices()
        alphas = np.arctan2(v[:, 1], v[:, 0])
        # Unwrap to handle the -π/+π boundary crossing
        alphas_unwrapped = np.unwrap(alphas)
        diffs = np.diff(alphas_unwrapped)
        expected_step = 2.0 * np.pi / 6.0
        np.testing.assert_allclose(diffs, expected_step, atol=1e-9)

    def test_vdc_zero_gives_origin_vertices(self):
        v = svm_hexagon_vertices(vdc=0.0)
        np.testing.assert_allclose(v, 0.0, atol=1e-12)

    def test_first_vertex_on_positive_x_axis(self):
        """The first vertex should lie on the positive alpha (x) axis (θ=0)."""
        v = svm_hexagon_vertices()
        assert v[0, 1] == pytest.approx(0.0, abs=1e-9)  # beta ≈ 0
        assert v[0, 0] > 0.0


# ---------------------------------------------------------------------------
# svm_reference_vector
# ---------------------------------------------------------------------------


class TestSvmReferenceVector:
    def test_output_shape(self):
        ref = svm_reference_vector(0.0)
        assert ref.shape == (2,)

    def test_at_theta_zero_points_positive_alpha(self):
        """At θ=0 the reference vector is entirely along the alpha axis."""
        ref = svm_reference_vector(0.0)
        assert ref[0] > 0.0
        assert ref[1] == pytest.approx(0.0, abs=1e-9)

    def test_at_theta_pi_half_points_positive_beta(self):
        ref = svm_reference_vector(np.pi / 2.0)
        assert ref[0] == pytest.approx(0.0, abs=1e-9)
        assert ref[1] > 0.0

    def test_magnitude_default_vdc(self):
        """Magnitude for vdc=1 should equal 2/3."""
        for theta in np.linspace(0, 2 * np.pi, 13):
            ref = svm_reference_vector(theta, vdc=1.0)
            mag = np.linalg.norm(ref)
            assert mag == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_magnitude_scales_with_vdc(self):
        ref1 = svm_reference_vector(0.5, vdc=1.0)
        ref2 = svm_reference_vector(0.5, vdc=2.0)
        assert np.linalg.norm(ref2) == pytest.approx(
            2.0 * np.linalg.norm(ref1), rel=1e-6
        )

    @pytest.mark.parametrize(
        "sector_angle",
        [
            0.0,
            np.pi / 3,
            2 * np.pi / 3,
            np.pi,
            4 * np.pi / 3,
            5 * np.pi / 3,
        ],
    )
    def test_sector_boundary_angles(self, sector_angle):
        """Reference vector at sector boundaries must have the correct magnitude."""
        ref = svm_reference_vector(sector_angle)
        mag = np.linalg.norm(ref)
        assert mag == pytest.approx(2.0 / 3.0, rel=1e-6)
