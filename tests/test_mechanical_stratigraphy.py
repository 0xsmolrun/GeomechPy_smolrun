import pytest

from geomechpy.mechanical_stratigraphy import (
    Lithology,
    MechanicalStratigraphyCalculation,
)


class TestLithologyCodes:
    def test_lithology_codes(self) -> None:
        # Mechanical stratigraphy convention: Sand=0, Shale=1, Limestone=2, Coal=6
        assert Lithology.SAND == 0
        assert Lithology.SHALE == 1
        assert Lithology.LIMESTONE == 2
        assert Lithology.COAL == 6


class TestDetermineLithology:
    def test_low_gamma_ray_is_sand(self) -> None:
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=40, gr_threshold=75) == 0

    def test_high_gamma_ray_is_shale(self) -> None:
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=120, gr_threshold=75) == 1

    def test_gamma_ray_at_threshold_is_shale(self) -> None:
        # Readings at or above the threshold are classified as shale
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=75, gr_threshold=75) == 1

    def test_gamma_ray_just_below_threshold_is_sand(self) -> None:
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=74.999, gr_threshold=75) == 0

    def test_coal_flag_overrides_gamma_ray(self) -> None:
        # A shale-level Gamma Ray reading is still coal when the coal flag is set
        assert MechanicalStratigraphyCalculation.determine_lithology(
            gamma_ray=120, gr_threshold=75, coal_flag=True
        ) == 6

    def test_limestone_flag_overrides_gamma_ray(self) -> None:
        assert MechanicalStratigraphyCalculation.determine_lithology(
            gamma_ray=40, gr_threshold=75, limestone_flag=True
        ) == 2

    def test_coal_flag_takes_precedence_over_limestone_flag(self) -> None:
        result = MechanicalStratigraphyCalculation.determine_lithology(
            gamma_ray=40, gr_threshold=75, coal_flag=True, limestone_flag=True
        )
        assert result == 6

    def test_default_threshold(self) -> None:
        # Default threshold (baseline) is 75 gAPI
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=50) == 0
        assert MechanicalStratigraphyCalculation.determine_lithology(gamma_ray=90) == 1


class TestDetermineLithologyArray:
    def test_returns_expected_codes(self) -> None:
        gamma_ray = [30, 80, 60, 150]
        result = MechanicalStratigraphyCalculation.determine_lithology_array(
            gamma_ray=gamma_ray, gr_threshold=75
        )
        assert result == [0, 1, 0, 1]

    def test_coal_flag_array(self) -> None:
        gamma_ray = [30, 80, 60]
        coal_flag = [False, False, True]
        result = MechanicalStratigraphyCalculation.determine_lithology_array(
            gamma_ray=gamma_ray, gr_threshold=75, coal_flag=coal_flag
        )
        assert result == [0, 1, 6]

    def test_limestone_flag_array(self) -> None:
        gamma_ray = [30, 80, 60]
        limestone_flag = [True, False, False]
        result = MechanicalStratigraphyCalculation.determine_lithology_array(
            gamma_ray=gamma_ray, gr_threshold=75, limestone_flag=limestone_flag
        )
        assert result == [2, 1, 0]

    def test_no_flags_defaults_to_gamma_ray_classification(self) -> None:
        gamma_ray = [10, 200]
        result = MechanicalStratigraphyCalculation.determine_lithology_array(
            gamma_ray=gamma_ray, gr_threshold=75
        )
        assert result == [0, 1]

    def test_mismatched_coal_flag_length_raises(self) -> None:
        with pytest.raises(ValueError):
            MechanicalStratigraphyCalculation.determine_lithology_array(
                gamma_ray=[30, 80], coal_flag=[False]
            )

    def test_mismatched_limestone_flag_length_raises(self) -> None:
        with pytest.raises(ValueError):
            MechanicalStratigraphyCalculation.determine_lithology_array(
                gamma_ray=[30, 80], limestone_flag=[True]
            )
