import math

import numpy as np
import pytest

from geomechpy.near_wellbore_stresses import (
    BoreholeGeneralStresses,
    BoreholeWallStresses,
    NearWellboreStressesCalculation,
    PrincipalStressesBoreholewall,
)

TOLERANCE = 1e-6


class TestKirschBoreholeWallStresses:
    def test_returns_borehole_wall_stresses_dataclass(self) -> None:
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=np.linspace(0, 360, 37),
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert isinstance(result, BoreholeWallStresses)

    def test_output_arrays_have_correct_length(self) -> None:
        theta = np.linspace(0, 360, 37)
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=theta,
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert len(result.sigma_boreholewall_rr) == len(theta)
        assert len(result.sigma_boreholewall_tt) == len(theta)
        assert len(result.sigma_boreholewall_zz) == len(theta)
        assert len(result.sigma_boreholewall_tz) == len(theta)
        assert len(result.sigma_boreholewall_rt) == len(theta)
        assert len(result.sigma_boreholewall_rz) == len(theta)

    def test_sigma_rr_equals_mud_minus_pore_pressure(self) -> None:
        # sigma_rr is the borehole wall boundary condition; it is constant around the circumference
        mud_pressure = 6000
        pore_pressure = 5000
        theta = np.linspace(0, 360, 37)
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=pore_pressure,
            shmax_azimuth=0,
            mud_pressure=mud_pressure,
            theta=theta,
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert result.sigma_boreholewall_rr == pytest.approx(np.full(len(theta), mud_pressure - pore_pressure), rel=TOLERANCE)

    def test_sigma_rt_is_zero_on_borehole_wall(self) -> None:
        theta = np.linspace(0, 360, 37)
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=theta,
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert result.sigma_boreholewall_rt == pytest.approx(np.zeros(len(theta)), abs=TOLERANCE)

    def test_sigma_rz_is_zero_on_borehole_wall(self) -> None:
        theta = np.linspace(0, 360, 37)
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=theta,
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert result.sigma_boreholewall_rz == pytest.approx(np.zeros(len(theta)), abs=TOLERANCE)

    def test_known_value_sigma_tt_vertical_well_balanced_mud(self) -> None:
        # Vertical borehole, shmax aligned North, balanced mud (mud=pprs), no shear (sxy=sxz=syz=0)
        # At theta=0 (top of hole): sigma_tt = 3*shmin - shmax = 3*10000 - 12000 = 18000
        # At theta=90:              sigma_tt = 3*shmax - shmin = 3*12000 - 10000 = 26000
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=np.array([0.0, 90.0]),
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert result.sigma_boreholewall_tt[0] == pytest.approx(18000, rel=TOLERANCE)
        assert result.sigma_boreholewall_tt[1] == pytest.approx(26000, rel=TOLERANCE)

    def test_known_value_sigma_zz_vertical_well_balanced_mud(self) -> None:
        # sigma_zz at theta=0: svert - pr*(2*(shmax-shmin)) = 13000 - 0.25*4000 = 12000
        # sigma_zz at theta=90: svert + pr*(2*(shmax-shmin)) = 13000 + 0.25*4000 = 14000
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(
            shmin=10000,
            shmax=12000,
            svert=13000,
            pore_pressure=5000,
            shmax_azimuth=0,
            mud_pressure=5000,
            theta=np.array([0.0, 90.0]),
            poisson_ratio_static=0.25,
            borehole_deviation=0,
            borehole_azimuth=0,
        )
        assert result.sigma_boreholewall_zz[0] == pytest.approx(12000, rel=TOLERANCE)
        assert result.sigma_boreholewall_zz[1] == pytest.approx(14000, rel=TOLERANCE)


class TestKirschBoreholeStressesGeneral:
    def test_returns_general_stresses_dataclass(self) -> None:
        theta = np.linspace(0, 360, 19)
        result = NearWellboreStressesCalculation.calculate_kirsch_borehole_stresses_general(
            shmin=10000, shmax=12000, svert=13000, pore_pressure=5000,
            shmax_azimuth=0, mud_pressure=5000, theta=theta,
            radius=np.full(len(theta), 8.5), poisson_ratio_static=0.25,
            borehole_deviation=30, borehole_azimuth=20, bitsize=8.5,
        )
        assert isinstance(result, BoreholeGeneralStresses)

    def test_reduces_to_wall_stresses_at_radius_equal_bitsize(self) -> None:
        # At r = bitsize the general Kirsch solution must equal the borehole-wall solution,
        # including for a deviated well where the shear terms are non-zero.
        theta = np.linspace(0, 360, 37)
        bitsize = 8.5
        common = dict(
            shmin=10000, shmax=12000, svert=13000, pore_pressure=5000,
            shmax_azimuth=15, mud_pressure=6000, theta=theta,
            poisson_ratio_static=0.25, borehole_deviation=35, borehole_azimuth=50,
        )
        wall = NearWellboreStressesCalculation.calculate_kirsch_borehole_wall_stresses(**common)
        general = NearWellboreStressesCalculation.calculate_kirsch_borehole_stresses_general(
            radius=np.full(len(theta), bitsize), bitsize=bitsize, **common
        )
        assert general.sigma_general_rr == pytest.approx(wall.sigma_boreholewall_rr, rel=TOLERANCE, abs=1e-6)
        assert general.sigma_general_tt == pytest.approx(wall.sigma_boreholewall_tt, rel=TOLERANCE, abs=1e-6)
        assert general.sigma_general_zz == pytest.approx(wall.sigma_boreholewall_zz, rel=TOLERANCE, abs=1e-6)
        assert general.sigma_general_tz == pytest.approx(wall.sigma_boreholewall_tz, rel=TOLERANCE, abs=1e-6)
        assert general.sigma_general_rt == pytest.approx(wall.sigma_boreholewall_rt, abs=1e-6)
        assert general.sigma_general_rz == pytest.approx(wall.sigma_boreholewall_rz, abs=1e-6)


class TestPrincipalStressesAnalytical:
    def test_returns_principal_stresses_dataclass(self) -> None:
        result = NearWellboreStressesCalculation.calculate_principal_stresses_analytical_boreholewall(
            sigma_boreholewall_rr=np.array([500.0]),
            sigma_boreholewall_tt=np.array([10000.0]),
            sigma_boreholewall_zz=np.array([8000.0]),
            sigma_boreholewall_tz=np.array([0.0]),
        )
        assert isinstance(result, PrincipalStressesBoreholewall)

    def test_principal_stresses_are_ordered(self) -> None:
        result = NearWellboreStressesCalculation.calculate_principal_stresses_analytical_boreholewall(
            sigma_boreholewall_rr=np.array([500.0, 600.0]),
            sigma_boreholewall_tt=np.array([10000.0, 8000.0]),
            sigma_boreholewall_zz=np.array([8000.0, 10000.0]),
            sigma_boreholewall_tz=np.array([1000.0, 1500.0]),
        )
        assert all(result.sigma_boreholewall_1 >= result.sigma_boreholewall_2)
        assert all(result.sigma_boreholewall_2 >= result.sigma_boreholewall_3)

    def test_no_shear_stress_principal_stresses_are_sorted_inputs(self) -> None:
        # With sigma_tz=0 the principal stresses are just the sorted (rr, tt, zz) values.
        result = NearWellboreStressesCalculation.calculate_principal_stresses_analytical_boreholewall(
            sigma_boreholewall_rr=np.array([500.0]),
            sigma_boreholewall_tt=np.array([10000.0]),
            sigma_boreholewall_zz=np.array([8000.0]),
            sigma_boreholewall_tz=np.array([0.0]),
        )
        assert result.sigma_boreholewall_1[0] == pytest.approx(10000.0, rel=TOLERANCE)
        assert result.sigma_boreholewall_2[0] == pytest.approx(8000.0, rel=TOLERANCE)
        assert result.sigma_boreholewall_3[0] == pytest.approx(500.0, rel=TOLERANCE)

    def test_known_values(self) -> None:
        # rr=500, tt=10000, zz=8000, tz=1000
        # In-plane roots: 9000 +/- 1000*sqrt(2) ≈ 10414.214 / 7585.786
        # Sorted with rr=500: sigma_1≈10414.214, sigma_2≈7585.786, sigma_3=500
        # theta_tortuosity = 0.5 * degrees(arctan(2000/2000)) = 22.5
        result = NearWellboreStressesCalculation.calculate_principal_stresses_analytical_boreholewall(
            sigma_boreholewall_rr=np.array([500.0]),
            sigma_boreholewall_tt=np.array([10000.0]),
            sigma_boreholewall_zz=np.array([8000.0]),
            sigma_boreholewall_tz=np.array([1000.0]),
        )
        assert result.sigma_boreholewall_1[0] == pytest.approx(9000 + 1000 * math.sqrt(2), rel=TOLERANCE)
        assert result.sigma_boreholewall_2[0] == pytest.approx(9000 - 1000 * math.sqrt(2), rel=TOLERANCE)
        assert result.sigma_boreholewall_3[0] == pytest.approx(500.0, rel=TOLERANCE)
        assert result.theta_boreholewall_tortuosity[0] == pytest.approx(22.5, rel=TOLERANCE)
