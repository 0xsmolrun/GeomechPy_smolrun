import pytest
from geomechpy.unit_converter import UnitConverter

TOLERANCE = 1e-6

class TestConvertEquivalentMudWeight:
    def test_ppg_to_sg_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(8.345404, 'ppg', 'sg')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_sg_to_kg_per_m3_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.0, 'sg', 'kg/m3')
        assert result == pytest.approx(1000.0, rel=TOLERANCE)

    def test_sg_to_ppg_drilling_mud(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.2, 'sg', 'ppg')
        expected = 1.2 * 8.345404
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_kg_per_m3_to_ppg_seawater(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1025.0, 'kg/m3', 'ppg')
        expected = (1025.0 / 1000.0) * 8.345404
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_g_per_cm3_to_lb_per_ft3_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.0, 'g/cm3', 'lb/ft3')
        assert result == pytest.approx(62.42796, rel=TOLERANCE)

    def test_psi_per_ft_to_sg_fresh_water_gradient(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(0.433515, 'psi/ft', 'sg')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_pcf_to_kg_per_m3(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(65.0, 'pcf', 'kg/m3')
        expected = (65.0 / 62.42796) * 1000.0
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_lb_gal_alias_is_identical_to_ppg(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(10.0, 'lb/gal', 'ppg')
        assert result == pytest.approx(10.0, rel=TOLERANCE)

    def test_pcf_alias_is_identical_to_lb_per_ft3(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(65.0, 'pcf', 'lb/ft3')
        assert result == pytest.approx(65.0, rel=TOLERANCE)


    def test_sg_to_mpa_per_m_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.0, 'sg', 'mpa/m')
        assert result == pytest.approx(0.00980665, rel=TOLERANCE)

    def test_sg_to_kpa_per_m_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.0, 'sg', 'kpa/m')
        assert result == pytest.approx(9.80665, rel=TOLERANCE)

    def test_sg_to_bar_per_m_fresh_water(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(1.0, 'sg', 'bar/m')
        assert result == pytest.approx(0.0980665, rel=TOLERANCE)

    def test_mpa_per_m_to_kpa_per_m(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(0.00980665, 'mpa/m', 'kpa/m')
        assert result == pytest.approx(9.80665, rel=TOLERANCE)

    def test_same_unit_returns_input_unchanged(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight(9.5, 'ppg', 'ppg')
        assert result == 9.5

    def test_round_trip_ppg_to_sg_to_ppg(self) -> None:
        intermediate = UnitConverter.convert_equivalent_mud_weight(10.5, 'ppg', 'sg')
        result = UnitConverter.convert_equivalent_mud_weight(intermediate, 'sg', 'ppg')
        assert result == pytest.approx(10.5, rel=TOLERANCE)

    def test_invalid_output_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight(1.0, 'ppg', 'psi')

    def test_invalid_input_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight(1.0, 'invalid', 'sg')

    def test_returns_float_type(self) -> None:
        assert isinstance(UnitConverter.convert_equivalent_mud_weight(1.0, 'sg', 'ppg'), float)


class TestConvertEquivalentMudWeightArray:
    """Array variant of equivalent mud weight conversions."""

    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_array([8.345404, 16.690808], 'ppg', 'sg')
        assert result[0] == pytest.approx(1.0, rel=TOLERANCE)
        assert result[1] == pytest.approx(2.0, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_array([1.0, 1.2], 'sg', 'ppg')
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        assert UnitConverter.convert_equivalent_mud_weight_array([], 'sg', 'ppg') == []


class TestConvertPressure:
    def test_mpa_to_psi(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'mpa', 'psi')
        expected = 1_000_000.0 / 6894.757293168
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_kpa_to_bar(self) -> None:
        result = UnitConverter.convert_pressure(1000.0, 'kpa', 'bar')
        assert result == pytest.approx(10.0, rel=TOLERANCE)

    def test_atm_to_pa(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'atm', 'pa')
        assert result == pytest.approx(101325.0, rel=TOLERANCE)

    def test_psi_to_pa(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'psi', 'pa')
        assert result == pytest.approx(6894.757293168, rel=TOLERANCE)

    def test_760_mmhg_to_kpa_standard_atmosphere(self) -> None:
        result = UnitConverter.convert_pressure(760.0, 'mmhg', 'kpa')
        expected = (760.0 * 133.322387415) / 1000.0
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_bar_to_mbar(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'bar', 'mbar')
        assert result == pytest.approx(1000.0, rel=TOLERANCE)

    def test_gpa_to_mpa(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'gpa', 'mpa')
        assert result == pytest.approx(1000.0, rel=TOLERANCE)

    def test_atm_to_psi(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'atm', 'psi')
        expected = 101325.0 / 6894.757293168
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_atm_to_inhg_standard_atmosphere(self) -> None:
        result = UnitConverter.convert_pressure(1.0, 'atm', 'inhg')
        expected = 101325.0 / 3386.389
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_same_unit_returns_input_unchanged(self) -> None:
        result = UnitConverter.convert_pressure(5000.0, 'psi', 'psi')
        assert result == 5000.0

    def test_round_trip_psi_to_mpa_to_psi(self) -> None:
        intermediate = UnitConverter.convert_pressure(5000.0, 'psi', 'mpa')
        result = UnitConverter.convert_pressure(intermediate, 'mpa', 'psi')
        assert result == pytest.approx(5000.0, rel=TOLERANCE)

    def test_invalid_output_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_pressure(1.0, 'mpa', 'ppg')

    def test_invalid_input_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_pressure(1.0, 'invalid', 'psi')

    def test_returns_float_type(self) -> None:
        assert isinstance(UnitConverter.convert_pressure(1.0, 'mpa', 'psi'), float)


class TestConvertPressureArray:
    """Array variant of pressure conversions."""

    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_pressure_array([1.0, 2.0], 'mpa', 'psi')
        expected_one_mpa_in_psi = 1_000_000.0 / 6894.757293168
        assert result[0] == pytest.approx(expected_one_mpa_in_psi, rel=TOLERANCE)
        assert result[1] == pytest.approx(2.0 * expected_one_mpa_in_psi, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_pressure_array([1.0, 2.0], 'bar', 'kpa')
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        assert UnitConverter.convert_pressure_array([], 'psi', 'mpa') == []


class TestConvertLength:
    def test_ft_to_m(self) -> None:
        result = UnitConverter.convert_length(1.0, 'ft', 'm')
        assert result == pytest.approx(0.3048, rel=TOLERANCE)

    def test_ft_to_inches(self) -> None:
        result = UnitConverter.convert_length(1.0, 'ft', 'in')
        assert result == pytest.approx(12.0, rel=TOLERANCE)

    def test_one_mile_to_feet(self) -> None:
        result = UnitConverter.convert_length(1.0, 'mi', 'ft')
        assert result == pytest.approx(5280.0, rel=TOLERANCE)

    def test_one_yard_to_feet(self) -> None:
        result = UnitConverter.convert_length(1.0, 'yd', 'ft')
        assert result == pytest.approx(3.0, rel=TOLERANCE)

    def test_100_cm_to_m(self) -> None:
        result = UnitConverter.convert_length(100.0, 'cm', 'm')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_1000_mm_to_m(self) -> None:
        result = UnitConverter.convert_length(1000.0, 'mm', 'm')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_km_to_miles(self) -> None:
        result = UnitConverter.convert_length(1.0, 'km', 'mi')
        expected = 1000.0 / 1609.344
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_m_to_inches(self) -> None:
        result = UnitConverter.convert_length(1.0, 'm', 'in')
        expected = 1.0 / 0.0254
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_foot_alias_is_identical_to_ft(self) -> None:
        result = UnitConverter.convert_length(1.0, 'foot', 'ft')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_feet_alias_is_identical_to_ft(self) -> None:
        result = UnitConverter.convert_length(1.0, 'feet', 'ft')
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_same_unit_returns_input_unchanged(self) -> None:
        result = UnitConverter.convert_length(5000.0, 'ft', 'ft')
        assert result == 5000.0

    def test_round_trip_ft_to_m_to_ft(self) -> None:
        intermediate = UnitConverter.convert_length(10000.0, 'ft', 'm')
        result = UnitConverter.convert_length(intermediate, 'm', 'ft')
        assert result == pytest.approx(10000.0, rel=TOLERANCE)

    def test_invalid_output_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_length(1.0, 'ft', 'nautical_mile')

    def test_invalid_input_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_length(1.0, 'fathom', 'm')

    def test_returns_float_type(self) -> None:
        assert isinstance(UnitConverter.convert_length(1.0, 'ft', 'm'), float)


class TestConvertLengthArray:
    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_length_array([1.0, 2.0], 'ft', 'm')
        assert result[0] == pytest.approx(0.3048, rel=TOLERANCE)
        assert result[1] == pytest.approx(0.6096, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_length_array([100.0, 200.0], 'cm', 'm')
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        assert UnitConverter.convert_length_array([], 'ft', 'm') == []

class TestConvertAcousticVelocityToVelocity:
    def test_m_per_s_to_ft_per_s(self) -> None:
        result = UnitConverter.convert_acoustic(1000.0, 'm/s', 'ft/s')
        expected = 1000.0 / 0.3048
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_mph_to_m_per_s(self) -> None:
        result = UnitConverter.convert_acoustic(60.0, 'mph', 'm/s')
        assert result == pytest.approx(26.8224, rel=TOLERANCE)

    def test_kft_per_s_to_ft_per_s(self) -> None:
        result = UnitConverter.convert_acoustic(1.0, 'kft/s', 'ft/s')
        assert result == pytest.approx(1000.0, rel=TOLERANCE)

    def test_ft_per_sec_alias_is_identical_to_ft_per_s(self) -> None:
        result = UnitConverter.convert_acoustic(5000.0, 'ft/sec', 'ft/s')
        assert result == pytest.approx(5000.0, rel=TOLERANCE)


class TestConvertAcousticSlownessToSlowness:
    def test_us_per_ft_to_us_per_m(self) -> None:
        result = UnitConverter.convert_acoustic(305.0, 'us/ft', 'us/m')
        expected = 305.0 * 3.280839895
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_us_per_m_to_s_per_m(self) -> None:
        result = UnitConverter.convert_acoustic(100.0, 'us/m', 's/m')
        assert result == pytest.approx(1e-4, rel=TOLERANCE)

    def test_us_per_foot_alias_is_identical_to_us_per_ft(self) -> None:
        result = UnitConverter.convert_acoustic(70.0, 'us/foot', 'us/ft')
        assert result == pytest.approx(70.0, rel=TOLERANCE)

    def test_us_per_meter_alias_is_identical_to_us_per_m(self) -> None:
        result = UnitConverter.convert_acoustic(120.0, 'us/meter', 'us/m')
        assert result == pytest.approx(120.0, rel=TOLERANCE)


class TestConvertAcousticVelocityToSlowness:
    """Cross-domain: velocity → slowness via s = 1/v."""

    def test_m_per_s_to_us_per_ft(self) -> None:
        result = UnitConverter.convert_acoustic(5000.0, 'm/s', 'us/ft')
        slowness_seconds_per_meter = 1.0 / 5000.0
        expected = slowness_seconds_per_meter / 3.280839895e-6
        assert result == pytest.approx(expected, rel=TOLERANCE)

    def test_ft_per_s_to_us_per_ft_round_number(self) -> None:
        result = UnitConverter.convert_acoustic(2000.0, 'ft/s', 'us/ft')
        assert result == pytest.approx(500.0, rel=TOLERANCE)


class TestConvertAcousticSlownessToVelocity:
    def test_us_per_ft_to_m_per_s(self) -> None:
        result = UnitConverter.convert_acoustic(200.0, 'us/ft', 'm/s')
        assert result == pytest.approx(1524.0, rel=TOLERANCE)

    def test_us_per_ft_to_ft_per_s_round_number(self) -> None:
        result = UnitConverter.convert_acoustic(500.0, 'us/ft', 'ft/s')
        assert result == pytest.approx(2000.0, rel=TOLERANCE)


class TestConvertAcousticEdgeCasesAndErrors:
    def test_velocity_slowness_round_trip(self) -> None:
        intermediate = UnitConverter.convert_acoustic(70.0, 'us/ft', 'm/s')
        result = UnitConverter.convert_acoustic(intermediate, 'm/s', 'us/ft')
        assert result == pytest.approx(70.0, rel=TOLERANCE)

    def test_same_unit_returns_input_unchanged(self) -> None:
        result = UnitConverter.convert_acoustic(80.0, 'us/ft', 'us/ft')
        assert result == 80.0

    def test_zero_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_acoustic(0.0, 'm/s', 'ft/s')

    def test_negative_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_acoustic(-100.0, 'm/s', 'ft/s')

    def test_invalid_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_acoustic(1000.0, 'm/s', 'hands_per_fortnight')

    def test_returns_float_type(self) -> None:
        assert isinstance(UnitConverter.convert_acoustic(1000.0, 'm/s', 'ft/s'), float)


class TestConvertAcousticArray:
    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_acoustic_array([1000.0, 2000.0], 'm/s', 'ft/s')
        assert result[0] == pytest.approx(1000.0 / 0.3048, rel=TOLERANCE)
        assert result[1] == pytest.approx(2000.0 / 0.3048, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_acoustic_array([70.0, 80.0], 'us/ft', 'm/s')
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        assert UnitConverter.convert_acoustic_array([], 'm/s', 'us/ft') == []


class TestConvertFromPressureToEquivalentMudWeight:
    """Scalar pressure → EMW conversions using the hydrostatic pressure law.

    All expected values derived from:
        SG = P_pa / (GRAVITY * TVD_m)
    where GRAVITY = 9806.65 Pa/(SG·m).

    TVD resolution: 'psi' input → TVD in feet; all others → TVD in metres.
    """

    def test_fresh_water_pa_at_1000m_gives_sg_1(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=9806650.0,
            tvd=1000.0,
            unit_in='pa',
            unit_out='sg',
        )
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_fresh_water_mpa_at_1000m_gives_sg_1(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=9.80665,
            tvd=1000.0,
            unit_in='mpa',
            unit_out='sg',
        )
        assert result == pytest.approx(1.0, rel=TOLERANCE)

    def test_fresh_water_pa_at_1000m_gives_mpa_per_m(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=9806650.0,
            tvd=1000.0,
            unit_in='pa',
            unit_out='mpa/m',
        )
        assert result == pytest.approx(0.00980665, rel=TOLERANCE)

    def test_psi_input_uses_feet_for_tvd(self) -> None:
        pa_expected = 5000.0 * 6894.757293168
        tvd_m_expected = 10000.0 * 0.3048
        sg_expected = pa_expected / (9806.65 * tvd_m_expected)
        ppg_expected = sg_expected * 8.345404
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=5000.0,
            tvd=10000.0,
            unit_in='psi',
            unit_out='ppg',
        )
        assert result == pytest.approx(ppg_expected, rel=TOLERANCE)

    def test_round_trip_with_inverse_function(self) -> None:
        pressure_psi = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=10.0,
            tvd=10000.0,
            unit_in='ppg',
            unit_out='psi',
        )
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=pressure_psi,
            tvd=10000.0,
            unit_in='psi',
            unit_out='ppg',
        )
        assert result == pytest.approx(10.0, rel=TOLERANCE)

    def test_zero_tvd_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
                pressure_value=1000.0,
                tvd=0.0,
                unit_in='pa',
                unit_out='sg',
            )

    def test_negative_tvd_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
                pressure_value=1000.0,
                tvd=-500.0,
                unit_in='pa',
                unit_out='sg',
            )

    def test_invalid_input_pressure_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
                pressure_value=1.0,
                tvd=1000.0,
                unit_in='ppg',
                unit_out='sg',
            )

    def test_invalid_output_emw_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
                pressure_value=1.0,
                tvd=1000.0,
                unit_in='mpa',
                unit_out='psi',
            )

    def test_returns_float_type(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=9806650.0,
            tvd=1000.0,
            unit_in='pa',
            unit_out='sg',
        )
        assert isinstance(result, float)


class TestConvertFromPressureToEquivalentMudWeightArray:
    """Array variant of pressure → EMW conversions."""

    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight_array(
            pressure_values=[9806650.0, 19613300.0],
            tvd_values=[1000.0, 1000.0],
            unit_in='pa',
            unit_out='sg',
        )
        assert result[0] == pytest.approx(1.0, rel=TOLERANCE)
        assert result[1] == pytest.approx(2.0, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight_array(
            pressure_values=[9806650.0],
            tvd_values=[1000.0],
            unit_in='pa',
            unit_out='sg',
        )
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight_array(
            pressure_values=[],
            tvd_values=[],
            unit_in='pa',
            unit_out='sg',
        )
        assert result == []


class TestConvertEquivalentMudWeightToPressure:
    """Scalar EMW → pressure conversions using the hydrostatic pressure law.

    All expected values derived from:
        P_pa = SG * GRAVITY * TVD_m
    where GRAVITY = 9806.65 Pa/(SG·m).

    TVD resolution: 'psi' output → TVD in feet; all others → TVD in metres.
    """

    def test_fresh_water_at_1000m_gives_exact_pa(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=1.0,
            tvd=1000.0,
            unit_in='sg',
            unit_out='pa',
        )
        assert result == pytest.approx(9806650.0, rel=TOLERANCE)

    def test_fresh_water_at_1000m_gives_exact_mpa(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=1.0,
            tvd=1000.0,
            unit_in='sg',
            unit_out='mpa',
        )
        assert result == pytest.approx(9.80665, rel=TOLERANCE)

    def test_psi_output_uses_feet_for_tvd(self) -> None:
        sg_value = 10.0 / 8.345404
        tvd_metres = 10000.0 * 0.3048
        pa_expected = sg_value * 9806.65 * tvd_metres
        psi_expected = pa_expected / 6894.757293168
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=10.0,
            tvd=10000.0,
            unit_in='ppg',
            unit_out='psi',
        )
        assert result == pytest.approx(psi_expected, rel=TOLERANCE)

    def test_kg_per_m3_to_bar_at_500m(self) -> None:
        pa_expected = 1.2 * 9806.65 * 500.0
        bar_expected = pa_expected / 100000.0
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=1200.0,
            tvd=500.0,
            unit_in='kg/m3',
            unit_out='bar',
        )
        assert result == pytest.approx(bar_expected, rel=TOLERANCE)

    def test_round_trip_with_inverse_function(self) -> None:
        pressure_mpa = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=1.2,
            tvd=2000.0,
            unit_in='sg',
            unit_out='mpa',
        )
        result = UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
            pressure_value=pressure_mpa,
            tvd=2000.0,
            unit_in='mpa',
            unit_out='sg',
        )
        assert result == pytest.approx(1.2, rel=TOLERANCE)

    def test_zero_tvd_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight_to_pressure(
                mud_weight_value=1.0,
                tvd=0.0,
                unit_in='sg',
                unit_out='pa',
            )

    def test_negative_tvd_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight_to_pressure(
                mud_weight_value=1.0,
                tvd=-1000.0,
                unit_in='sg',
                unit_out='pa',
            )

    def test_invalid_input_emw_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight_to_pressure(
                mud_weight_value=1.0,
                tvd=1000.0,
                unit_in='psi',
                unit_out='mpa',
            )

    def test_invalid_output_pressure_unit_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            UnitConverter.convert_equivalent_mud_weight_to_pressure(
                mud_weight_value=1.0,
                tvd=1000.0,
                unit_in='sg',
                unit_out='ppg',
            )

    def test_returns_float_type(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure(
            mud_weight_value=1.0,
            tvd=1000.0,
            unit_in='sg',
            unit_out='pa',
        )
        assert isinstance(result, float)


class TestConvertEquivalentMudWeightToPressureArray:
    """Array variant of EMW → pressure conversions."""

    def test_each_element_matches_independent_scalar_result(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure_array(
            mud_weight_values=[1.0, 2.0],
            tvd_values=[1000.0, 1000.0],
            unit_in='sg',
            unit_out='pa',
        )
        assert result[0] == pytest.approx(9806650.0, rel=TOLERANCE)
        assert result[1] == pytest.approx(19613300.0, rel=TOLERANCE)

    def test_returns_list_type(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure_array(
            mud_weight_values=[1.0],
            tvd_values=[1000.0],
            unit_in='sg',
            unit_out='pa',
        )
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self) -> None:
        result = UnitConverter.convert_equivalent_mud_weight_to_pressure_array(
            mud_weight_values=[],
            tvd_values=[],
            unit_in='sg',
            unit_out='pa',
        )
        assert result == []
