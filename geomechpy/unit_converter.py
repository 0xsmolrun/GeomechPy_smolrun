from dataclasses import dataclass


class UnitConverterConstants:
    """Shared conversion-factor tables and physical constants for UnitConverter.

    All factors follow a single convention: multiplying a value in the given
    unit by the factor yields the SI baseline value (Pa, m, m/s, or s/m).
    Mud-weight factors are an exception: dividing by the factor yields SG.

    Reference: N/A
    """

    # Hydrostatic pressure exerted by 1 SG fluid over 1 metre TVD.
    # Derived from: 1000 kg/m3 (fresh water) * 9.80665 m/s2 = 9806.65 Pa/(SG·m)
    GRAVITY = 9806.65

    # Pressure unit keys that imply imperial TVD in feet.
    # All other pressure keys imply metric TVD in metres.
    FIELD_UNITS = {'psi', 'mpsi', 'ppg', 'lb/gal', 'pcf', 'lb/ft3', 'psi/ft'}

    # Conversion factors relative to 1.0 SG (fresh-water baseline).
    # value_in_unit / factor = equivalent SG value.
    # Gradient units (mpa/m, bar/m, kpa/m) represent the pressure exerted per
    # metre of TVD by a 1 SG fluid, derived directly from GRAVITY:
    #   mpa/m = GRAVITY / 1e6,  bar/m = GRAVITY / 1e5,  kpa/m = GRAVITY / 1e3
    SG_CONVERSION_FACTORS = {
        'sg': 1.0,
        'g/cm3': 1.0,
        'ppg': 8.345404,
        'lb/gal': 8.345404,
        'kg/m3': 1000.0,
        'pcf': 62.42796,
        'lb/ft3': 62.42796,
        'psi/ft': 0.433515,
        'mpa/m': 0.00980665,
        'bar/m': 0.0980665,
        'kpa/m': 9.80665,
    }

    # Conversion factors relative to 1.0 Pascal (Pa baseline).
    # value_in_unit * factor = equivalent Pascal value.
    # Reference: https://www.nist.gov/pml/owm/metric-si/unit-conversion/
    PA_CONVERSION_FACTORS = {
        'pa': 1.0,
        'kpa': 1000.0,
        'mpa': 1000000.0,
        'gpa': 1000000000.0,
        'psi': 6894.757293168,
        'mpsi': 6894757293.168,
        'bar': 100000.0,
        'mbar': 100.0,
        'atm': 101325.0,
        'torr': 133.322368421,
        'mmhg': 133.322387415,
        'inhg': 3386.389,
    }

    # Conversion factors relative to 1.0 metre (m baseline).
    # value_in_unit * factor = equivalent metre value.
    # Reference: ISA CCST Conversions document, Table 2.
    METER_CONVERSION_FACTORS = {
        'm': 1.0,
        'meter': 1.0,
        'meters': 1.0,
        'mm': 0.001,
        'cm': 0.01,
        'km': 1000.0,
        'in': 0.0254,
        'inch': 0.0254,
        'inches': 0.0254,
        'ft': 0.3048,
        'foot': 0.3048,
        'feet': 0.3048,
        'yd': 0.9144,
        'yard': 0.9144,
        'mi': 1609.344,
        'mile': 1609.344,
    }

    # Conversion factors relative to 1.0 m/s (velocity baseline).
    # value_in_unit * factor = equivalent m/s value.
    # Reference: ISA CCST Conversions document, Table 12.
    METER_PER_SECOND_CONVERSION_FACTORS = {
        'm/s': 1.0,
        'ft/s': 0.3048,
        'ft/sec': 0.3048,
        'km/h': 1.0 / 3.6,
        'mph': 0.44704,
        'kft/s': 304.8,
    }

    # Conversion factors relative to 1.0 s/m (slowness baseline).
    # value_in_unit * factor = equivalent s/m value.
    SECONDS_PER_METER_CONVERSION_FACTORS = {
        's/m': 1.0,
        's/ft': 3.280839895,
        'us/m': 1e-6,
        'us/meter': 1e-6,
        'us/ft': 3.280839895e-6,
        'us/foot': 3.280839895e-6,
    }


@dataclass(frozen=True)
class UnitConverter:
    """Convert physical quantities between standard industry and SI units.

    Supports equivalent mud weight, pressure, length, and acoustic
    (velocity/slowness) unit conversions used in petroleum geomechanics
    workflows. All conversions route through a well-defined SI baseline value
    (SG, Pascal, metre, m/s or s/m) and back, ensuring a consistent and
    extendable conversion pattern.

    Depth-dependent conversions between absolute pressure and equivalent mud
    weight use the hydrostatic pressure law: P [Pa] = SG * GRAVITY * TVD [m].

    Reference: N/A
    """

    @staticmethod
    def convert_equivalent_mud_weight(value: float, unit_in: str, unit_out: str) -> float:
        """Convert equivalent mud weight or pressure gradient between standard units.

        Conversion routes through the fresh-water SG baseline (1 SG = 1 g/cm3 =
        1000 kg/m3). Includes oilfield density units and pressure-gradient units
        (MPa/m, bar/m, kPa/m, psi/ft) that are all equivalent representations of
        fluid column weight per unit depth.

        Equation type: Ratio scaling via fresh-water SG baseline.

        Applicable for: All standard mud-weight and equivalent-density unit pairs
        used in drilling engineering and geomechanics.

        Reference: N/A

        Supported units: 'sg', 'g/cm3', 'ppg', 'lb/gal', 'kg/m3', 'pcf', 'lb/ft3',
                         'psi/ft', 'mpa/m', 'bar/m', 'kpa/m'

        Args:
            value (float): Mud weight magnitude expressed in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'ppg', 'sg', 'kg/m3'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'ppg', 'sg', 'kg/m3'. Unit: dimensionless

        Returns:
            converted_equivalent_mud_weight (float): Mud weight in the output unit. Unit: unit_out
        """
        if unit_in == unit_out:
            return float(value)

        if unit_in not in UnitConverterConstants.SG_CONVERSION_FACTORS or unit_out not in UnitConverterConstants.SG_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported unit conversion from '{unit_in}' to '{unit_out}'")

        # Convert the input value to the SG baseline
        value_in_sg = value / UnitConverterConstants.SG_CONVERSION_FACTORS[unit_in]

        # Convert the SG baseline value to the requested output unit
        converted_equivalent_mud_weight = value_in_sg * UnitConverterConstants.SG_CONVERSION_FACTORS[unit_out]

        return float(converted_equivalent_mud_weight)

    @staticmethod
    def convert_equivalent_mud_weight_array(
        values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of equivalent mud weight values between standard units.

        Args:
            values (list[float]): Mud weight magnitudes in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'ppg', 'sg', 'kg/m3'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'ppg', 'sg', 'kg/m3'. Unit: dimensionless

        Returns:
            converted_equivalent_mud_weights (list[float]): Mud weight magnitudes
                in the output unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_equivalent_mud_weight(
                value=value,
                unit_in=unit_in,
                unit_out=unit_out,
            )
            for value in values
        ]

    @staticmethod
    def convert_pressure(value: float, unit_in: str, unit_out: str) -> float:
        """Convert pressure between standard SI, metric, and oilfield units via a Pascal baseline.

        Equation type: Ratio scaling via Pascal baseline.

        Applicable for: All standard pressure unit pairs used in petroleum engineering
        and geomechanics.

        Reference: https://www.nist.gov/pml/owm/metric-si/unit-conversion/

        Supported units: 'pa', 'kpa', 'mpa', 'gpa', 'psi', 'mpsi', 'bar', 'mbar',
                         'atm', 'torr', 'mmhg', 'inhg'

        Args:
            value (float): Pressure magnitude expressed in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'psi', 'mpa', 'bar'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'psi', 'mpa', 'bar'. Unit: dimensionless

        Returns:
            converted_pressure (float): Pressure magnitude in the output unit. Unit: unit_out
        """
        if unit_in == unit_out:
            return float(value)

        if unit_in not in UnitConverterConstants.PA_CONVERSION_FACTORS or unit_out not in UnitConverterConstants.PA_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported unit conversion from '{unit_in}' to '{unit_out}'")

        # Convert the input value to the Pascal baseline
        value_in_pascal = value * UnitConverterConstants.PA_CONVERSION_FACTORS[unit_in]

        # Convert the Pascal baseline value to the requested output unit
        converted_pressure = value_in_pascal / UnitConverterConstants.PA_CONVERSION_FACTORS[unit_out]

        return float(converted_pressure)

    @staticmethod
    def convert_pressure_array(
        values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of pressure values between standard SI, metric, and oilfield units.

        Args:
            values (list[float]): Pressure magnitudes in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'psi', 'mpa', 'bar'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'psi', 'mpa', 'bar'. Unit: dimensionless

        Returns:
            converted_pressures (list[float]): Pressure magnitudes in the output unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_pressure(value=value, unit_in=unit_in, unit_out=unit_out)
            for value in values
        ]

    @staticmethod
    def convert_length(value: float, unit_in: str, unit_out: str) -> float:
        """Convert length between metric, imperial, and oilfield units via a metre baseline.

        Equation type: Ratio scaling via metre baseline.

        Applicable for: All standard length unit pairs used in petroleum engineering
        and geomechanics.

        Reference: ISA CCST Conversions document, Table 2.

        Supported units: 'm', 'meter', 'meters', 'mm', 'cm', 'km', 'in', 'inch', 'inches',
                         'ft', 'foot', 'feet', 'yd', 'yard', 'mi', 'mile'

        Args:
            value (float): Length magnitude expressed in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'ft', 'm', 'km'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'ft', 'm', 'km'. Unit: dimensionless

        Returns:
            converted_length (float): Length magnitude in the output unit. Unit: unit_out
        """
        if unit_in == unit_out:
            return float(value)

        if unit_in not in UnitConverterConstants.METER_CONVERSION_FACTORS or unit_out not in UnitConverterConstants.METER_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported unit conversion from '{unit_in}' to '{unit_out}'")

        # Convert the input value to the metre baseline
        value_in_metres = value * UnitConverterConstants.METER_CONVERSION_FACTORS[unit_in]

        # Convert the metre baseline value to the requested output unit
        converted_length = value_in_metres / UnitConverterConstants.METER_CONVERSION_FACTORS[unit_out]

        return float(converted_length)

    @staticmethod
    def convert_length_array(
        values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of length values between metric, imperial, and oilfield units.

        Args:
            values (list[float]): Length magnitudes in the input unit. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'ft', 'm', 'km'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'ft', 'm', 'km'. Unit: dimensionless

        Returns:
            converted_lengths (list[float]): Length magnitudes in the output unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_length(value=value, unit_in=unit_in, unit_out=unit_out)
            for value in values
        ]

    @staticmethod
    def convert_acoustic(value: float, unit_in: str, unit_out: str) -> float:
        """Convert acoustic velocity or slowness between standard oilfield and SI units.

        Handles same-domain conversions (velocity-to-velocity, slowness-to-slowness) and
        cross-domain conversions (velocity-to-slowness and slowness-to-velocity) using
        the reciprocal relationship: slowness [s/m] = 1 / velocity [m/s].

        Equation type: Ratio scaling with reciprocal cross-domain inversion.

        Applicable for: Compressional and shear sonic log data in petroleum geomechanics
        workflows.

        Reference: N/A

        Supported velocity units: 'm/s', 'ft/s', 'ft/sec' (alias), 'km/h', 'mph', 'kft/s'
        Supported slowness units: 's/m', 's/ft', 'us/m', 'us/meter' (alias),
                                   'us/ft', 'us/foot' (alias)

        Args:
            value (float): Acoustic value expressed in the input unit. Must be > 0. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'm/s', 'us/ft'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'm/s', 'us/ft'. Unit: dimensionless

        Returns:
            converted_acoustic (float): Acoustic value in the output unit. Unit: unit_out
        """
        if value <= 0:
            raise ValueError("Acoustic values must be strictly greater than zero.")

        if unit_in == unit_out:
            return float(value)

        # Classify input and output units by domain (velocity or slowness)
        input_is_velocity = unit_in in UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS
        input_is_slowness = unit_in in UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS
        output_is_velocity = unit_out in UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS
        output_is_slowness = unit_out in UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS

        if not (input_is_velocity or input_is_slowness) or not (output_is_velocity or output_is_slowness):
            raise ValueError(f"Unsupported unit conversion from '{unit_in}' to '{unit_out}'.")

        # Velocity to Velocity: scale via m/s baseline
        if input_is_velocity and output_is_velocity:
            velocity_meters_per_second = value * UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS[unit_in]
            converted_acoustic = velocity_meters_per_second / UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS[unit_out]

        # Slowness to Slowness: scale via s/m baseline
        elif input_is_slowness and output_is_slowness:
            slowness_seconds_per_meter = value * UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS[unit_in]
            converted_acoustic = slowness_seconds_per_meter / UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS[unit_out]

        # Velocity to Slowness: convert to m/s baseline, then invert to s/m, then scale
        elif input_is_velocity and output_is_slowness:
            velocity_meters_per_second = value * UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS[unit_in]
            slowness_seconds_per_meter = 1.0 / velocity_meters_per_second
            converted_acoustic = slowness_seconds_per_meter / UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS[unit_out]

        # Slowness to Velocity: convert to s/m baseline, then invert to m/s, then scale
        else:
            slowness_seconds_per_meter = value * UnitConverterConstants.SECONDS_PER_METER_CONVERSION_FACTORS[unit_in]
            velocity_meters_per_second = 1.0 / slowness_seconds_per_meter
            converted_acoustic = velocity_meters_per_second / UnitConverterConstants.METER_PER_SECOND_CONVERSION_FACTORS[unit_out]

        return float(converted_acoustic)

    @staticmethod
    def convert_acoustic_array(
        values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of acoustic velocity or slowness values between standard units.

        Args:
            values (list[float]): Acoustic values in the input unit. All values must be > 0. Unit: unit_in
            unit_in (str): Source unit key, e.g. 'm/s', 'us/ft'. Unit: dimensionless
            unit_out (str): Target unit key, e.g. 'm/s', 'us/ft'. Unit: dimensionless

        Returns:
            converted_acoustics (list[float]): Acoustic values in the output unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_acoustic(value=value, unit_in=unit_in, unit_out=unit_out)
            for value in values
        ]

    @staticmethod
    def convert_from_pressure_to_equivalent_mud_weight(
        pressure_value: float,
        tvd: float,
        unit_in: str,
        unit_out: str,
    ) -> float:
        """Convert an absolute fluid pressure at a given depth to equivalent mud weight.

        Uses the hydrostatic pressure law rearranged for SG:
            SG = P [Pa] / (GRAVITY [Pa/(SG·m)] * TVD [m])

        TVD is automatically interpreted in the correct depth unit based on the
        input pressure system: field-unit inputs ('psi', 'mpsi') imply TVD in feet;
        all SI and metric inputs ('mpa', 'kpa', 'bar', 'pa', etc.) imply TVD in metres.

        Equation type: Hydrostatic pressure law, rearranged (SG = P / (GRAVITY * TVD_m)).

        Applicable for: Converting pore pressure, overburden, or horizontal stress
        magnitudes to equivalent mud weight in petroleum geomechanics workflows.

        Reference: N/A

        Args:
            pressure_value (float): Fluid pressure magnitude. Unit: unit_in
            tvd (float): True vertical depth. Unit: ft if unit_in is 'psi' or 'mpsi';
                         m for all SI/metric inputs ('pa', 'mpa', 'kpa', 'bar', etc.).
                         Must be > 0.
            unit_in (str): Source pressure unit key, e.g. 'psi', 'mpa', 'kpa', 'bar', 'pa'.
                           Unit: dimensionless
            unit_out (str): Target equivalent mud weight unit key, e.g. 'ppg', 'sg',
                            'kg/m3', 'mpa/m'. Unit: dimensionless

        Returns:
            converted_equivalent_mud_weight (float): Equivalent mud weight in the output
                unit. Unit: unit_out
        """
        if tvd <= 0:
            raise ValueError("True vertical depth (tvd) must be greater than zero.")
        if unit_in not in UnitConverterConstants.PA_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported input pressure unit: '{unit_in}'")
        if unit_out not in UnitConverterConstants.SG_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported output equivalent mud weight unit: '{unit_out}'")

        # Resolve TVD to metres: field-unit pressure inputs assume TVD is in feet
        field_pressure_units = {'psi', 'mpsi'}
        tvd_metres = tvd * 0.3048 if unit_in in field_pressure_units else float(tvd)

        # Convert the input pressure to the Pascal baseline
        pressure_in_pascal = pressure_value * UnitConverterConstants.PA_CONVERSION_FACTORS[unit_in]

        # Derive specific gravity: SG = P_pa / (GRAVITY * TVD_m)
        value_in_sg = pressure_in_pascal / (UnitConverterConstants.GRAVITY * tvd_metres)

        # Scale SG to the requested equivalent mud weight output unit
        converted_equivalent_mud_weight = value_in_sg * UnitConverterConstants.SG_CONVERSION_FACTORS[unit_out]

        return float(converted_equivalent_mud_weight)

    @staticmethod
    def convert_from_pressure_to_equivalent_mud_weight_array(
        pressure_values: list[float],
        tvd_values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of absolute pressures at corresponding depths to equivalent mud weights.

        Args:
            pressure_values (list[float]): Fluid pressure magnitudes. Unit: unit_in
            tvd_values (list[float]): True vertical depths paired with each pressure value.
                         Unit: ft if unit_in is 'psi' or 'mpsi'; m otherwise. All values > 0.
            unit_in (str): Source pressure unit key, e.g. 'psi', 'mpa'. Unit: dimensionless
            unit_out (str): Target equivalent mud weight unit key, e.g. 'ppg', 'sg'.
                            Unit: dimensionless

        Returns:
            converted_equivalent_mud_weights (list[float]): Equivalent mud weight values
                in the output unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_from_pressure_to_equivalent_mud_weight(
                pressure_value=pressure,
                tvd=depth,
                unit_in=unit_in,
                unit_out=unit_out,
            )
            for pressure, depth in zip(pressure_values, tvd_values)
        ]

    @staticmethod
    def convert_equivalent_mud_weight_to_pressure(
        mud_weight_value: float,
        tvd: float,
        unit_in: str,
        unit_out: str,
    ) -> float:
        """Convert an equivalent mud weight at a given depth to absolute hydrostatic pressure.

        Uses the hydrostatic pressure law:
            P [Pa] = SG * GRAVITY [Pa/(SG·m)] * TVD [m]

        TVD is automatically interpreted in the correct depth unit based on the
        output pressure system: field-unit outputs ('psi', 'mpsi') imply TVD in feet;
        all SI and metric outputs ('mpa', 'kpa', 'bar', 'pa', etc.) imply TVD in metres.

        Equation type: Hydrostatic pressure law (P = SG * GRAVITY * TVD_m).

        Applicable for: Converting mud weight or equivalent density to hydrostatic
        pressure in petroleum geomechanics workflows.

        Reference: N/A

        Args:
            mud_weight_value (float): Equivalent mud weight magnitude. Unit: unit_in
            tvd (float): True vertical depth. Unit: ft if unit_out is 'psi' or 'mpsi';
                         m for all SI/metric outputs ('pa', 'mpa', 'kpa', 'bar', etc.).
                         Must be > 0.
            unit_in (str): Source equivalent mud weight unit key, e.g. 'ppg', 'sg',
                           'kg/m3'. Unit: dimensionless
            unit_out (str): Target pressure unit key, e.g. 'psi', 'mpa', 'kpa', 'pa'.
                            Unit: dimensionless

        Returns:
            converted_pressure (float): Hydrostatic pressure in the output unit. Unit: unit_out
        """
        if tvd <= 0:
            raise ValueError("True vertical depth (tvd) must be greater than zero.")
        if unit_in not in UnitConverterConstants.SG_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported input equivalent mud weight unit: '{unit_in}'")
        if unit_out not in UnitConverterConstants.PA_CONVERSION_FACTORS:
            raise ValueError(f"Unsupported output pressure unit: '{unit_out}'")

        # Resolve TVD to metres: field-unit pressure outputs assume TVD is in feet
        field_pressure_units = {'psi', 'mpsi'}
        tvd_metres = tvd * 0.3048 if unit_out in field_pressure_units else float(tvd)

        # Normalise mud weight to the SG baseline
        value_in_sg = mud_weight_value / UnitConverterConstants.SG_CONVERSION_FACTORS[unit_in]

        # Compute absolute hydrostatic pressure in Pascal: P = SG * GRAVITY * TVD_m
        pressure_in_pascal = value_in_sg * UnitConverterConstants.GRAVITY * tvd_metres

        # Convert Pascal baseline to the requested output pressure unit
        converted_pressure = pressure_in_pascal / UnitConverterConstants.PA_CONVERSION_FACTORS[unit_out]

        return float(converted_pressure)

    @staticmethod
    def convert_equivalent_mud_weight_to_pressure_array(
        mud_weight_values: list[float],
        tvd_values: list[float],
        unit_in: str,
        unit_out: str,
    ) -> list[float]:
        """Convert a list of equivalent mud weights at corresponding depths to absolute pressures.

        Args:
            mud_weight_values (list[float]): Equivalent mud weight magnitudes. Unit: unit_in
            tvd_values (list[float]): True vertical depths paired with each mud weight value.
                         Unit: ft if unit_out is 'psi' or 'mpsi'; m otherwise. All values > 0.
            unit_in (str): Source equivalent mud weight unit key, e.g. 'ppg', 'sg'.
                           Unit: dimensionless
            unit_out (str): Target pressure unit key, e.g. 'psi', 'mpa'. Unit: dimensionless

        Returns:
            converted_pressures (list[float]): Hydrostatic pressure values in the output
                unit. Unit: unit_out
        """
        return [
            UnitConverter.convert_equivalent_mud_weight_to_pressure(
                mud_weight_value=mud_weight,
                tvd=depth,
                unit_in=unit_in,
                unit_out=unit_out,
            )
            for mud_weight, depth in zip(mud_weight_values, tvd_values)
        ]
