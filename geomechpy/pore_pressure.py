import math
from dataclasses import dataclass


class PorePressureCalculation:
    """Computation of the Pore Pressure using gradient based methods.

    Reference:
       Zhang, Jon Jincai. Applied petroleum geomechanics. Vol. 1. Cambridge: Gulf Professional Publishing, 2019. Chapter 6.1"""

    @staticmethod
    def calculate_pore_pressure_onshore(tvd: float, formation_pore_pressure_gradient: float = 0.47, air_gap: float = 0.0) -> float:
        """Calculates pore pressure from tvd and pore pressure gradient in onshore setting.

        Args:
            tvd (float): True Vertical Depth. Unit: Depth Unit [ft]
            formation_pore_pressure_gradient (float): Pore pressure depth gradient. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47
            air_gap (float): Distance from Drill Floor to Ground Level. Usually reported as Kelly bushing (KB) or Elevation Ground Level. Unit: Depth Unit [ft]. Defaults to 0.0

        Returns:
            pore_pressure (float): Pore pressure for onshore setting. Unit: Pressure Unit [psi]"""
        air_gradient = 0.0004
        air_pressure = air_gradient * air_gap
        if tvd >= 0 and tvd < air_gap:
            pore_pressure = air_gradient * tvd
        else:
            pore_pressure = air_pressure + formation_pore_pressure_gradient * (tvd - air_gap)

        return pore_pressure

    @staticmethod
    def calculate_pore_pressure_offshore(tvd: float, formation_pore_pressure_gradient: float = 0.47, air_gap: float = 0.0, water_depth: float = 0.0, sea_water_pressure_gradient: float = 0.47) -> float:
        """Calculates pore pressure from tvd and pore pressure gradient in offshore setting.

        Args:
            tvd (float): True Vertical Depth. Unit: Depth Unit [ft]
            formation_pore_pressure_gradient (float): Pore pressure depth gradient. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47
            air_gap (float): Distance from Drill Floor to mean sea level. Usually reported as Kelly bushing (KB). Unit: Depth Unit [ft]. Defaults to 0.0 ft
            water_depth (float): Water Depth measured from the mean sea level to sea bottom at well location. Unit: Depth Unit [ft]. Defaults to 0.0 ft
            sea_water_pressure_gradient (float): Water gradient of the sea water. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47 psi/ft

        Returns:
            pore_pressure (float): Pore pressure for offshore setting. Unit: Pressure Unit [psi]"""
        air_gradient = 0.0004
        air_pressure = air_gradient * air_gap

        water_pressure = sea_water_pressure_gradient * water_depth

        if tvd >= 0 and tvd < air_gap:
            pore_pressure = air_gradient * tvd
        elif tvd >= air_gap and tvd <= (air_gap + water_depth):
            pore_pressure = air_pressure + sea_water_pressure_gradient * (tvd - air_gap)
        else:
            pore_pressure = air_pressure + water_pressure + formation_pore_pressure_gradient * (tvd - water_depth - air_gap)

        return pore_pressure

    @staticmethod
    def calculate_pore_pressure_onshore_array(tvd: list[float], formation_pore_pressure_gradient: float = 0.47, air_gap: float = 0.0) -> list[float]:
        """Calculates pore pressure for an array of tvd values in onshore setting.

        Args:
            tvd (list[float]): True Vertical Depth values. Unit: Depth Unit [ft]
            formation_pore_pressure_gradient (float): Pore pressure depth gradient. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47
            air_gap (float): Distance from Drill Floor to Ground Level. Usually reported as Kelly bushing (KB) or Elevation Ground Level. Unit: Depth Unit [ft]. Defaults to 0.0

        Returns:
            pore_pressure (list[float]): Pore pressure values for onshore setting. Unit: Pressure Unit [psi]"""
        return [
            PorePressureCalculation.calculate_pore_pressure_onshore(
                tvd=value,
                formation_pore_pressure_gradient=formation_pore_pressure_gradient,
                air_gap=air_gap,
            )
            for value in tvd
        ]

    @staticmethod
    def calculate_pore_pressure_offshore_array(tvd: list[float], formation_pore_pressure_gradient: float = 0.47, air_gap: float = 0.0, water_depth: float = 0.0, sea_water_pressure_gradient: float = 0.47) -> list[float]:
        """Calculates pore pressure for an array of tvd values in offshore setting.

        Args:
            tvd (list[float]): True Vertical Depth values. Unit: Depth Unit [ft]
            formation_pore_pressure_gradient (float): Pore pressure depth gradient. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47
            air_gap (float): Distance from Drill Floor to mean sea level. Usually reported as Kelly bushing (KB). Unit: Depth Unit [ft]. Defaults to 0.0 ft
            water_depth (float): Water Depth measured from the mean sea level to sea bottom at well location. Unit: Depth Unit [ft]. Defaults to 0.0 ft
            sea_water_pressure_gradient (float): Water gradient of the sea water. Unit: Depth Gradient Unit [psi/ft]. Defaults to 0.47 psi/ft

        Returns:
            pore_pressure (list[float]): Pore pressure values for offshore setting. Unit: Pressure Unit [psi]"""
        return [
            PorePressureCalculation.calculate_pore_pressure_offshore(
                tvd=value,
                formation_pore_pressure_gradient=formation_pore_pressure_gradient,
                air_gap=air_gap,
                water_depth=water_depth,
                sea_water_pressure_gradient=sea_water_pressure_gradient,
            )
            for value in tvd
        ]


class EffectiveStressPorePressure:
    """Computation of pore pressure through the effective stress framework.

    The effective stress framework estimates the vertical effective stress from
    a compaction-sensitive log (velocity, resistivity, sonic slowness or porosity),
    then recovers pore pressure with the Terzaghi relation
    (pore_pressure = overburden_stress - effective_vertical_stress). Each log based
    method requires a normal compaction trend as the undercompacted (hydrostatic)
    reference, so the matching normal trend line calculations are provided alongside.

    Reference:
        Zhang, Jon Jincai. Applied petroleum geomechanics. Vol. 1. Cambridge: Gulf Professional Publishing, 2019. Chapter 6.
    """

    @staticmethod
    def calculate_effective_stress_bowers(compressional_velocity_ft_per_s: float, mudline_velocity_ft_per_s: float, bowers_coefficient_a: float, bowers_exponent_b: float) -> float:
        """Calculate the vertical effective stress from compressional velocity using the Bowers law.

        Equation type: Power law inversion (sveff = ((vp - vp0) / A)**(1 / B))

        Applicable for: Shales and generic clastic rocks in the virgin (loading) compaction regime.

        Reference:
            Bowers, G. L., 1995. Pore pressure estimation from velocity data: accounting for overpressure mechanisms besides undercompaction. SPE Drilling and Completion, 10(02), pp. 89-95.

        Args:
            compressional_velocity_ft_per_s (float): Measured compressional (P-wave) velocity. Unit: Velocity Unit [ft/s]
            mudline_velocity_ft_per_s (float): Compressional velocity at the mudline (sediment surface). Unit: Velocity Unit [ft/s]
            bowers_coefficient_a (float): Bowers virgin curve multiplier coefficient A. Unit: Velocity over Pressure Unit [ft/s/psi**B]
            bowers_exponent_b (float): Bowers virgin curve exponent B. Unit: Dimensionless

        Returns:
            effective_vertical_stress_psi (float): Vertical effective stress. Unit: Pressure Unit [psi]"""
        velocity_increment_ft_per_s = compressional_velocity_ft_per_s - mudline_velocity_ft_per_s
        effective_vertical_stress_psi = (velocity_increment_ft_per_s / bowers_coefficient_a) ** (1.0 / bowers_exponent_b)

        return float(effective_vertical_stress_psi)

    @staticmethod
    def calculate_effective_stress_eaton_resistivity(observed_resistivity_ohm_m: float, normal_resistivity_ohm_m: float, normal_effective_vertical_stress_psi: float, eaton_resistivity_exponent: float = 1.2) -> float:
        """Calculate the vertical effective stress from resistivity using the Eaton resistivity method.

        Equation type: Power law (sveff = (res / resnorm)**n * sveffnorm)

        Applicable for: Shales where overpressure is generated by undercompaction.

        Reference:
            Eaton, B. A., 1975. The equation for geopressure prediction from well logs. SPE 5544, Fall Meeting of the Society of Petroleum Engineers of AIME.

        Args:
            observed_resistivity_ohm_m (float): Measured formation resistivity. Unit: Resistivity Unit [ohm.m]
            normal_resistivity_ohm_m (float): Resistivity read from the normal compaction trend at the same depth. Unit: Resistivity Unit [ohm.m]
            normal_effective_vertical_stress_psi (float): Vertical effective stress under hydrostatic (normally pressured) conditions. Unit: Pressure Unit [psi]
            eaton_resistivity_exponent (float): Eaton resistivity exponent. Unit: Dimensionless. Defaults to 1.2

        Returns:
            effective_vertical_stress_psi (float): Vertical effective stress. Unit: Pressure Unit [psi]"""
        resistivity_ratio = observed_resistivity_ohm_m / normal_resistivity_ohm_m
        effective_vertical_stress_psi = resistivity_ratio ** eaton_resistivity_exponent * normal_effective_vertical_stress_psi

        return float(effective_vertical_stress_psi)

    @staticmethod
    def calculate_effective_stress_eaton_sonic(observed_sonic_slowness_us_per_ft: float, normal_sonic_slowness_us_per_ft: float, normal_effective_vertical_stress_psi: float, eaton_sonic_exponent: float = 3.0) -> float:
        """Calculate the vertical effective stress from sonic slowness using the Eaton sonic method.

        Equation type: Power law (sveff = (dtnorm / dt)**n * sveffnorm)

        Applicable for: Shales where overpressure is generated by undercompaction.

        Reference:
            Eaton, B. A., 1975. The equation for geopressure prediction from well logs. SPE 5544, Fall Meeting of the Society of Petroleum Engineers of AIME.

        Args:
            observed_sonic_slowness_us_per_ft (float): Measured sonic slowness (compressional transit time). Unit: Slowness Unit [us/ft]
            normal_sonic_slowness_us_per_ft (float): Sonic slowness read from the normal compaction trend at the same depth. Unit: Slowness Unit [us/ft]
            normal_effective_vertical_stress_psi (float): Vertical effective stress under hydrostatic (normally pressured) conditions. Unit: Pressure Unit [psi]
            eaton_sonic_exponent (float): Eaton sonic exponent. Unit: Dimensionless. Defaults to 3.0

        Returns:
            effective_vertical_stress_psi (float): Vertical effective stress. Unit: Pressure Unit [psi]"""
        sonic_slowness_ratio = normal_sonic_slowness_us_per_ft / observed_sonic_slowness_us_per_ft
        effective_vertical_stress_psi = sonic_slowness_ratio ** eaton_sonic_exponent * normal_effective_vertical_stress_psi

        return float(effective_vertical_stress_psi)

    @staticmethod
    def calculate_effective_stress_traugott(observed_porosity_fraction: float, normal_porosity_fraction: float, normal_effective_vertical_stress_psi: float, traugott_exponent: float = 5.5) -> float:
        """Calculate the vertical effective stress from porosity using the Traugott porosity compaction law.

        Equation type: Power law (sveff = sveffnorm * ((1 - por) / (1 - pornorm))**n)

        Applicable for: Shales and mudrocks whose porosity follows a compaction trend.

        Reference:
            Traugott, M., 1997. Pore/fracture pressure determinations in deep water. World Oil, 218(8), pp. 68-70.

        Args:
            observed_porosity_fraction (float): Measured porosity. Unit: Fraction [v/v]
            normal_porosity_fraction (float): Porosity read from the normal compaction trend at the same depth. Unit: Fraction [v/v]
            normal_effective_vertical_stress_psi (float): Vertical effective stress under hydrostatic (normally pressured) conditions. Unit: Pressure Unit [psi]
            traugott_exponent (float): Traugott porosity compaction exponent. Unit: Dimensionless. Defaults to 5.5

        Returns:
            effective_vertical_stress_psi (float): Vertical effective stress. Unit: Pressure Unit [psi]"""
        solidity_ratio = (1.0 - observed_porosity_fraction) / (1.0 - normal_porosity_fraction)
        effective_vertical_stress_psi = normal_effective_vertical_stress_psi * solidity_ratio ** traugott_exponent

        return float(effective_vertical_stress_psi)

    @staticmethod
    def calculate_effective_stress_dutta(porosity_fraction: float, reference_effective_vertical_stress_psi: float, dutta_exponent_beta: float) -> float:
        """Calculate the vertical effective stress from porosity using the Dutta effective load velocity inversion.

        Equation type: Exponential load inversion (vr = por / (1 - por); sveff = sveff0 * exp(vr**beta))

        Applicable for: Shales characterised with a porosity (void ratio) to effective load transform.

        Reference:
            Dutta, N. C., 2002. Geopressure prediction using seismic data: Current status and the road ahead. Geophysics, 67(6), pp. 2012-2041.

        Args:
            porosity_fraction (float): Measured porosity. Unit: Fraction [v/v]
            reference_effective_vertical_stress_psi (float): Reference vertical effective stress scaling factor sveff0. Unit: Pressure Unit [psi]
            dutta_exponent_beta (float): Dutta void ratio exponent beta. Unit: Dimensionless

        Returns:
            effective_vertical_stress_psi (float): Vertical effective stress. Unit: Pressure Unit [psi]"""
        void_ratio = porosity_fraction / (1.0 - porosity_fraction)
        effective_vertical_stress_psi = reference_effective_vertical_stress_psi * math.exp(void_ratio ** dutta_exponent_beta)

        return float(effective_vertical_stress_psi)

    @staticmethod
    def calculate_normal_compaction_trend_resistivity(tvd: float, surface_resistivity_ohm_m: float, resistivity_compaction_coefficient_per_ft: float) -> float:
        """Calculate the normal compaction trend resistivity at a given depth.

        Equation type: Exponential law (resnorm = res0 * exp(b * tvd))

        Applicable for: Shale resistivity baselines that increase with compaction.

        Reference:
            Eaton, B. A., 1975. The equation for geopressure prediction from well logs. SPE 5544, Fall Meeting of the Society of Petroleum Engineers of AIME.

        Args:
            tvd (float): True Vertical Depth. Unit: Depth Unit [ft]
            surface_resistivity_ohm_m (float): Normal trend resistivity extrapolated to the surface. Unit: Resistivity Unit [ohm.m]
            resistivity_compaction_coefficient_per_ft (float): Exponential growth coefficient of the resistivity trend. Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_resistivity_ohm_m (float): Resistivity on the normal compaction trend at the given depth. Unit: Resistivity Unit [ohm.m]"""
        normal_resistivity_ohm_m = surface_resistivity_ohm_m * math.exp(resistivity_compaction_coefficient_per_ft * tvd)

        return float(normal_resistivity_ohm_m)

    @staticmethod
    def calculate_normal_compaction_trend_sonic_slowness(tvd: float, matrix_sonic_slowness_us_per_ft: float, mudline_sonic_slowness_us_per_ft: float, sonic_compaction_coefficient_per_ft: float) -> float:
        """Calculate the normal compaction trend sonic slowness at a given depth.

        Equation type: Exponential decline (dtnorm = dtmatrix + (dtmudline - dtmatrix) * exp(-c * tvd))

        Applicable for: Shale sonic slowness baselines that decline towards the matrix value with compaction.

        Reference:
            Zhang, J., 2011. Pore pressure prediction from well logs: Methods, modeling and uncertainty. Earth-Science Reviews, 108(1-2), pp. 50-63.

        Args:
            tvd (float): True Vertical Depth. Unit: Depth Unit [ft]
            matrix_sonic_slowness_us_per_ft (float): Sonic slowness of the rock matrix at infinite compaction. Unit: Slowness Unit [us/ft]
            mudline_sonic_slowness_us_per_ft (float): Sonic slowness at the mudline (sediment surface). Unit: Slowness Unit [us/ft]
            sonic_compaction_coefficient_per_ft (float): Exponential decline coefficient of the sonic slowness trend. Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_sonic_slowness_us_per_ft (float): Sonic slowness on the normal compaction trend at the given depth. Unit: Slowness Unit [us/ft]"""
        slowness_above_matrix_us_per_ft = mudline_sonic_slowness_us_per_ft - matrix_sonic_slowness_us_per_ft
        normal_sonic_slowness_us_per_ft = matrix_sonic_slowness_us_per_ft + slowness_above_matrix_us_per_ft * math.exp(-sonic_compaction_coefficient_per_ft * tvd)

        return float(normal_sonic_slowness_us_per_ft)

    @staticmethod
    def calculate_normal_compaction_trend_porosity(tvd: float, surface_porosity_fraction: float, porosity_compaction_coefficient_per_ft: float) -> float:
        """Calculate the normal compaction trend porosity at a given depth using Athy law.

        Equation type: Exponential decline (pornorm = por0 * exp(-c * tvd))

        Applicable for: Shale porosity baselines that decline with compaction.

        Reference:
            Athy, L. F., 1930. Density, porosity, and compaction of sedimentary rocks. AAPG Bulletin, 14(1), pp. 1-24.

        Args:
            tvd (float): True Vertical Depth. Unit: Depth Unit [ft]
            surface_porosity_fraction (float): Normal trend porosity extrapolated to the surface. Unit: Fraction [v/v]
            porosity_compaction_coefficient_per_ft (float): Exponential decline coefficient of the porosity trend (Athy coefficient). Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_porosity_fraction (float): Porosity on the normal compaction trend at the given depth. Unit: Fraction [v/v]"""
        normal_porosity_fraction = surface_porosity_fraction * math.exp(-porosity_compaction_coefficient_per_ft * tvd)

        return float(normal_porosity_fraction)

    @staticmethod
    def calculate_pore_pressure_from_effective_stress(overburden_stress_psi: float, effective_vertical_stress_psi: float) -> float:
        """Calculate pore pressure from overburden and effective vertical stress using the Terzaghi relation.

        Equation type: Linear law (pore_pressure = overburden_stress - effective_vertical_stress)

        Applicable for: Vertical effective stress framework with a Biot coefficient of one.

        Reference:
            Terzaghi, K., 1943. Theoretical Soil Mechanics. John Wiley and Sons, New York.

        Args:
            overburden_stress_psi (float): Total vertical (overburden) stress at the depth of interest. Unit: Pressure Unit [psi]
            effective_vertical_stress_psi (float): Vertical effective stress at the depth of interest. Unit: Pressure Unit [psi]

        Returns:
            pore_pressure_psi (float): Pore pressure. Unit: Pressure Unit [psi]"""
        pore_pressure_psi = overburden_stress_psi - effective_vertical_stress_psi

        return float(pore_pressure_psi)

    @staticmethod
    def calculate_effective_stress_bowers_array(compressional_velocity_ft_per_s: list[float], mudline_velocity_ft_per_s: float, bowers_coefficient_a: float, bowers_exponent_b: float) -> list[float]:
        """Calculate the vertical effective stress for an array of compressional velocity values using the Bowers law.

        Args:
            compressional_velocity_ft_per_s (list[float]): Measured compressional (P-wave) velocity values. Unit: Velocity Unit [ft/s]
            mudline_velocity_ft_per_s (float): Compressional velocity at the mudline (sediment surface). Unit: Velocity Unit [ft/s]
            bowers_coefficient_a (float): Bowers virgin curve multiplier coefficient A. Unit: Velocity over Pressure Unit [ft/s/psi**B]
            bowers_exponent_b (float): Bowers virgin curve exponent B. Unit: Dimensionless

        Returns:
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_effective_stress_bowers(
                compressional_velocity_ft_per_s=velocity_value,
                mudline_velocity_ft_per_s=mudline_velocity_ft_per_s,
                bowers_coefficient_a=bowers_coefficient_a,
                bowers_exponent_b=bowers_exponent_b,
            )
            for velocity_value in compressional_velocity_ft_per_s
        ]

    @staticmethod
    def calculate_effective_stress_eaton_resistivity_array(observed_resistivity_ohm_m: list[float], normal_resistivity_ohm_m: list[float], normal_effective_vertical_stress_psi: list[float], eaton_resistivity_exponent: float = 1.2) -> list[float]:
        """Calculate the vertical effective stress for an array of resistivity values using the Eaton resistivity method.

        Args:
            observed_resistivity_ohm_m (list[float]): Measured formation resistivity values. Unit: Resistivity Unit [ohm.m]
            normal_resistivity_ohm_m (list[float]): Resistivity values read from the normal compaction trend at the matching depths. Unit: Resistivity Unit [ohm.m]
            normal_effective_vertical_stress_psi (list[float]): Hydrostatic vertical effective stress values at the matching depths. Unit: Pressure Unit [psi]
            eaton_resistivity_exponent (float): Eaton resistivity exponent. Unit: Dimensionless. Defaults to 1.2

        Returns:
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_effective_stress_eaton_resistivity(
                observed_resistivity_ohm_m=observed_value,
                normal_resistivity_ohm_m=normal_value,
                normal_effective_vertical_stress_psi=normal_stress_value,
                eaton_resistivity_exponent=eaton_resistivity_exponent,
            )
            for observed_value, normal_value, normal_stress_value in zip(observed_resistivity_ohm_m, normal_resistivity_ohm_m, normal_effective_vertical_stress_psi)
        ]

    @staticmethod
    def calculate_effective_stress_eaton_sonic_array(observed_sonic_slowness_us_per_ft: list[float], normal_sonic_slowness_us_per_ft: list[float], normal_effective_vertical_stress_psi: list[float], eaton_sonic_exponent: float = 3.0) -> list[float]:
        """Calculate the vertical effective stress for an array of sonic slowness values using the Eaton sonic method.

        Args:
            observed_sonic_slowness_us_per_ft (list[float]): Measured sonic slowness values. Unit: Slowness Unit [us/ft]
            normal_sonic_slowness_us_per_ft (list[float]): Sonic slowness values read from the normal compaction trend at the matching depths. Unit: Slowness Unit [us/ft]
            normal_effective_vertical_stress_psi (list[float]): Hydrostatic vertical effective stress values at the matching depths. Unit: Pressure Unit [psi]
            eaton_sonic_exponent (float): Eaton sonic exponent. Unit: Dimensionless. Defaults to 3.0

        Returns:
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_effective_stress_eaton_sonic(
                observed_sonic_slowness_us_per_ft=observed_value,
                normal_sonic_slowness_us_per_ft=normal_value,
                normal_effective_vertical_stress_psi=normal_stress_value,
                eaton_sonic_exponent=eaton_sonic_exponent,
            )
            for observed_value, normal_value, normal_stress_value in zip(observed_sonic_slowness_us_per_ft, normal_sonic_slowness_us_per_ft, normal_effective_vertical_stress_psi)
        ]

    @staticmethod
    def calculate_effective_stress_traugott_array(observed_porosity_fraction: list[float], normal_porosity_fraction: list[float], normal_effective_vertical_stress_psi: list[float], traugott_exponent: float = 5.5) -> list[float]:
        """Calculate the vertical effective stress for an array of porosity values using the Traugott porosity compaction law.

        Args:
            observed_porosity_fraction (list[float]): Measured porosity values. Unit: Fraction [v/v]
            normal_porosity_fraction (list[float]): Porosity values read from the normal compaction trend at the matching depths. Unit: Fraction [v/v]
            normal_effective_vertical_stress_psi (list[float]): Hydrostatic vertical effective stress values at the matching depths. Unit: Pressure Unit [psi]
            traugott_exponent (float): Traugott porosity compaction exponent. Unit: Dimensionless. Defaults to 5.5

        Returns:
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_effective_stress_traugott(
                observed_porosity_fraction=observed_value,
                normal_porosity_fraction=normal_value,
                normal_effective_vertical_stress_psi=normal_stress_value,
                traugott_exponent=traugott_exponent,
            )
            for observed_value, normal_value, normal_stress_value in zip(observed_porosity_fraction, normal_porosity_fraction, normal_effective_vertical_stress_psi)
        ]

    @staticmethod
    def calculate_effective_stress_dutta_array(porosity_fraction: list[float], reference_effective_vertical_stress_psi: float, dutta_exponent_beta: float) -> list[float]:
        """Calculate the vertical effective stress for an array of porosity values using the Dutta effective load velocity inversion.

        Args:
            porosity_fraction (list[float]): Measured porosity values. Unit: Fraction [v/v]
            reference_effective_vertical_stress_psi (float): Reference vertical effective stress scaling factor sveff0. Unit: Pressure Unit [psi]
            dutta_exponent_beta (float): Dutta void ratio exponent beta. Unit: Dimensionless

        Returns:
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_effective_stress_dutta(
                porosity_fraction=porosity_value,
                reference_effective_vertical_stress_psi=reference_effective_vertical_stress_psi,
                dutta_exponent_beta=dutta_exponent_beta,
            )
            for porosity_value in porosity_fraction
        ]

    @staticmethod
    def calculate_normal_compaction_trend_resistivity_array(tvd: list[float], surface_resistivity_ohm_m: float, resistivity_compaction_coefficient_per_ft: float) -> list[float]:
        """Calculate the normal compaction trend resistivity for an array of depth values.

        Args:
            tvd (list[float]): True Vertical Depth values. Unit: Depth Unit [ft]
            surface_resistivity_ohm_m (float): Normal trend resistivity extrapolated to the surface. Unit: Resistivity Unit [ohm.m]
            resistivity_compaction_coefficient_per_ft (float): Exponential growth coefficient of the resistivity trend. Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_resistivity_ohm_m (list[float]): Resistivity values on the normal compaction trend. Unit: Resistivity Unit [ohm.m]"""
        return [
            EffectiveStressPorePressure.calculate_normal_compaction_trend_resistivity(
                tvd=value,
                surface_resistivity_ohm_m=surface_resistivity_ohm_m,
                resistivity_compaction_coefficient_per_ft=resistivity_compaction_coefficient_per_ft,
            )
            for value in tvd
        ]

    @staticmethod
    def calculate_normal_compaction_trend_sonic_slowness_array(tvd: list[float], matrix_sonic_slowness_us_per_ft: float, mudline_sonic_slowness_us_per_ft: float, sonic_compaction_coefficient_per_ft: float) -> list[float]:
        """Calculate the normal compaction trend sonic slowness for an array of depth values.

        Args:
            tvd (list[float]): True Vertical Depth values. Unit: Depth Unit [ft]
            matrix_sonic_slowness_us_per_ft (float): Sonic slowness of the rock matrix at infinite compaction. Unit: Slowness Unit [us/ft]
            mudline_sonic_slowness_us_per_ft (float): Sonic slowness at the mudline (sediment surface). Unit: Slowness Unit [us/ft]
            sonic_compaction_coefficient_per_ft (float): Exponential decline coefficient of the sonic slowness trend. Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_sonic_slowness_us_per_ft (list[float]): Sonic slowness values on the normal compaction trend. Unit: Slowness Unit [us/ft]"""
        return [
            EffectiveStressPorePressure.calculate_normal_compaction_trend_sonic_slowness(
                tvd=value,
                matrix_sonic_slowness_us_per_ft=matrix_sonic_slowness_us_per_ft,
                mudline_sonic_slowness_us_per_ft=mudline_sonic_slowness_us_per_ft,
                sonic_compaction_coefficient_per_ft=sonic_compaction_coefficient_per_ft,
            )
            for value in tvd
        ]

    @staticmethod
    def calculate_normal_compaction_trend_porosity_array(tvd: list[float], surface_porosity_fraction: float, porosity_compaction_coefficient_per_ft: float) -> list[float]:
        """Calculate the normal compaction trend porosity for an array of depth values using Athy law.

        Args:
            tvd (list[float]): True Vertical Depth values. Unit: Depth Unit [ft]
            surface_porosity_fraction (float): Normal trend porosity extrapolated to the surface. Unit: Fraction [v/v]
            porosity_compaction_coefficient_per_ft (float): Exponential decline coefficient of the porosity trend (Athy coefficient). Unit: Inverse Depth Unit [1/ft]

        Returns:
            normal_porosity_fraction (list[float]): Porosity values on the normal compaction trend. Unit: Fraction [v/v]"""
        return [
            EffectiveStressPorePressure.calculate_normal_compaction_trend_porosity(
                tvd=value,
                surface_porosity_fraction=surface_porosity_fraction,
                porosity_compaction_coefficient_per_ft=porosity_compaction_coefficient_per_ft,
            )
            for value in tvd
        ]

    @staticmethod
    def calculate_pore_pressure_from_effective_stress_array(overburden_stress_psi: list[float], effective_vertical_stress_psi: list[float]) -> list[float]:
        """Calculate pore pressure for an array of overburden and effective vertical stress values using the Terzaghi relation.

        Args:
            overburden_stress_psi (list[float]): Total vertical (overburden) stress values. Unit: Pressure Unit [psi]
            effective_vertical_stress_psi (list[float]): Vertical effective stress values. Unit: Pressure Unit [psi]

        Returns:
            pore_pressure_psi (list[float]): Pore pressure values. Unit: Pressure Unit [psi]"""
        return [
            EffectiveStressPorePressure.calculate_pore_pressure_from_effective_stress(
                overburden_stress_psi=overburden_value,
                effective_vertical_stress_psi=effective_stress_value,
            )
            for overburden_value, effective_stress_value in zip(overburden_stress_psi, effective_vertical_stress_psi)
        ]
