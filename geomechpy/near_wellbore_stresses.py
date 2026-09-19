from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from geomechpy.toolbox import rotate_nev_to_toh, rotate_stress_to_shmax


@dataclass(frozen=True)
class BoreholeWallStresses:
    """Near wellbore stress components on the borehole wall.

    Attributes:
        sigma_boreholewall_rr (npt.NDArray[np.float64]): Radial near wellbore stress component. Unit: Pressure
        sigma_boreholewall_tt (npt.NDArray[np.float64]): Tangential near wellbore stress component. Unit: Pressure
        sigma_boreholewall_zz (npt.NDArray[np.float64]): Axial near wellbore stress component. Unit: Pressure
        sigma_boreholewall_tz (npt.NDArray[np.float64]): Tangential-axial shear stress component. Unit: Pressure
        sigma_boreholewall_rt (npt.NDArray[np.float64]): Radial-tangential shear stress component. Unit: Pressure
        sigma_boreholewall_rz (npt.NDArray[np.float64]): Radial-axial shear stress component. Unit: Pressure
    """

    sigma_boreholewall_rr: npt.NDArray[np.float64]
    sigma_boreholewall_tt: npt.NDArray[np.float64]
    sigma_boreholewall_zz: npt.NDArray[np.float64]
    sigma_boreholewall_tz: npt.NDArray[np.float64]
    sigma_boreholewall_rt: npt.NDArray[np.float64]
    sigma_boreholewall_rz: npt.NDArray[np.float64]


@dataclass(frozen=True)
class BoreholeGeneralStresses:
    """Near wellbore stress components at any azimuthal and radial position around a borehole.

    Attributes:
        sigma_general_rr (npt.NDArray[np.float64]): Radial near wellbore stress component. Unit: Pressure
        sigma_general_tt (npt.NDArray[np.float64]): Tangential near wellbore stress component. Unit: Pressure
        sigma_general_zz (npt.NDArray[np.float64]): Axial near wellbore stress component. Unit: Pressure
        sigma_general_tz (npt.NDArray[np.float64]): Tangential-axial shear stress component. Unit: Pressure
        sigma_general_rt (npt.NDArray[np.float64]): Radial-tangential shear stress component. Unit: Pressure
        sigma_general_rz (npt.NDArray[np.float64]): Radial-axial shear stress component. Unit: Pressure
    """

    sigma_general_rr: npt.NDArray[np.float64]
    sigma_general_tt: npt.NDArray[np.float64]
    sigma_general_zz: npt.NDArray[np.float64]
    sigma_general_tz: npt.NDArray[np.float64]
    sigma_general_rt: npt.NDArray[np.float64]
    sigma_general_rz: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PrincipalStressesBoreholewall:
    """Principal stresses derived from the near wellbore stresses on the borehole wall.

    Attributes:
        sigma_boreholewall_1 (npt.NDArray[np.float64]): Maximum principal stress. Unit: Pressure
        sigma_boreholewall_2 (npt.NDArray[np.float64]): Intermediate principal stress. Unit: Pressure
        sigma_boreholewall_3 (npt.NDArray[np.float64]): Minimum principal stress. Unit: Pressure
        theta_boreholewall_tortuosity (npt.NDArray[np.float64]): Wellbore tortuosity (angle between borehole axis and plane of zero shear stress). Unit: [deg]
    """

    sigma_boreholewall_1: npt.NDArray[np.float64]
    sigma_boreholewall_2: npt.NDArray[np.float64]
    sigma_boreholewall_3: npt.NDArray[np.float64]
    theta_boreholewall_tortuosity: npt.NDArray[np.float64]


class NearWellboreStressesCalculation:
    """Computation of the near wellbore stresses azimuthally and radially around a borehole for any borehole orientation.

    Reference:
        Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Chapter 4.
        Jaeger, John Conrad, Neville GW Cook, and Robert Zimmerman. Fundamentals of rock mechanics. John Wiley & Sons, 2009; Chapter 2.3, eq. 2.31 - 2.33."""

    @staticmethod
    def calculate_kirsch_borehole_wall_stresses(
        shmin: float,
        shmax: float,
        svert: float,
        pore_pressure: float,
        shmax_azimuth: float,
        mud_pressure: float,
        theta: npt.NDArray[np.float64],
        poisson_ratio_static: float,
        borehole_deviation: float,
        borehole_azimuth: float,
    ) -> BoreholeWallStresses:
        """Compute the near wellbore stresses on the borehole wall for any borehole orientation using the Kirsch solution.

        Reference: Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Chapter 4 eq. 4.89 - 4.94.

        Args:
            shmin (float): Minimum horizontal stress magnitude. Unit: Pressure Unit [psi]
            shmax (float): Maximum horizontal stress magnitude. Unit: Pressure Unit [psi]
            svert (float): Vertical stress magnitude. Unit: Pressure Unit [psi]
            pore_pressure (float): Pore pressure. Unit: Pressure Unit [psi]
            shmax_azimuth (float): Direction of the maximum horizontal stress magnitude relative to Geographic NORTH. Unit: [deg]
            mud_pressure (float): Mud pressure inside the borehole. Unit: Pressure Unit [psi]
            theta (npt.NDArray[np.float64]): Azimuthal angles around the borehole circumference measured relative to top of hole (TOH). Unit: [deg]
            poisson_ratio_static (float): Static Poisson's ratio. Unit: unitless
            borehole_deviation (float): Borehole inclination. Unit: [deg]
            borehole_azimuth (float): Borehole azimuth. Unit: [deg]

        Returns:
            BoreholeWallStresses: Dataclass containing the radial, tangential, axial and shear stress components on the borehole wall. See `BoreholeWallStresses` for details. Unit: consistent with input pressure unit
        """
        stress_tensor_nev = rotate_stress_to_shmax(shmin, shmax, svert, shmax_azimuth)
        stress_toh = rotate_nev_to_toh(borehole_deviation, borehole_azimuth, stress_tensor_nev)

        sx0 = stress_toh[0, 0]
        sy0 = stress_toh[1, 1]
        sz0 = stress_toh[2, 2]
        sxy0 = stress_toh[1, 0]
        syz0 = stress_toh[2, 1]
        sxz0 = stress_toh[2, 0]

        theta_rad = np.deg2rad(theta)

        sigma_boreholewall_rr = (mud_pressure - pore_pressure) * np.ones(len(theta))
        sigma_boreholewall_tt = (sx0 + sy0) - 2 * (sx0 - sy0) * np.cos(2 * theta_rad) - 4 * sxy0 * np.sin(2 * theta_rad) - (mud_pressure - pore_pressure)
        sigma_boreholewall_zz = sz0 - poisson_ratio_static * (2 * (sx0 - sy0) * np.cos(2 * theta_rad) + 4 * sxy0 * np.sin(2 * theta_rad))
        sigma_boreholewall_rt = np.zeros(len(theta))
        sigma_boreholewall_tz = 2 * (-sxz0 * np.sin(theta_rad) + syz0 * np.cos(theta_rad))
        sigma_boreholewall_rz = np.zeros(len(theta))

        return BoreholeWallStresses(
            sigma_boreholewall_rr,
            sigma_boreholewall_tt,
            sigma_boreholewall_zz,
            sigma_boreholewall_tz,
            sigma_boreholewall_rt,
            sigma_boreholewall_rz,
        )

    @staticmethod
    def calculate_kirsch_borehole_stresses_general(
        shmin: float,
        shmax: float,
        svert: float,
        pore_pressure: float,
        shmax_azimuth: float,
        mud_pressure: float,
        theta: npt.NDArray[np.float64],
        radius: npt.NDArray[np.float64],
        poisson_ratio_static: float,
        borehole_deviation: float,
        borehole_azimuth: float,
        bitsize: float,
    ) -> BoreholeGeneralStresses:
        """Compute the near wellbore stresses around a wellbore at any azimuthal or radial position for a single depth for any borehole orientation using the Kirsch solution.

        Reference: Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Chapter 4 eq. 4.83 - 4.92.

        Args:
            shmin (float): Minimum horizontal stress magnitude. Unit: Pressure Unit [psi]
            shmax (float): Maximum horizontal stress magnitude. Unit: Pressure Unit [psi]
            svert (float): Vertical stress magnitude. Unit: Pressure Unit [psi]
            pore_pressure (float): Pore pressure. Unit: Pressure Unit [psi]
            shmax_azimuth (float): Direction of the maximum horizontal stress magnitude relative to Geographic NORTH. Unit: [deg]
            mud_pressure (float): Mud pressure inside the borehole. Unit: Pressure Unit [psi]
            theta (npt.NDArray[np.float64]): Azimuthal angles around the borehole circumference measured relative to top of hole (TOH). Unit: [deg]
            radius (npt.NDArray[np.float64]): Radial position measured from the borehole centre. Unit: [in]
            poisson_ratio_static (float): Static Poisson's ratio. Unit: unitless
            borehole_deviation (float): Borehole inclination. Unit: [deg]
            borehole_azimuth (float): Borehole azimuth. Unit: [deg]
            bitsize (float): Borehole radius. Unit: [in]

        Returns:
            BoreholeGeneralStresses: Dataclass containing the radial, tangential, axial and shear stress components azimuthally and radially around a borehole. See `BoreholeGeneralStresses` for details. Unit: consistent with input pressure unit
        """
        stress_tensor_nev = rotate_stress_to_shmax(shmin, shmax, svert, shmax_azimuth)
        stress_toh = rotate_nev_to_toh(borehole_deviation, borehole_azimuth, stress_tensor_nev)

        sx0 = stress_toh[0, 0]
        sy0 = stress_toh[1, 1]
        sz0 = stress_toh[2, 2]
        sxy0 = stress_toh[1, 0]
        syz0 = stress_toh[2, 1]
        sxz0 = stress_toh[2, 0]

        theta_rad = np.deg2rad(theta)

        a0 = (bitsize ** 2) / (radius ** 2)
        a1_p = 1 + bitsize ** 2 / radius ** 2
        a1_m = 1 - bitsize ** 2 / radius ** 2
        a2 = 1 + 3 * bitsize ** 4 / radius ** 4
        a3m = 1 - 3 * bitsize ** 4 / radius ** 4 + 2 * bitsize ** 2 / radius ** 2
        a3p = 1 + 3 * bitsize ** 4 / radius ** 4 - 4 * bitsize ** 2 / radius ** 2

        sigma_general_rr = 0.5 * (sx0 + sy0) * a1_m + 0.5 * (sx0 - sy0) * a3p * np.cos(2 * theta_rad) + sxy0 * a3p * np.sin(2 * theta_rad) + (mud_pressure - pore_pressure) * a0
        sigma_general_tt = 0.5 * (sx0 + sy0) * a1_p - 0.5 * (sx0 - sy0) * a2 * np.cos(2 * theta_rad) - sxy0 * a2 * np.sin(2 * theta_rad) - (mud_pressure - pore_pressure) * a0
        sigma_general_zz = sz0 - poisson_ratio_static * (2 * (sx0 - sy0) * a0 * np.cos(2 * theta_rad) + 4 * sxy0 * a0 * np.sin(2 * theta_rad))
        sigma_general_rt = (-0.5 * (sx0 - sy0) * a3m * np.sin(2 * theta_rad)) + sxy0 * a3m * np.cos(2 * theta_rad)
        sigma_general_tz = (-sxz0 * np.sin(theta_rad) + syz0 * np.cos(theta_rad)) * a1_p
        sigma_general_rz = (sxz0 * np.cos(theta_rad) + syz0 * np.sin(theta_rad)) * a1_m

        return BoreholeGeneralStresses(
            sigma_general_rr,
            sigma_general_tt,
            sigma_general_zz,
            sigma_general_tz,
            sigma_general_rt,
            sigma_general_rz,
        )

    @staticmethod
    def calculate_principal_stresses_analytical_boreholewall(
        sigma_boreholewall_rr: npt.NDArray[np.float64],
        sigma_boreholewall_tt: npt.NDArray[np.float64],
        sigma_boreholewall_zz: npt.NDArray[np.float64],
        sigma_boreholewall_tz: npt.NDArray[np.float64],
    ) -> PrincipalStressesBoreholewall:
        """Compute the principal stresses from the near wellbore stresses on the borehole wall.

        Reference: Jaeger, John Conrad, Neville GW Cook, and Robert Zimmerman. Fundamentals of rock mechanics. John Wiley & Sons, 2009; Chapter 2.3, eq. 2.31 - 2.33.

        Args:
            sigma_boreholewall_rr (npt.NDArray[np.float64]): Radial near wellbore stress component computed at azimuthal positions relative to the borehole axis. Unit: Pressure Unit [psi]
            sigma_boreholewall_tt (npt.NDArray[np.float64]): Tangential near wellbore stress component computed at azimuthal positions relative to the borehole axis. Unit: Pressure Unit [psi]
            sigma_boreholewall_zz (npt.NDArray[np.float64]): Axial near wellbore stress component computed at azimuthal positions relative to the borehole axis. Unit: Pressure Unit [psi]
            sigma_boreholewall_tz (npt.NDArray[np.float64]): Tangential-axial shear stress component at the borehole wall. Unit: Pressure Unit [psi]

        Returns:
            PrincipalStressesBoreholewall: Dataclass containing the maximum, intermediate and minimum principal stresses and the wellbore tortuosity angle. See `PrincipalStressesBoreholewall` for details. Unit: consistent with input pressure unit
        """
        sigma_a = 0.5 * (sigma_boreholewall_tt + sigma_boreholewall_zz) + 0.5 * np.sqrt((sigma_boreholewall_tt - sigma_boreholewall_zz) ** 2 + 4 * sigma_boreholewall_tz ** 2)
        sigma_b = 0.5 * (sigma_boreholewall_tt + sigma_boreholewall_zz) - 0.5 * np.sqrt((sigma_boreholewall_tt - sigma_boreholewall_zz) ** 2 + 4 * sigma_boreholewall_tz ** 2)
        stacked = np.stack([sigma_a, sigma_b, sigma_boreholewall_rr], axis=0)
        sorted_stacked = np.sort(stacked, axis=0)
        sigma_boreholewall_1 = sorted_stacked[2]
        sigma_boreholewall_2 = sorted_stacked[1]
        sigma_boreholewall_3 = sorted_stacked[0]

        theta_boreholewall_tortuosity = 0.5 * np.degrees(np.arctan(2 * sigma_boreholewall_tz / (sigma_boreholewall_tt - sigma_boreholewall_zz)))

        return PrincipalStressesBoreholewall(
            sigma_boreholewall_1,
            sigma_boreholewall_2,
            sigma_boreholewall_3,
            theta_boreholewall_tortuosity,
        )
