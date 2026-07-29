import math

import numpy as np
import numpy.typing as npt


def rotate_stress_to_shmax(
    shmin: float,
    shmax: float,
    svert: float,
    shmax_azimuth: float,
) -> npt.NDArray[np.float64]:
    """Rotate the principal stress tensor into the direction of the maximum horizontal stress.

    Assumes the vertical stress is not tilted.

    Reference:
        Jaeger, John Conrad, Neville GW Cook, and Robert Zimmerman. Fundamentals of rock mechanics. John Wiley & Sons, 2009, Chapter 2.3.

    Args:
        shmin (float): Minimum horizontal stress magnitude. Unit: Pressure
        shmax (float): Maximum horizontal stress magnitude. Unit: Pressure
        svert (float): Vertical stress magnitude. Unit: Pressure
        shmax_azimuth (float): Direction of the maximum horizontal stress magnitude relative to Geographic NORTH. Unit: [deg]

    Returns:
        npt.NDArray[np.float64]: Stress tensor in NEV coordinate system (North-East-Vertical). Unit: same as input pressure unit
    """
    stress_tensor = [shmax, shmin, svert] * np.identity(3)

    shmax_azimuth_rad = shmax_azimuth * (math.pi / 180)

    nev_rotation_matrix = np.array([
        [math.cos(shmax_azimuth_rad), math.sin(shmax_azimuth_rad), 0],
        [-math.sin(shmax_azimuth_rad), math.cos(shmax_azimuth_rad), 0],
        [0, 0, 1],
    ])

    stress_nev = np.matmul(np.transpose(nev_rotation_matrix), np.matmul(stress_tensor, nev_rotation_matrix))
    return stress_nev


def rotate_nev_to_toh(
    borehole_deviation: float,
    borehole_azimuth: float,
    stress_tensor_nev: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Rotate the stress tensor from the geographic reference into the borehole reference system using top of hole as reference.

    Reference:
        Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Appendix C; eq C.58.

    Args:
        borehole_deviation (float): Borehole inclination. Unit: [deg]
        borehole_azimuth (float): Borehole azimuth. Unit: [deg]
        stress_tensor_nev (npt.NDArray[np.float64]): Stress tensor in NEV coordinate system (North-East-Vertical). Unit: Pressure

    Returns:
        npt.NDArray[np.float64]: Stress tensor in TOH coordinate system (Top of Hole). Unit: same as input pressure unit
    """
    bh_azimuth_rad = borehole_azimuth * (math.pi / 180)
    bh_deviation_rad = borehole_deviation * (math.pi / 180)

    nev_to_toh = np.array([
        [math.cos(bh_azimuth_rad) * math.cos(bh_deviation_rad), math.sin(bh_azimuth_rad) * math.cos(bh_deviation_rad), -math.sin(bh_deviation_rad)],
        [-math.sin(bh_azimuth_rad), math.cos(bh_azimuth_rad), 0],
        [math.cos(bh_azimuth_rad) * math.sin(bh_deviation_rad), math.sin(bh_azimuth_rad) * math.sin(bh_deviation_rad), math.cos(bh_deviation_rad)],
    ])

    stress_tensor_toh = np.matmul(nev_to_toh, np.matmul(stress_tensor_nev, np.transpose(nev_to_toh)))
    return stress_tensor_toh


def determine_lithology(
    gamma_ray: float,
    gr_threshold: float = 75.0,
    coal_flag: bool = False,
    limestone_flag: bool = False,
) -> int:
    """Determine the mechanical stratigraphy lithology for a single Gamma Ray sample.

    Args:
        gamma_ray (float): Gamma Ray reading. Unit: [gAPI]
        gr_threshold (float): Gamma Ray cutoff separating clean (sand) from shale.
            Unit: [gAPI]. Default: 75.0
        coal_flag (bool): Coal flag for this sample. Default: False
        limestone_flag (bool): Limestone flag for this sample. Default: False

    Returns:
        int: Lithology code. Sand = 0, Shale = 1, Limestone = 2, Coal = 6
    """
    if coal_flag:
        return 6
    if limestone_flag:
        return 2
    if gamma_ray > gr_threshold:
        return 1  # shale
    return 0  # sand


def determine_lithology_array(
    gamma_ray: list[float],
    gr_threshold: float = 75.0,
    coal_flag: list[bool] | None = None,
    limestone_flag: list[bool] | None = None,
) -> list[int]:
    """Determine the mechanical stratigraphy lithology for a Gamma Ray log.

    Applies :func:`determine_lithology` to every sample of a Gamma Ray curve. The
    ``coal_flag`` and ``limestone_flag`` sequences, when provided, must be the same
    length as ``gamma_ray``; when omitted the corresponding flag is treated as
    False for every sample.

    Args:
        gamma_ray (list[float]): Gamma Ray readings. Unit: [gAPI]
        gr_threshold (float): Gamma Ray cutoff separating clean (sand) from shale.
            Unit: [gAPI]. Default: 75.0
        coal_flag (list[bool] | None): Per-sample coal flags. Default: None (all False)
        limestone_flag (list[bool] | None): Per-sample limestone flags. Default: None (all False)

    Returns:
        list[int]: Lithology codes. Sand = 0, Shale = 1, Limestone = 2, Coal = 6

    Raises:
        ValueError: If ``coal_flag`` or ``limestone_flag`` is provided with a length
            that does not match ``gamma_ray``.
    """
    n = len(gamma_ray)
    if coal_flag is None:
        coal_flag = [False] * n
    if limestone_flag is None:
        limestone_flag = [False] * n
    if len(coal_flag) != n or len(limestone_flag) != n:
        raise ValueError(
            "coal_flag and limestone_flag, when provided, must have the same length as gamma_ray"
        )
    return [
        determine_lithology(gr, gr_threshold, c, l)
        for gr, c, l in zip(gamma_ray, coal_flag, limestone_flag, strict=True)
    ]
