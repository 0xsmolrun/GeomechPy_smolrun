"""Mechanical stratigraphy: discrete lithology classification from the Gamma Ray log."""

from enum import IntEnum


class Lithology(IntEnum):
    """Discrete lithology codes for mechanical stratigraphy (written to a lithology curve).

    Reference:
        Rider, Malcolm H. The Geological Interpretation of Well Logs. 2nd ed. Rider-French Consulting Ltd., 2002. Chapter 'The Gamma Ray Log'."""

    SAND = 0
    SHALE = 1
    LIMESTONE = 2
    COAL = 6


class MechanicalStratigraphyCalculation:
    """Computation of the mechanical stratigraphy (discrete lithology) from the Gamma Ray log using a gAPI cutoff.

    Reference:
        Rider, Malcolm H. The Geological Interpretation of Well Logs. 2nd ed. Rider-French Consulting Ltd., 2002. Chapter 'The Gamma Ray Log'."""

    @staticmethod
    def determine_lithology(
        gamma_ray: float,
        gr_threshold: float = 75.0,
        coal_flag: bool = False,
        limestone_flag: bool = False,
    ) -> int:
        """Determine the discrete lithology code from a Gamma Ray reading.

        Equation type: Threshold classification (gamma_ray >= gr_threshold -> shale, otherwise sand), with coal and limestone overrides.

        Applicable for: Clastic sequences (sandstone vs shale); coal and limestone flagged from independent logs.

        Reference: Rider, Malcolm H. The Geological Interpretation of Well Logs. 2nd ed. Rider-French Consulting Ltd., 2002. Chapter 'The Gamma Ray Log'.

        Args:
            gamma_ray (float): Gamma Ray reading. Unit: [gAPI]
            gr_threshold (float): Gamma Ray cutoff (baseline); readings at or above this value are shale. Unit: [gAPI]. Defaults to 75.0 gAPI
            coal_flag (bool): When True the sample is classified as coal, overriding the Gamma Ray. Unit: unitless (boolean). Defaults to False
            limestone_flag (bool): When True the sample is classified as limestone, overriding the Gamma Ray; ignored when coal_flag is True. Unit: unitless (boolean). Defaults to False

        Returns:
            lithology_code (int): Discrete lithology code (Sand = 0, Shale = 1, Limestone = 2, Coal = 6). Unit: unitless (lithology code)"""
        if coal_flag:
            lithology_code = int(Lithology.COAL)
        elif limestone_flag:
            lithology_code = int(Lithology.LIMESTONE)
        elif gamma_ray >= gr_threshold:
            lithology_code = int(Lithology.SHALE)
        else:
            lithology_code = int(Lithology.SAND)

        return lithology_code

    @staticmethod
    def determine_lithology_array(
        gamma_ray: list[float],
        gr_threshold: float = 75.0,
        coal_flag: list[bool] | None = None,
        limestone_flag: list[bool] | None = None,
    ) -> list[int]:
        """Determine the discrete lithology code for an array of Gamma Ray readings.

        Equation type: Threshold classification (gamma_ray >= gr_threshold -> shale, otherwise sand), with coal and limestone overrides, applied per sample.

        Applicable for: Clastic sequences (sandstone vs shale); coal and limestone flagged from independent logs.

        Reference: Rider, Malcolm H. The Geological Interpretation of Well Logs. 2nd ed. Rider-French Consulting Ltd., 2002. Chapter 'The Gamma Ray Log'.

        Args:
            gamma_ray (list[float]): Gamma Ray readings. Unit: [gAPI]
            gr_threshold (float): Gamma Ray cutoff (baseline); readings at or above this value are shale. Unit: [gAPI]. Defaults to 75.0 gAPI
            coal_flag (list[bool] | None): Per-sample coal flags; must match the length of gamma_ray when provided. Unit: unitless (boolean). Defaults to None (all False)
            limestone_flag (list[bool] | None): Per-sample limestone flags; must match the length of gamma_ray when provided. Unit: unitless (boolean). Defaults to None (all False)

        Returns:
            lithology_codes (list[int]): Discrete lithology codes (Sand = 0, Shale = 1, Limestone = 2, Coal = 6). Unit: unitless (lithology code)

        Raises:
            ValueError: If coal_flag or limestone_flag is provided with a length that does not match gamma_ray."""
        coal_flag_per_sample = coal_flag if coal_flag is not None else [False] * len(gamma_ray)
        limestone_flag_per_sample = limestone_flag if limestone_flag is not None else [False] * len(gamma_ray)

        if len(coal_flag_per_sample) != len(gamma_ray):
            raise ValueError("coal_flag must have the same length as gamma_ray")
        if len(limestone_flag_per_sample) != len(gamma_ray):
            raise ValueError("limestone_flag must have the same length as gamma_ray")

        return [
            MechanicalStratigraphyCalculation.determine_lithology(
                gamma_ray=gamma_ray_value,
                gr_threshold=gr_threshold,
                coal_flag=coal_flag_value,
                limestone_flag=limestone_flag_value,
            )
            for gamma_ray_value, coal_flag_value, limestone_flag_value in zip(
                gamma_ray, coal_flag_per_sample, limestone_flag_per_sample
            )
        ]
