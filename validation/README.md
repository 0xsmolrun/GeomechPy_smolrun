# GeomechPy Examples: Elastic Properties & Rock Strength

This folder contains an example-driven Jupyter notebook demonstrating practical usage of:

- `geomechpy/elastic_properties.py` (`ElasticPropertiesConverter`)
- `geomechpy/rock_strength.py` (`RockStrengthPropertiesConverter`)
- `geomechpy/unit_converter.py` (`UnitConverter`)

## Files

- `elastic_properties_examples.ipynb` — executed example notebook (outputs included) showing real geomechanics workflows
- `geomechpy/` — local copy of the three modules (for self-contained execution)

## What is demonstrated

1. Pairwise conversions between all elastic moduli (Young's, bulk, shear, Lamé, P-wave, Poisson's ratio)
2. Dynamic elastic properties computed from sonic log data (DTCO/DTSM + density), both from slowness and velocity
3. Unit conversion between SI (GPa) and oilfield (Mpsi, psi) units
4. Rock strength estimates: Plumb UCS, tensile strength, Lal friction angle (scalar and array variants)
5. Full end-to-end log processing workflow

Open the notebook and inspect the saved outputs (or re-run all cells) to verify the calculations.

**Note:** The Plumb UCS correlation currently returns unrealistically small values (order of 1 psi) for typical moduli. This appears to be a units issue in the library (multiplier 0.2103 applied to Mpsi yielding psi-scale results). The notebook reports the actual library output so the behavior can be reviewed and fixed.
