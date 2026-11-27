# Validation notebooks

Worked, executed notebooks that exercise GeomechPy modules end to end and keep their
output cells, so the numbers can be checked without re-running anything.

## `elastic_rock_strength_unit_validation.ipynb`

Tests **every public function** in `geomechpy/elastic_properties.py` and
`geomechpy/rock_strength.py`, using `geomechpy/unit_converter.py` as the unit bridge
between them. Each step prints its result and is guarded by `assert`; the final cell
prints `ALL CHECKS PASSED`.

Coverage:

- `UnitConverter` — `convert_pressure`, `convert_equivalent_mud_weight`,
  `convert_length`, `convert_acoustic` and their `_array` variants, checked against
  first-principles constants and round-trips.
- `ElasticPropertiesConverter` — the `ElasticProperties` dataclass, all 15 pairwise
  `convert_from_*` methods (scalar and `_array`), and `convert_dynamic_*` from both
  slowness and velocity, cross-checked through `convert_acoustic`.
- `RockStrengthPropertiesConverter` — Plumb UCS, tensile strength and Lal friction
  angle (scalar and `_array`), fed Young's modulus in Mpsi via `convert_pressure`.

### Run it

```bash
pip install jupyter nbconvert pandas numpy
jupyter nbconvert --to notebook --execute --inplace \
    validation/elastic_rock_strength_unit_validation.ipynb
```

### Notes on `unit_converter.py` observed while writing this

- The `convert_acoustic` unit keys are the slash forms in the factor tables
  (`"us/ft"`, `"m/s"`, `"s/ft"`, …); the shorthand aliases in its docstring
  (`"usft"`, `"ms"`) are not recognised.
- `from_equivalent_mud_weight_to_pressure` and
  `convert_from_pressure_to_equivalent_mud_weight` call `convert_pressure(unit)` with a
  single argument and raise `TypeError`; they are not used by this notebook.
