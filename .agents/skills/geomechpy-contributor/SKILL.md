# GeomechPy Contributor Context

You are a Python developer contributing to the GeomechPy library — a scientific geomechanics library for petroleum engineering workflows.

## Readability Rules

- No code shortcuts: avoid compressing logic onto a single line when a multi-line version is clearer.
- No chained operations on one line if readability suffers.
- All variable names must be fully self-explanatory — including intermediate or temporary ones.
  - ❌ Avoid: `b1`, `a0`, `tmp`, `val`, `x`
  - ✅ Use: `lithostatic_gradient`, `air_pressure`, `friction_angle_multiplier`
- Well-known domain abbreviations (e.g. `tvd`, `ucs`, `dtco`, `fang`) are acceptable only when widely recognised in petroleum geomechanics.

## Class & Module Structure

- Each module defines **one class** representing a category of computation (e.g. `RockStrengthPropertiesConverter`, `OverburdenStressCalculation`).
- All methods are `@staticmethod`.
- Scalar functions accept a single `float`; array variants accept `list[float]` and are named with the `_array` suffix.
- Array variants are implemented as list comprehensions that delegate to the scalar function.

```python
@staticmethod
def calculate_overburden_stress_onshore_array(
    tvd: list[float],
    lithostatic_gradient: float = 1.05,
    air_gap_ft: float = 0.0,
) -> list[float]:
    return [
        OverburdenStressCalculation.calculate_overburden_stress_onshore(
            tvd=value,
            lithostatic_gradient=lithostatic_gradient,
            air_gap_ft=air_gap_ft,
        )
        for value in tvd
    ]
```

## Units Convention

Physical units must appear **in both places**:

1. **Variable name suffix** where practical: `pressure_mpa`, `depth_ft`, `gradient_psi_per_ft`
2. **Docstring** in every `Args` and `Returns` entry: `Unit: psi/ft`

## Documentation — Required Docstring Structure

Every function must have a complete docstring. Use this structure (Google-style):

```
One-line summary of what the function computes.

Equation type: (if applicable, e.g. "Linear law (y = a*x)")

Applicable for: (if applicable, e.g. "Shale", "Generic clastic rocks")

Reference: Full citation — Author, Year. Title. Conference/Journal/Book.
           Use "N/A" only if truly no reference exists.

Args:
    param_name (type): Human-readable description. Unit: X
    param_name (type): Human-readable description. Unit: X. Defaults to Y.

Returns:
    return_var_name (type): Human-readable description. Unit: X
```

Rules:
- Every `Args` entry must include the unit.
- Every `Returns` entry must include the unit **and** the return variable name.
- Class-level docstrings must also include a `Reference:` section.
- Do not leave any section empty or with placeholder text.

**Example (scalar function):**

```python
@staticmethod
def convert_yme_sta_to_ucs_plumb(yme_sta_mpsi: float) -> float:
    """Convert static Young's modulus to UCS using Plumb Generic correlation.

    Equation type: Linear law (y = a*x)

    Applicable for: Generic clastic rocks.

    Reference: Plumb, R. A., 1994. Influence of composition and texture on the
               failure properties of clastic rocks. Eurock 94, Rock Mechanics in
               Petroleum Engineering conference, Delft, Netherlands, pp. 13-20.

    Args:
        yme_sta_mpsi (float): Static Young's modulus magnitude. Unit: Mpsi

    Returns:
        ucs_plumb_generic_psi (float): Unconfined compressive strength from
            Plumb generic correlation. Unit: psi
    """
    multiplier = 0.210306770614015
    ucs_plumb_generic_psi = multiplier * yme_sta_mpsi

    return float(ucs_plumb_generic_psi)
```

## Fit Existing Style

Before writing new code, scan an existing module to align with its patterns:

- `geomechpy/rock_strength.py` — converter class (`@dataclass(frozen=True)`) example
- `geomechpy/overburden_stress.py` — calculation class example
- `geomechpy/pore_pressure.py` — additional reference

## Git & Version Control

- **Never push code automatically.** After completing code changes, always stop and let the user decide when and whether to push to the remote.
- Inform the user that the changes are ready and suggest they push when they are satisfied — for example: *"The changes are ready. Run `git push` when you'd like to publish them."*
- Only run `git push` (or any push-equivalent command) if the user **explicitly** requests it in their message (e.g. "push it", "go ahead and push", "push to remote").
- Committing locally is acceptable when the user asks to commit, but never follow a commit immediately with a push unless explicitly told to do so.

## Checklist Before Submitting Code

- [ ] All variable names are fully descriptive (no single-letter or abbreviated temporaries)
- [ ] Units appear in both variable names and docstrings
- [ ] Every function has a complete docstring with `Args:`, `Returns:`, and `Reference:`
- [ ] Scalar function has a corresponding `_array` variant
- [ ] No logic compressed onto a single line at the cost of readability
- [ ] New code follows the class/static-method structure of existing modules
