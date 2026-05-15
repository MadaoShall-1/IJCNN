"""Physics unit lookup and SI normalization helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


SI_UNIT_BY_DIMENSION = {
    "length": "m",
    "area": "m^2",
    "volume": "m^3",
    "time": "s",
    "mass": "kg",
    "velocity": "m/s",
    "acceleration": "m/s^2",
    "force": "N",
    "energy": "J",
    "power": "W",
    "charge": "C",
    "linear_charge_density": "C/m",
    "surface_charge_density": "C/m^2",
    "coulomb_constant_unit": "N*m^2/C^2",
    "density": "kg/m^3",
    "voltage": "V",
    "electric_field": "V/m",
    "resistance": "Ω",
    "current": "A",
    "capacitance": "F",
    "inductance": "H",
    "frequency": "Hz",
    "angular_frequency": "rad/s",
    "magnetic_field": "T",
    "magnetic_flux": "Wb",
    "temperature": "K",
    "angle": "rad",
    "pressure": "Pa",
    "specific_heat_capacity": "J/(kg*K)",
    "specific_latent_heat": "J/kg",
    "count": None,
    "turn_density": "turn/m",
}


UNIT_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "m": {"unit_symbol": "m", "unit_name": "meter", "dimension": "length", "to_si": 1.0},
    "m^2": {"unit_symbol": "m^2", "unit_name": "square meter", "dimension": "area", "to_si": 1.0},
    "m²": {"unit_symbol": "m^2", "unit_name": "square meter", "dimension": "area", "to_si": 1.0},
    "m?": {"unit_symbol": "m^2", "unit_name": "square meter", "dimension": "area", "to_si": 1.0},
    "cm": {"unit_symbol": "cm", "unit_name": "centimeter", "dimension": "length", "to_si": 0.01},
    "cm^2": {"unit_symbol": "cm^2", "unit_name": "square centimeter", "dimension": "area", "to_si": 1e-4},
    "cm2": {"unit_symbol": "cm^2", "unit_name": "square centimeter", "dimension": "area", "to_si": 1e-4},
    "cm²": {"unit_symbol": "cm^2", "unit_name": "square centimeter", "dimension": "area", "to_si": 1e-4},
    "cm?": {"unit_symbol": "cm^2", "unit_name": "square centimeter", "dimension": "area", "to_si": 1e-4},
    "cm??": {"unit_symbol": "cm^2", "unit_name": "square centimeter", "dimension": "area", "to_si": 1e-4},
    "mm": {"unit_symbol": "mm", "unit_name": "millimeter", "dimension": "length", "to_si": 0.001},
    "mm^2": {"unit_symbol": "mm^2", "unit_name": "square millimeter", "dimension": "area", "to_si": 1e-6},
    "mm2": {"unit_symbol": "mm^2", "unit_name": "square millimeter", "dimension": "area", "to_si": 1e-6},
    "mm²": {"unit_symbol": "mm^2", "unit_name": "square millimeter", "dimension": "area", "to_si": 1e-6},
    "mm?": {"unit_symbol": "mm^2", "unit_name": "square millimeter", "dimension": "area", "to_si": 1e-6},
    "mm??": {"unit_symbol": "mm^2", "unit_name": "square millimeter", "dimension": "area", "to_si": 1e-6},
    "km": {"unit_symbol": "km", "unit_name": "kilometer", "dimension": "length", "to_si": 1000.0},
    "s": {"unit_symbol": "s", "unit_name": "second", "dimension": "time", "to_si": 1.0},
    "sec": {"unit_symbol": "s", "unit_name": "second", "dimension": "time", "to_si": 1.0},
    "second": {"unit_symbol": "s", "unit_name": "second", "dimension": "time", "to_si": 1.0},
    "seconds": {"unit_symbol": "s", "unit_name": "second", "dimension": "time", "to_si": 1.0},
    "min": {"unit_symbol": "min", "unit_name": "minute", "dimension": "time", "to_si": 60.0},
    "minute": {"unit_symbol": "min", "unit_name": "minute", "dimension": "time", "to_si": 60.0},
    "minutes": {"unit_symbol": "min", "unit_name": "minute", "dimension": "time", "to_si": 60.0},
    "h": {"unit_symbol": "h", "unit_name": "hour", "dimension": "time", "to_si": 3600.0},
    "hour": {"unit_symbol": "h", "unit_name": "hour", "dimension": "time", "to_si": 3600.0},
    "hours": {"unit_symbol": "h", "unit_name": "hour", "dimension": "time", "to_si": 3600.0},
    "kg": {"unit_symbol": "kg", "unit_name": "kilogram", "dimension": "mass", "to_si": 1.0},
    "g": {"unit_symbol": "g", "unit_name": "gram", "dimension": "mass", "to_si": 0.001},
    "m/s": {"unit_symbol": "m/s", "unit_name": "meter per second", "dimension": "velocity", "to_si": 1.0},
    "km/h": {"unit_symbol": "km/h", "unit_name": "kilometer per hour", "dimension": "velocity", "to_si": 1000.0 / 3600.0},
    "m/s^2": {"unit_symbol": "m/s^2", "unit_name": "meter per second squared", "dimension": "acceleration", "to_si": 1.0},
    "m/s²": {"unit_symbol": "m/s^2", "unit_name": "meter per second squared", "dimension": "acceleration", "to_si": 1.0},
    "m/sÂ²": {"unit_symbol": "m/s^2", "unit_name": "meter per second squared", "dimension": "acceleration", "to_si": 1.0},
    "N": {"unit_symbol": "N", "unit_name": "newton", "dimension": "force", "to_si": 1.0},
    "J": {"unit_symbol": "J", "unit_name": "joule", "dimension": "energy", "to_si": 1.0},
    "mJ": {"unit_symbol": "mJ", "unit_name": "millijoule", "dimension": "energy", "to_si": 0.001},
    "μJ": {"unit_symbol": "μJ", "unit_name": "microjoule", "dimension": "energy", "to_si": 1e-6},
    "uJ": {"unit_symbol": "uJ", "unit_name": "microjoule", "dimension": "energy", "to_si": 1e-6},
    "nJ": {"unit_symbol": "nJ", "unit_name": "nanojoule", "dimension": "energy", "to_si": 1e-9},
    "W": {"unit_symbol": "W", "unit_name": "watt", "dimension": "power", "to_si": 1.0},
    "kW": {"unit_symbol": "kW", "unit_name": "kilowatt", "dimension": "power", "to_si": 1000.0},
    "mW": {"unit_symbol": "mW", "unit_name": "milliwatt", "dimension": "power", "to_si": 0.001},
    "MW": {"unit_symbol": "MW", "unit_name": "megawatt", "dimension": "power", "to_si": 1e6},
    "kJ": {"unit_symbol": "kJ", "unit_name": "kilojoule", "dimension": "energy", "to_si": 1000.0},
    "MJ": {"unit_symbol": "MJ", "unit_name": "megajoule", "dimension": "energy", "to_si": 1e6},
    "eV": {"unit_symbol": "eV", "unit_name": "electronvolt", "dimension": "energy", "to_si": 1.602176634e-19},
    "C": {"unit_symbol": "C", "unit_name": "coulomb", "dimension": "charge", "to_si": 1.0},
    "mC": {"unit_symbol": "mC", "unit_name": "millicoulomb", "dimension": "charge", "to_si": 0.001},
    "μC": {"unit_symbol": "μC", "unit_name": "microcoulomb", "dimension": "charge", "to_si": 1e-6},
    "uC": {"unit_symbol": "uC", "unit_name": "microcoulomb", "dimension": "charge", "to_si": 1e-6},
    "nC": {"unit_symbol": "nC", "unit_name": "nanocoulomb", "dimension": "charge", "to_si": 1e-9},
    "pC": {"unit_symbol": "pC", "unit_name": "picocoulomb", "dimension": "charge", "to_si": 1e-12},
    "V": {"unit_symbol": "V", "unit_name": "volt", "dimension": "voltage", "to_si": 1.0},
    "V/m": {"unit_symbol": "V/m", "unit_name": "volt per meter", "dimension": "electric_field", "to_si": 1.0},
    "V / m": {"unit_symbol": "V/m", "unit_name": "volt per meter", "dimension": "electric_field", "to_si": 1.0},
    "V/ m": {"unit_symbol": "V/m", "unit_name": "volt per meter", "dimension": "electric_field", "to_si": 1.0},
    "V /m": {"unit_symbol": "V/m", "unit_name": "volt per meter", "dimension": "electric_field", "to_si": 1.0},
    "N/C": {"unit_symbol": "N/C", "unit_name": "newton per coulomb", "dimension": "electric_field", "to_si": 1.0},
    "N / C": {"unit_symbol": "N/C", "unit_name": "newton per coulomb", "dimension": "electric_field", "to_si": 1.0},
    "C/m": {"unit_symbol": "C/m", "unit_name": "coulomb per meter", "dimension": "linear_charge_density", "to_si": 1.0},
    "C/m^2": {"unit_symbol": "C/m^2", "unit_name": "coulomb per square meter", "dimension": "surface_charge_density", "to_si": 1.0},
    "C/m²": {"unit_symbol": "C/m^2", "unit_name": "coulomb per square meter", "dimension": "surface_charge_density", "to_si": 1.0},
    "C/mÂ²": {"unit_symbol": "C/m^2", "unit_name": "coulomb per square meter", "dimension": "surface_charge_density", "to_si": 1.0},
    "N*m^2/C^2": {"unit_symbol": "N*m^2/C^2", "unit_name": "newton square meter per square coulomb", "dimension": "coulomb_constant_unit", "to_si": 1.0},
    "N·m²/C²": {"unit_symbol": "N*m^2/C^2", "unit_name": "newton square meter per square coulomb", "dimension": "coulomb_constant_unit", "to_si": 1.0},
    "N m^2 / C^2": {"unit_symbol": "N*m^2/C^2", "unit_name": "newton square meter per square coulomb", "dimension": "coulomb_constant_unit", "to_si": 1.0},
    "kg/m^3": {"unit_symbol": "kg/m^3", "unit_name": "kilogram per cubic meter", "dimension": "density", "to_si": 1.0},
    "g/cm^3": {"unit_symbol": "g/cm^3", "unit_name": "gram per cubic centimeter", "dimension": "density", "to_si": 1000.0},
    "Ω": {"unit_symbol": "Ω", "unit_name": "ohm", "dimension": "resistance", "to_si": 1.0},
    "Ω": {"unit_symbol": "Ω", "unit_name": "ohm", "dimension": "resistance", "to_si": 1.0},
    "Ohm": {"unit_symbol": "Ω", "unit_name": "ohm", "dimension": "resistance", "to_si": 1.0},
    "ohm": {"unit_symbol": "Ω", "unit_name": "ohm", "dimension": "resistance", "to_si": 1.0},
    "A": {"unit_symbol": "A", "unit_name": "ampere", "dimension": "current", "to_si": 1.0},
    "mA": {"unit_symbol": "mA", "unit_name": "milliampere", "dimension": "current", "to_si": 0.001},
    "F": {"unit_symbol": "F", "unit_name": "farad", "dimension": "capacitance", "to_si": 1.0},
    "μF": {"unit_symbol": "μF", "unit_name": "microfarad", "dimension": "capacitance", "to_si": 1e-6},
    "uF": {"unit_symbol": "uF", "unit_name": "microfarad", "dimension": "capacitance", "to_si": 1e-6},
    "nF": {"unit_symbol": "nF", "unit_name": "nanofarad", "dimension": "capacitance", "to_si": 1e-9},
    "pF": {"unit_symbol": "pF", "unit_name": "picofarad", "dimension": "capacitance", "to_si": 1e-12},
    "H": {"unit_symbol": "H", "unit_name": "henry", "dimension": "inductance", "to_si": 1.0},
    "mH": {"unit_symbol": "mH", "unit_name": "millihenry", "dimension": "inductance", "to_si": 0.001},
    "Hz": {"unit_symbol": "Hz", "unit_name": "hertz", "dimension": "frequency", "to_si": 1.0},
    "kHz": {"unit_symbol": "kHz", "unit_name": "kilohertz", "dimension": "frequency", "to_si": 1000.0},
    "T": {"unit_symbol": "T", "unit_name": "tesla", "dimension": "magnetic_field", "to_si": 1.0},
    "Wb": {"unit_symbol": "Wb", "unit_name": "weber", "dimension": "magnetic_flux", "to_si": 1.0},
    "°C": {"unit_symbol": "°C", "unit_name": "degree Celsius", "dimension": "temperature", "to_si": 1.0},
    "Celsius": {"unit_symbol": "°C", "unit_name": "degree Celsius", "dimension": "temperature", "to_si": 1.0},
    "K": {"unit_symbol": "K", "unit_name": "kelvin", "dimension": "temperature", "to_si": 1.0},
    "°": {"unit_symbol": "deg", "unit_name": "degree", "dimension": "angle", "to_si": 0.017453292519943295},
    "Â°": {"unit_symbol": "deg", "unit_name": "degree", "dimension": "angle", "to_si": 0.017453292519943295},
    "deg": {"unit_symbol": "deg", "unit_name": "degree", "dimension": "angle", "to_si": 0.017453292519943295},
    "degree": {"unit_symbol": "deg", "unit_name": "degree", "dimension": "angle", "to_si": 0.017453292519943295},
    "degrees": {"unit_symbol": "deg", "unit_name": "degree", "dimension": "angle", "to_si": 0.017453292519943295},
    "rad": {"unit_symbol": "rad", "unit_name": "radian", "dimension": "angle", "to_si": 1.0},
    "rad/s": {"unit_symbol": "rad/s", "unit_name": "radian per second", "dimension": "angular_frequency", "to_si": 1.0},
    # Specific heat capacity (treated as a single composite unit; SI is J/(kg*K))
    "J/kg.K": {"unit_symbol": "J/(kg*K)", "unit_name": "joule per kilogram kelvin", "dimension": "specific_heat_capacity", "to_si": 1.0},
    "J/(kg.K)": {"unit_symbol": "J/(kg*K)", "unit_name": "joule per kilogram kelvin", "dimension": "specific_heat_capacity", "to_si": 1.0},
    "J/(kg*K)": {"unit_symbol": "J/(kg*K)", "unit_name": "joule per kilogram kelvin", "dimension": "specific_heat_capacity", "to_si": 1.0},
    "J/kg/K": {"unit_symbol": "J/(kg*K)", "unit_name": "joule per kilogram kelvin", "dimension": "specific_heat_capacity", "to_si": 1.0},
    "J/kg·K": {"unit_symbol": "J/(kg*K)", "unit_name": "joule per kilogram kelvin", "dimension": "specific_heat_capacity", "to_si": 1.0},
    # Specific latent heat (per kg)
    "J/kg": {"unit_symbol": "J/kg", "unit_name": "joule per kilogram", "dimension": "specific_latent_heat", "to_si": 1.0},
    "kJ/kg": {"unit_symbol": "J/kg", "unit_name": "joule per kilogram", "dimension": "specific_latent_heat", "to_si": 1000.0},
    # Mass — extended
    "ton": {"unit_symbol": "ton", "unit_name": "metric ton", "dimension": "mass", "to_si": 1000.0},
    "tons": {"unit_symbol": "ton", "unit_name": "metric ton", "dimension": "mass", "to_si": 1000.0},
    "tonne": {"unit_symbol": "ton", "unit_name": "metric ton", "dimension": "mass", "to_si": 1000.0},
    "mg": {"unit_symbol": "mg", "unit_name": "milligram", "dimension": "mass", "to_si": 1e-6},
    # Volume
    "L": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "l": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "liter": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "liters": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "litre": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "litres": {"unit_symbol": "L", "unit_name": "liter", "dimension": "volume", "to_si": 0.001},
    "mL": {"unit_symbol": "mL", "unit_name": "milliliter", "dimension": "volume", "to_si": 1e-6},
    "ml": {"unit_symbol": "mL", "unit_name": "milliliter", "dimension": "volume", "to_si": 1e-6},
    "m^3": {"unit_symbol": "m^3", "unit_name": "cubic meter", "dimension": "volume", "to_si": 1.0},
    "m³": {"unit_symbol": "m^3", "unit_name": "cubic meter", "dimension": "volume", "to_si": 1.0},
    "cm^3": {"unit_symbol": "cm^3", "unit_name": "cubic centimeter", "dimension": "volume", "to_si": 1e-6},
    "cm³": {"unit_symbol": "cm^3", "unit_name": "cubic centimeter", "dimension": "volume", "to_si": 1e-6},
    # Pressure
    "Pa": {"unit_symbol": "Pa", "unit_name": "pascal", "dimension": "pressure", "to_si": 1.0},
    "kPa": {"unit_symbol": "kPa", "unit_name": "kilopascal", "dimension": "pressure", "to_si": 1000.0},
    "MPa": {"unit_symbol": "MPa", "unit_name": "megapascal", "dimension": "pressure", "to_si": 1e6},
    "atm": {"unit_symbol": "atm", "unit_name": "atmosphere", "dimension": "pressure", "to_si": 101325.0},
    "bar": {"unit_symbol": "bar", "unit_name": "bar", "dimension": "pressure", "to_si": 100000.0},
    "mmHg": {"unit_symbol": "mmHg", "unit_name": "millimeter of mercury", "dimension": "pressure", "to_si": 133.322},
    # Dimensionless counts (turns of a coil, oscillations, etc.) — these have
    # a real numeric value but no SI conversion.
    "turn": {"unit_symbol": "turn", "unit_name": "turn", "dimension": "count", "to_si": 1.0},
    "turns": {"unit_symbol": "turn", "unit_name": "turn", "dimension": "count", "to_si": 1.0},
    "turns/m": {"unit_symbol": "turn/m", "unit_name": "turns per meter", "dimension": "turn_density", "to_si": 1.0},
    "turn/m": {"unit_symbol": "turn/m", "unit_name": "turns per meter", "dimension": "turn_density", "to_si": 1.0},
    "turns/cm": {"unit_symbol": "turn/m", "unit_name": "turns per meter", "dimension": "turn_density", "to_si": 100.0},
    "oscillation": {"unit_symbol": "osc", "unit_name": "oscillation", "dimension": "count", "to_si": 1.0},
    "oscillations": {"unit_symbol": "osc", "unit_name": "oscillation", "dimension": "count", "to_si": 1.0},
}


def canonicalize_unit(unit_symbol: str) -> str:
    """Normalize equivalent textual unit spellings before lookup."""
    return (
        unit_symbol.strip()
        .replace("µ", "μ")
        .replace("Ω", "Ω")
        .replace("×", "*")
        .replace("Â²", "^2")
        .replace("²", "^2")
        .replace("?", "^2")
        .replace("·", "*")
        .replace("Â·", "*")
        .replace("ohms", "ohm")
        .replace("Ohms", "Ohm")
        .replace("m / s", "m/s")
        .replace("m/ s", "m/s")
        .replace("m /s", "m/s")
        .replace("V / m", "V/m")
        .replace("V/ m", "V/m")
        .replace("V /m", "V/m")
        .replace("N / C", "N/C")
        .replace("s²", "s^2")
    )


def get_unit_info(unit_symbol: str, context: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Return unit metadata, using context to disambiguate C."""
    unit = canonicalize_unit(unit_symbol)
    lowered = (context or "").lower()
    if unit == "C" and ("temperature" in lowered or "celsius" in lowered or "°c" in lowered):
        return UNIT_DEFINITIONS["°C"]
    return UNIT_DEFINITIONS.get(unit)


def normalize_quantity(
    value: float,
    unit_symbol: str,
    context: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Convert a value and unit to the SI value and canonical SI symbol."""
    info = get_unit_info(unit_symbol, context)
    if not info:
        return None, None
    dimension = str(info["dimension"])
    if dimension == "temperature" and info["unit_symbol"] == "°C":
        return value + 273.15, "K"
    return value * float(info["to_si"]), SI_UNIT_BY_DIMENSION.get(dimension)