import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.unit_normalizer import get_unit_info, normalize_quantity


def test_normalizes_common_units():
    assert normalize_quantity(4, "μF") == (4e-6, "F")
    assert normalize_quantity(72, "km/h") == (20.0, "m/s")
    assert normalize_quantity(3, "cm") == (0.03, "m")


def test_c_disambiguation():
    assert get_unit_info("C", "charge of 2 C")["dimension"] == "charge"
    assert get_unit_info("C", "temperature is 20 C")["dimension"] == "temperature"


def test_area_and_compound_units():
    assert normalize_quantity(4, "cm²") == pytest.approx((4e-4, "m^2"))
    assert get_unit_info("C/m²")["dimension"] == "surface_charge_density"
    assert get_unit_info("N·m²/C²")["dimension"] == "coulomb_constant_unit"


def test_small_scientific_notation_extraction():
    from parser.rule_extractor import extract_quantities

    quantities = extract_quantities("A charge is 5.10^-9 C.")
    assert quantities["q"]["value"] == pytest.approx(5e-9)
