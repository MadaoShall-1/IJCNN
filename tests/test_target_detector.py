import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.target_detector import detect_target


def test_current_target():
    assert detect_target("Find the current.") == ("I", "A")


def test_electric_field_target():
    assert detect_target("Determine the electric field strength.") == ("E", "V/m")


def test_capacitor_energy_target():
    assert detect_target("Calculate the energy stored in the capacitor.") == ("U_cap", "J")


def test_relative_error_target():
    assert detect_target("Find the percentage relative error.") == ("percent_error", None)


def test_ac_rlc_targets():
    assert detect_target("Find the impedance of the series RLC circuit.") == ("Z", "Ω")
    assert detect_target("Calculate the power factor.") == ("power_factor", None)
    assert detect_target("Determine the resonant frequency.") == ("f_res", "Hz")


def test_dielectric_target():
    assert detect_target("Find the new energy after inserting a dielectric.") == ("U_after", "J")


def test_stage03_targets():
    assert detect_target("Find the total energy of the system.") == ("U_total", "J")
    assert detect_target("Find the reduction in energy.") == ("delta_U", "J")
    assert detect_target("What is the relationship between E1 and E2?") == ("relation_E", None)
    assert detect_target("Find the value of R2.") == ("R2", "Ω")
    assert detect_target("Find the quality factor.") == ("Q_factor", None)
    assert detect_target("Find the number of turns per unit length.") == ("n_turns", None)
