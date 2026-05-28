import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.main import parse_problem


def assert_pass(parse):
    assert parse["metadata"]["verifier_status"] == "PASS", parse["metadata"]["verifier_errors"]


def test_ohms_law_end_to_end():
    parsed = parse_problem("A 10 Ω resistor is connected to a 5 V battery. Find the current.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "I"


def test_capacitor_energy_end_to_end():
    parsed = parse_problem("A 4 μF capacitor is connected to a 12 V battery. Calculate the energy stored in the capacitor.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "U_cap"


def test_kinematics_end_to_end():
    parsed = parse_problem("A car starts from rest and accelerates uniformly at 2 m/s^2 for 5 s. Find its final velocity.")
    assert_pass(parsed)
    assert parsed["known_quantities"]["v_0"]["value"] == 0


def test_wave_end_to_end():
    parsed = parse_problem("A wave has frequency 50 Hz and wavelength 2 m. Find its speed.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "v_wave"


def test_coulomb_force_end_to_end():
    parsed = parse_problem("Two point charges q1 = 6 nC and q2 = 3 nC are separated by 4 cm. Determine the electric force between them.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "F_e"


def test_series_rlc_impedance_end_to_end():
    parsed = parse_problem("A series RLC circuit has R = 10 Ω, L = 0.2 H, C = 100 μF and frequency 50 Hz. Find the impedance.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "Z"
    assert "series_rlc_impedance" in parsed["metadata"]["used_template_names"]


def test_inductive_reactance_end_to_end():
    parsed = parse_problem("A coil with inductance 0.5 H is connected to an AC source of frequency 60 Hz. Calculate the inductive reactance.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "X_L"


def test_capacitive_reactance_end_to_end():
    parsed = parse_problem("A capacitor of 20 μF is connected to an AC source of frequency 50 Hz. Find the capacitive reactance.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "X_C"


def test_series_rlc_rms_current_end_to_end():
    parsed = parse_problem("A series RLC circuit has R = 8 Ω, L = 0.1 H, C = 200 μF and rms voltage 120 V at 60 Hz. Find the rms current.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "I_rms"


def test_lc_frequency_end_to_end():
    parsed = parse_problem("An LC circuit has L = 0.5 H and C = 20 μF. Find the oscillation frequency.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "f_osc"


def test_lc_energy_fraction_end_to_end():
    parsed = parse_problem("In an LC circuit, the charge is half the maximum charge. What fraction of the energy is stored in the magnetic field?")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "magnetic_energy_fraction"
    assert "q_over_Qmax" in parsed["known_quantities"]


def test_percent_error_end_to_end():
    parsed = parse_problem("The measured value is 9.8 m/s^2 and the true value is 10 m/s^2. Find the percentage relative error.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "percent_error"


def test_absolute_error_end_to_end():
    parsed = parse_problem("The measured length is 4.9 cm and the accepted value is 5.0 cm. Find the absolute error.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "abs_error"


def test_dielectric_connected_energy_end_to_end():
    parsed = parse_problem("A capacitor of capacitance 4 μF is connected to a 12 V battery. A dielectric with dielectric constant 3 is inserted while the battery remains connected. Find the new energy.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "U_after"
    assert "battery_connected" in parsed["conditions"]


def test_dielectric_disconnected_voltage_end_to_end():
    parsed = parse_problem("A capacitor of capacitance 4 μF is charged to 12 V and then disconnected from the battery. A dielectric with dielectric constant 3 is inserted. Find the new voltage.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "V_after"
    assert "battery_disconnected" in parsed["conditions"]


def test_uncertainty_relative_error_end_to_end():
    parsed = parse_problem("U = 6.0 ± 0.1 V and I = 2.0 ± 0.05 A. Find the relative error of R = U/I.")
    assert_pass(parsed)
    assert "resistance_error_from_voltage_current" in parsed["metadata"]["used_template_names"]


def test_dielectric_energy_reduction_end_to_end():
    parsed = parse_problem("A capacitor is charged to voltage V and disconnected. A dielectric with dielectric constant 4 is inserted. Find the reduction in energy.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "delta_U"
    assert "battery_disconnected" in parsed["conditions"]


def test_dielectric_energy_increase_end_to_end():
    parsed = parse_problem("A capacitor remains connected to a battery while a dielectric constant 3 is inserted. Find the increase in energy.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "delta_U"
    assert "battery_connected" in parsed["conditions"]


def test_ac_voltage_across_inductor_end_to_end():
    parsed = parse_problem("A series RLC circuit has L = 0.2 H, current 3 A, and frequency 50 Hz. Find the voltage across the inductor.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "U_L"


def test_lc_energy_fraction_relation_end_to_end():
    parsed = parse_problem("In an LC circuit, the electric energy is 1/4 of the total energy. Find the fraction of energy in the magnetic field.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "magnetic_energy_fraction"


def test_relative_motion_end_to_end():
    parsed = parse_problem("Two vehicles move toward each other at 56 km/h and 36 km/h and meet after 2 hours. Find the distance between them.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "d"


def test_wave_wavelength_end_to_end():
    parsed = parse_problem("A wave has frequency 50 Hz and speed 100 m/s. Find the wavelength.")
    assert_pass(parsed)
    assert parsed["unknown_quantity"] == "lambda"
