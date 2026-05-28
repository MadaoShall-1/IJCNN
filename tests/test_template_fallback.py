import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.template_fallback import propose_step_plan


def q(dimension):
    return {"dimension": dimension}


def outputs(plan):
    seen = []
    for step in plan:
        seen.extend((step.get("output_var") or {}).keys())
    return seen


def test_ohms_law_plan():
    plan, confidence = propose_step_plan({"V": q("voltage"), "R": q("resistance")}, "I")
    assert confidence > 0.5
    assert "I = V / R" in plan[0]["output_var"]["I"]


def test_capacitor_energy_plan():
    plan, _ = propose_step_plan({"C_cap": q("capacitance"), "V": q("voltage")}, "U_cap")
    assert plan[-1]["output_var"]["U_cap"] == "U_cap"


def test_wave_speed_plan():
    plan, _ = propose_step_plan({"f": q("frequency"), "lambda": q("length")}, "v_wave")
    assert "v_wave" in plan[-1]["output_var"]


def test_series_rlc_impedance_plan():
    plan, confidence = propose_step_plan(
        {"R": q("resistance"), "L_ind": q("inductance"), "C_cap": q("capacitance"), "f": q("frequency")},
        "Z",
    )
    assert confidence > 0.5
    assert {"omega", "X_L", "X_C", "X", "Z"}.issubset(set(outputs(plan)))


def test_inductive_reactance_plan():
    plan, _ = propose_step_plan({"L_ind": q("inductance"), "f": q("frequency")}, "X_L")
    assert "X_L" in outputs(plan)


def test_capacitive_reactance_plan():
    plan, _ = propose_step_plan({"C_cap": q("capacitance"), "f": q("frequency")}, "X_C")
    assert "X_C" in outputs(plan)


def test_series_rlc_current_plan():
    plan, _ = propose_step_plan(
        {"R": q("resistance"), "L_ind": q("inductance"), "C_cap": q("capacitance"), "V_rms": q("voltage"), "f": q("frequency")},
        "I_rms",
    )
    assert "I_rms" in outputs(plan)


def test_lc_frequency_plan():
    plan, _ = propose_step_plan({"L_ind": q("inductance"), "C_cap": q("capacitance")}, "f_osc")
    assert "f_osc" in outputs(plan)


def test_lc_energy_fraction_plan():
    plan, _ = propose_step_plan({"q_over_Qmax": q("dimensionless")}, "magnetic_energy_fraction")
    assert "electric_energy_fraction" in outputs(plan)
    assert "magnetic_energy_fraction" in outputs(plan)


def test_percent_error_plan():
    plan, _ = propose_step_plan({"measured_value": q("acceleration"), "true_value": q("acceleration")}, "percent_error")
    assert "abs_error" in outputs(plan)
    assert "percent_error" in outputs(plan)


def test_absolute_error_plan():
    plan, _ = propose_step_plan({"measured_value": q("length"), "accepted_value": q("length")}, "abs_error")
    assert "abs_error" in outputs(plan)


def test_dielectric_connected_energy_plan():
    plan, _ = propose_step_plan(
        {"C_cap": q("capacitance"), "V": q("voltage"), "epsilon_r": q("dimensionless")},
        "U_after",
        ["battery_connected"],
    )
    assert "C_after" in outputs(plan)
    assert "U_after" in outputs(plan)


def test_dielectric_disconnected_voltage_plan():
    plan, _ = propose_step_plan(
        {"C_cap": q("capacitance"), "V": q("voltage"), "epsilon_r": q("dimensionless")},
        "V_after",
        ["battery_disconnected"],
    )
    assert "V_after" in outputs(plan)


def test_resistance_error_from_voltage_current_plan():
    plan, _ = propose_step_plan({"V": q("voltage"), "delta_V": q("voltage"), "I": q("current"), "delta_I": q("current")}, "rel_error")
    assert "R" in outputs(plan)
    assert "rel_error" in outputs(plan)
    assert any(step.get("template_name") == "resistance_error_from_voltage_current" for step in plan)


def test_dielectric_energy_delta_plans():
    connected, _ = propose_step_plan({"epsilon_r": q("dimensionless")}, "delta_U", ["battery_connected"])
    disconnected, _ = propose_step_plan({"epsilon_r": q("dimensionless")}, "delta_U", ["battery_disconnected"])
    assert "delta_U" in outputs(connected)
    assert "delta_U" in outputs(disconnected)


def test_ac_voltage_across_inductor_plan():
    plan, _ = propose_step_plan({"L_ind": q("inductance"), "I": q("current"), "f": q("frequency")}, "U_L")
    assert "X_L" in outputs(plan)
    assert "U_L" in outputs(plan)


def test_lc_electric_energy_fraction_relation_plan():
    plan, _ = propose_step_plan({}, "magnetic_energy_fraction", relations=[{"type": "ratio", "left": "U_E", "right": "U_total", "factor": 0.25}])
    assert "magnetic_energy_fraction" in outputs(plan)


def test_relative_motion_and_wave_templates():
    relative, _ = propose_step_plan({"v": q("velocity"), "v2": q("velocity"), "t": q("time")}, "d")
    wave, _ = propose_step_plan({"v_wave": q("velocity"), "f": q("frequency")}, "lambda")
    assert "relative_speed" in outputs(relative)
    assert "lambda" in outputs(wave)
