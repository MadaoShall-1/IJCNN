"""Keyword-based physics domain classification."""

from __future__ import annotations

from typing import Dict, List, Tuple


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "mechanics": ["mass", "acceleration", "force", "speed", "velocity", "kinetic energy", "momentum", "block", "car"],
    "electricity": [
        "capacitor", "capacitance", "charge", "voltage", "battery", "resistor", "resistance", "current", "circuit",
        "ac circuit", "alternating current", "rms", "impedance", "reactance", "power factor", "phase angle",
        "rlc", "series rlc", "resonance", "lc circuit", "oscillation", "dielectric", "relative permittivity",
        "move toward each other", "moving towards each other", "meet after",
    ],
    "electromagnetism": ["electric field", "coulomb force", "electric force", "magnetic field", "magnetic flux", "emf", "induced"],
    "thermodynamics": ["ideal gas", "pressure", "volume", "temperature", "heat", "thermal"],
    "waves": ["frequency", "wavelength", "wave speed", "period", "wave"],
    "optics": ["lens", "mirror", "focal length", "image height", "refraction", "reflection"],
    "fluids": ["fluid", "buoyancy", "density", "viscosity", "flow", "pressure"],
    "measurement_error": ["relative error", "absolute error", "percentage error", "random error", "measured value", "true value", "accepted value", "uncertainty"],
}


SUBDOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "ohms_law": ["resistor", "resistance", "current", "voltage"],
    "capacitors": ["capacitor", "capacitance", "energy stored"],
    "coulombs_law": ["point charges", "coulomb force", "electric force"],
    "kinematics": ["from rest", "accelerates", "velocity", "speed", "distance"],
    "wave_relation": ["frequency", "wavelength", "wave speed"],
    "ac_circuit": ["ac circuit", "alternating current", "rms", "impedance", "reactance", "power factor", "phase angle"],
    "rlc_circuit": ["rlc", "series rlc"],
    "resonance": ["resonance", "resonant frequency", "resonance frequency"],
    "lc_oscillation": ["lc circuit", "oscillation", "oscillating circuit", "maximum charge", "maximum current", "energy oscillates"],
    "measurement_error": ["absolute error", "relative error", "percentage error", "random error", "measured value", "true value", "accepted value", "uncertainty"],
    "dielectric_capacitor": ["dielectric", "dielectric constant", "relative permittivity", "inserted dielectric", "battery disconnected", "battery connected"],
    "error_propagation": ["uncertainty", "error in", "relative error", "percentage error", "±", "+/-"],
    "relative_motion": ["relative velocity", "moving towards each other", "move toward each other", "moving in opposite directions", "downstream", "upstream", "current of river", "meet after"],
    "multi_stage_motion": ["first half", "second half", "for the first", "then", "returns", "round trip", "average speed"],
    "capacitor_network": ["capacitors in series", "capacitors in parallel", "equivalent capacitance"],
    "electric_field_superposition": ["resultant electric field", "net electric field", "field at point", "e1 and e2"],
    "coulomb_vector_geometry": ["force on q3", "charges placed at vertices", "equilateral triangle", "angle between forces", "resultant electric force"],
    "rlc_resonance": ["resonance", "resonant frequency", "quality factor", "bandwidth"],
    "lc_energy_exchange": ["lc circuit", "maximum charge", "maximum current", "fraction of energy", "electric energy", "magnetic energy"],
}


def classify_domain(problem_text: str) -> Tuple[List[str], List[str], float]:
    """Classify broad and narrow domains using keyword scores."""
    lowered = problem_text.lower()
    scores = {
        domain: sum(1 for keyword in keywords if keyword in lowered)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    positive = {domain: score for domain, score in scores.items() if score > 0}
    if not positive:
        return ["unknown"], [], 0.2
    max_score = max(positive.values())
    domains = [domain for domain, score in positive.items() if score == max_score]
    if "electricity" in positive and "electromagnetism" in positive:
        domains = ["electromagnetism"] if positive["electromagnetism"] >= positive["electricity"] else ["electricity"]
    sub_domains = [
        subdomain
        for subdomain, keywords in SUBDOMAIN_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    confidence = min(0.95, 0.45 + 0.12 * max_score)
    return domains, sub_domains, confidence
