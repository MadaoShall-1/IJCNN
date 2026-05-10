# parser/rule_preparser.py

import re
from typing import Dict, Any, Optional


TOPIC_KEYWORDS = {
    "electric_circuit": [
        "resistor", "resistance", "circuit", "voltage", "current",
        "ohm", "series", "parallel", "kirchhoff", "kvl", "kcl"
    ],
    "dynamics": [
        "force", "net force", "mass", "acceleration", "newton"
    ],
    "kinematics": [
        "velocity", "speed", "acceleration", "distance", "displacement",
        "time", "projectile"
    ],
    "energy": [
        "kinetic energy", "potential energy", "work", "power",
        "mechanical energy", "conservation of energy", "spring",
        "frictionless", "incline", "compression", "compresses"
    ],
    "momentum": [
        "momentum", "collision", "impulse"
    ],
    "waves": [
        "frequency", "wavelength", "wave speed", "period"
    ],
    "thermodynamics": [
        "heat", "temperature", "thermal", "specific heat",
        "gas", "pressure", "volume"
    ],
    "optics": [
        "lens", "mirror", "focal length", "image distance",
        "refraction", "reflection"
    ],
}


TARGET_PATTERNS = [
    ("equivalent_resistance", ["equivalent resistance", "total resistance", "effective resistance"]),
    ("voltage", ["voltage", "potential difference"]),
    ("current", ["current"]),
    ("net_force", ["net force", "force"]),
    ("acceleration", ["acceleration"]),
    ("velocity", ["velocity", "speed"]),
    ("distance", ["distance", "displacement"]),
    ("kinetic_energy", ["kinetic energy"]),
    ("potential_energy", ["potential energy"]),
    ("work", ["work done", "work"]),
    ("power", ["power"]),
    ("momentum", ["momentum"]),
    ("frequency", ["frequency"]),
    ("wavelength", ["wavelength"]),

]


UNIT_BY_TARGET = {
    "equivalent_resistance": "ohm_or_symbolic_resistance",
    "voltage": "V",
    "current": "A",
    "net_force": "N",
    "acceleration": "m/s^2",
    "velocity": "m/s",
    "distance": "m",
    "kinetic_energy": "J",
    "potential_energy": "J",
    "work": "J",
    "power": "W",
    "momentum": "kg*m/s",
    "frequency": "Hz",
    "wavelength": "m",
    "maximum_compression":"m",
    "resistance":"ohm_or_symbolic_resistance",
}


UNKNOWN_BY_TARGET = {
    "equivalent_resistance": ["R_eq"],
    "resistance":["R"],
    "voltage": ["V"],
    "current": ["I"],
    "net_force": ["F_net"],
    "acceleration": ["a"],
    "velocity": ["v"],
    "distance": ["d"],
    "kinetic_energy": ["KE"],
    "potential_energy": ["PE"],
    "work": ["W"],
    "power": ["P"],
    "momentum": ["p"],
    "frequency": ["f"],
    "wavelength": ["lambda"],
    "maximum_compression":["x"],
}


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("\u02db", "^2")
        .replace("\u00b2", "^2")
        .replace("\u0142", "^3")
        .replace("\u00b3", "^3")
    )


def detect_topic(text: str) -> tuple[str, float]:
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scores[topic] = score

    if not scores:
        return "unknown", 0.2

    best_topic = max(scores, key=scores.get)
    confidence = min(0.95, 0.45 + 0.12 * scores[best_topic])
    return best_topic, confidence


def detect_target_quantity(text: str) -> tuple[str, float]:
    question_patterns = [
        ("equivalent_resistance", [r"what is .*equivalent resistance", r"calculate .*equivalent resistance", r"find .*equivalent resistance"]),
        ("resistance", [r"what is .*resistance", r"calculate .*resistance", r"find .*resistance"]),
        ("voltage", [r"what is .*voltage", r"calculate .*voltage", r"find .*voltage"]),
        ("current", [r"what is .*current", r"calculate .*current", r"find .*current"]),
        ("net_force", [r"what is .*net force", r"calculate .*net force", r"find .*net force"]),
        ("acceleration", [r"what is .*acceleration", r"calculate .*acceleration", r"find .*acceleration"]),
        ("velocity", [r"what is .*velocity", r"what is .*speed", r"calculate .*velocity", r"find .*velocity"]),
        ("maximum_compression", [r"what is .*maximum compression", r"find .*maximum compression"]),
    ]

    for target, patterns in question_patterns:
        for p in patterns:
            if re.search(p, text):
                return target, 0.9

    for target, patterns in TARGET_PATTERNS:
        for p in patterns:
            if p in text:
                return target, 0.75

    if "what is" in text or "calculate" in text or "find" in text:
        return "unknown", 0.45

    return "unknown", 0.2

def detect_question_type_and_options(raw_text: str) -> tuple[str, Optional[Dict[str, str]], str]:
    # Handles formats like:
    # A. 3r B. r/3 C. 2r D. r
    option_pattern = r"([A-D])[\.\)]\s*([^A-D\n]+)"
    matches = re.findall(option_pattern, raw_text)

    if len(matches) >= 2:
        options = {letter: value.strip() for letter, value in matches}
        return "multiple_choice", options, "multiple_choice"

    lowered = raw_text.lower().strip()

    if lowered.startswith(("can ", "is ", "does ", "do ", "will ", "would ")):
        return "yes_no_uncertain", None, "yes_no_uncertain"

    if re.search(r"\b[a-zA-Z]\b", raw_text) and any(
        word in lowered for word in ["resistance", "resistor", "voltage", "current"]
    ):
        known = extract_known_variables(raw_text)
        if any(key in known for key in ["voltage", "current", "resistance"]):
            return "calculation", None, "numeric_value"
        return "calculation", None, "symbolic_expression"

    return "calculation", None, "numeric_value"


def extract_known_variables(raw_text: str) -> Dict[str, str]:
    text = (
        raw_text
        .replace("\u02db", "^2")
        .replace("\u00b2", "^2")
        .replace("\u0142", "^3")
        .replace("\u00b3", "^3")
    )
    known: Dict[str, str] = {}

    mass_match = re.search(r"(\d+(?:\.\d+)?)\s*kg", text, re.IGNORECASE)
    if mass_match:
        known["mass"] = f"{mass_match.group(1)} kg"

    acc_match = re.search(
        r"(\d+(?:\.\d+)?)\s*m\s*/\s*s\^?2",
        text,
        re.IGNORECASE
    )
    if acc_match:
        known["acceleration"] = f"{acc_match.group(1)} m/s^2"

    vel_match = re.search(
        r"(\d+(?:\.\d+)?)\s*m\s*/\s*s(?!\^?2)",
        text,
        re.IGNORECASE
    )
    if vel_match:
        known["velocity_or_speed"] = f"{vel_match.group(1)} m/s"

    time_match = re.search(r"(\d+(?:\.\d+)?)\s*s\b", text, re.IGNORECASE)
    if time_match:
        known["time"] = f"{time_match.group(1)} s"

    if re.search(r"each resistor.*\br\b|resistance of r", text, re.IGNORECASE):
        known["resistance_each"] = "r"

    resistance_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(ohm|Ω)",
        text,
        re.IGNORECASE
    )
    if resistance_match:
        known["resistance"] = f"{resistance_match.group(1)} ohm"

    voltage_match = re.search(r"(\d+(?:\.\d+)?)\s*V\b", text)
    if voltage_match:
        known["voltage"] = f"{voltage_match.group(1)} V"

    current_match = re.search(r"(\d+(?:\.\d+)?)\s*A\b", text)
    if current_match:
        known["current"] = f"{current_match.group(1)} A"

    return known


def detect_subtopic(topic: str, target: str, text: str) -> str:
    if topic == "electric_circuit":
        if target == "equivalent_resistance":
            return "equivalent_resistance"
        if "kirchhoff" in text or "kvl" in text or "kcl" in text:
            return "kirchhoff_laws"
        if "ohm" in text or (
            target in {"resistance", "voltage", "current"}
            and re.search(r"\b\d+(?:\.\d+)?\s*v\b", text)
            and re.search(r"\b\d+(?:\.\d+)?\s*a\b", text)
        ):
            return "ohms_law"

    if topic == "dynamics" and target == "net_force":
        return "newton_second_law"

    if topic == "energy":
        if target in ["kinetic_energy", "potential_energy"]:
            return target
        if "conservation" in text or (
            target == "maximum_compression"
            and any(w in text for w in ["spring", "frictionless", "incline"])
        ):
            return "energy_conservation"

    return "unknown"


def requires_diagram_reasoning(topic: str, text: str) -> bool:
    diagram_words = [
        "following circuit",
        "diagram",
        "figure",
        "shown",
        "image",
        "pictured",
        "below",
        "above",
    ]
    if any(w in text for w in diagram_words):
        return True

    if topic == "electric_circuit" and any(w in text for w in ["series", "parallel"]):
        return True

    return False


def rule_preparse(raw_text: str) -> Dict[str, Any]:
    text = normalize_text(raw_text)

    topic, topic_conf = detect_topic(text)
    target, target_conf = detect_target_quantity(text)
    question_type, options, answer_type = detect_question_type_and_options(raw_text)
    known_variables = extract_known_variables(raw_text)
    subtopic = detect_subtopic(topic, target, text)

    hints = {
        "possible_topic": topic,
        "possible_subtopic": subtopic,
        "possible_target_quantity": target,
        "known_variables": known_variables,
        "unknown_variables": UNKNOWN_BY_TARGET.get(target, []),
        "question_type": question_type,
        "answer_type": answer_type,
        "unit_expected": UNIT_BY_TARGET.get(target, "unknown"),
        "requires_diagram_reasoning": requires_diagram_reasoning(topic, text),
        "requires_formula_retrieval": True,
        "answer_options": options,
        "rule_confidence": round((topic_conf + target_conf) / 2, 2)
    }

    return hints
