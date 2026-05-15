"""Question-target detection for Stage 0 parses."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


TARGETS = [
    ("how many times greater", "ratio", None),
    ("how many times smaller", "ratio", None),
    ("how many times", "ratio", None),
    ("times greater", "ratio", None),
    ("times smaller", "ratio", None),
    ("ratio between", "ratio", None),
    ("ratio of", "ratio", None),
    ("percentage relative error", "percent_error", None),
    ("percent relative uncertainty", "percent_error", None),
    ("percentage error", "percent_error", None),
    ("percentage loss", "percent_error", None),
    ("relative error", "rel_error", None),
    ("relative uncertainty", "rel_error", None),
    ("absolute error", "abs_error", None),
    ("mean value", "mean_value", None),
    ("average value", "mean_value", None),
    ("measured value", "measured_value", None),
    ("true value", "true_value", None),
    ("new energy after inserting dielectric", "U_after", "J"),
    ("energy after inserting dielectric", "U_after", "J"),
    ("voltage after inserting dielectric", "V_after", "V"),
    ("charge after inserting dielectric", "Q_after", "C"),
    ("capacitance after inserting dielectric", "C_after", "F"),
    ("new capacitance", "C_after", "F"),
    ("new voltage", "V_after", "V"),
    ("new charge", "Q_after", "C"),
    ("new energy", "U_after", "J"),
    ("total energy of the system", "U_total", "J"),
    ("total oscillation energy", "U_total", "J"),
    ("total energy", "U_total", "J"),
    ("remaining energy", "U_after", "J"),
    ("energy after", "U_after", "J"),
    ("reduction in energy", "delta_U", "J"),
    ("decrease in energy", "delta_U", "J"),
    ("increase in energy", "delta_U", "J"),
    ("energy loss", "delta_U", "J"),
    ("energy stored in the capacitor", "U_cap", "J"),
    ("energy stored in capacitor", "U_cap", "J"),
    ("energy stored in magnetic field", "U_B", "J"),
    ("energy stored in electric field", "U_E", "J"),
    ("magnetic field energy", "U_B", "J"),
    ("electric field energy", "U_E", "J"),
    ("wc", "U_E", "J"),
    ("w_c", "U_E", "J"),
    ("wl", "U_B", "J"),
    ("w_l", "U_B", "J"),
    ("fraction of magnetic energy", "magnetic_energy_fraction", None),
    ("fraction of electric energy", "electric_energy_fraction", None),
    ("fraction of energy", "energy_fraction", None),
    ("power factor", "power_factor", None),
    ("efficiency of the circuit", "efficiency", None),
    ("efficiency", "efficiency", None),
    ("quality factor", "Q_factor", None),
    ("random error", "random_error", None),
    ("percentage relative uncertainty", "percent_error", None),
    ("dielectric constant", "epsilon_r", None),
    ("relative permittivity", "epsilon_r", None),
    ("total impedance", "Z", "Ω"),
    ("impedance", "Z", "Ω"),
    ("inductive reactance", "X_L", "Ω"),
    ("capacitive reactance", "X_C", "Ω"),
    ("reactance", "X", "Ω"),
    ("turn density", "n_turns_per_meter", None),
    ("number of turns per meter length", "n_turns_per_meter", None),
    ("number of turns per unit length", "n_turns_per_meter", None),
    ("amplitude of oscillation", "A_amp", None),
    ("amplitude", "A_amp", None),
    ("equation of motion", "equation_of_motion", None),
    ("relationship between E1 and E2", "relation_E", None),
    ("relation between E1 and E2", "relation_E", None),
    ("compare E1 and E2", "relation_E", None),
    ("ratio of E1 to E2", "relation_E", None),
    ("relationship between", "relation_generic", None),
    ("spring constant", "k", None),
    ("force constant", "k", None),
    ("total power of the circuit", "P_total", "W"),
    ("total power", "P_total", "W"),
    ("power consumption", "P", "W"),
    ("power of each lamp", "P_each", "W"),
    ("power of the circuit at resonance", "P", "W"),
    ("maximum power dissipation", "P_max", "W"),
    ("power p consumed", "P", "W"),
    ("average power", "P_avg", "W"),
    ("phase angle", "phi", "rad"),
    ("resonant angular frequency", "omega_0", "rad/s"),
    ("angular frequency", "omega", "rad/s"),
    ("resonance frequency", "f_res", "Hz"),
    ("resonant frequency", "f_res", "Hz"),
    ("oscillation frequency", "f_osc", "Hz"),
    ("frequency of oscillation", "f_osc", "Hz"),
    ("oscillation period", "T_osc", "s"),
    ("period of oscillation", "T_osc", "s"),
    ("maximum charge", "Q_max", "C"),
    ("maximum current", "I_max", "A"),
    ("rms current", "I_rms", "A"),
    ("current in the circuit", "I", "A"),
    ("rms voltage", "V_rms", "V"),
    ("voltage across the inductor", "U_L", "V"),
    ("voltage across inductor", "U_L", "V"),
    ("voltage across the capacitor", "U_C", "V"),
    ("voltage across capacitor", "U_C", "V"),
    ("voltage across the resistor", "U_R", "V"),
    ("voltage across resistor", "U_R", "V"),
    ("induced emf", "emf", "V"),
    ("electromotive force", "emf", "V"),
    ("electric field strength", "E", "V/m"),
    ("electric field", "E", "V/m"),
    ("coulomb force", "F_e", "N"),
    ("net electric force", "F_net", "N"),
    ("resultant electric force", "F_net", "N"),
    ("total electric force", "F_net", "N"),
    ("force acting on q3", "F_on_q3", "N"),
    ("force on q3", "F_on_q3", "N"),
    ("magnitude of the electric force", "F_e", "N"),
    ("electric force", "F_e", "N"),
    ("force acting on", "F_e", "N"),
    ("potential difference", "V", "V"),
    ("power consumed", "P", "W"),
    ("power dissipated", "P", "W"),
    ("charge stored", "Q", "C"),
    ("charge accumulated", "Q", "C"),
    ("charge on the capacitor", "Q", "C"),
    ("charge of the capacitor", "Q", "C"),
    ("sign and magnitude of q", "q", "C"),
    ("magnitude of q", "q", "C"),
    ("value of q", "q", "C"),
    ("value of q1", "q1", "C"),
    ("value of q2", "q2", "C"),
    ("value of q3", "q3", "C"),
    ("electric charge", "q", "C"),
    ("equivalent capacitance", "C_eq", "F"),
    ("capacitance", "C_cap", "F"),
    ("magnetic flux", "Phi_B", "Wb"),
    ("total flux linkage", "Phi_link", "Wb"),
    ("flux linkage", "Phi_link", "Wb"),
    ("magnetic field", "B", "T"),
    ("direction", "theta", "rad"),
    ("wave speed", "v_wave", "m/s"),
    ("average speed", "v_avg", "m/s"),
    ("wavelength", "lambda", "m"),
    ("frequency", "f", "Hz"),
    ("period", "T_period", "s"),
    ("inductance", "L_ind", "H"),
    ("final velocity", "v_final", "m/s"),
    ("final speed", "v_final", "m/s"),
    ("initial velocity", "v_0", "m/s"),
    ("initial speed", "v_0", "m/s"),
    ("net force", "F_net", "N"),
    ("resultant force", "F_net", "N"),
    # Energy phrases — longer first wins under length-sorted match below
    ("mechanical energy lost", "delta_E_mech", "J"),
    ("mechanical energy of the airplane", "E_mech", "J"),
    ("mechanical energy of the ball", "E_mech", "J"),
    ("mechanical energy of the oscillating object", "E_mech", "J"),
    ("mechanical energy", "E_mech", "J"),
    ("remaining electrical field energy", "U_after", "J"),
    ("remaining electric field energy", "U_after", "J"),
    ("stored energy", "U_E", "J"),
    ("kinetic energy", "KE", "J"),
    ("potential energy", "PE", "J"),
    # Mass and temperature targets
    ("mass of hot water", "m_water", "kg"),
    ("mass of oxygen gas", "m_gas", "kg"),
    ("mass of the ice cube", "m_ice", "kg"),
    ("mass of the object", "m_object", "kg"),
    ("mass of the ball", "m_object", "kg"),
    ("mass of the body", "m_object", "kg"),
    ("mass", "m", "kg"),
    ("specific latent heat of fusion", "L_fusion", "J/kg"),
    ("specific latent heat", "L_latent", "J/kg"),
    ("temperature of the furnace", "T_furnace", "°C"),
    ("initial temperature", "T_initial", "°C"),
    ("final temperature", "T_final", "°C"),
    # Optics
    ("focal length", "f_focal", "m"),
    ("height of the image", "h_image", "m"),
    # Geometry / path
    ("length of the path", "d_path", "m"),
    ("height above the ground", "h", "m"),
    # Circuit characteristic (the question is about resonance type / Q / etc.;
    # mapping to power_factor keeps it on the dimensionless branch and lets the
    # verifier route it via _verify_non_numeric without spurious target errors.
    ("circuit's characteristic", "power_factor", None),
    # Single-word fallbacks (lowest-priority because shortest after sort)
    ("power", "P", "W"),
    ("the power", "P", "W"),
    ("momentum", "p", None),
    ("acceleration", "a", "m/s^2"),
    ("displacement", "d", "m"),
    ("separation", "r", "m"),
    ("distance", "d", "m"),
    ("velocity", "v", "m/s"),
    ("speed", "v", "m/s"),
    ("charge", "q", "C"),
    ("angle", "theta", "rad"),
    ("force", "F_net", "N"),
    ("current", "I", "A"),
    ("voltage", "V", "V"),
    ("resistance", "R", "Ω"),
    ("work", "W", "J"),
    ("time", "t", "s"),
]

# Strong cues unambiguously introduce a question subject.
PRIMARY_CUES = (
    "find",
    "calculate",
    "determine",
    "compute",
    "evaluate",
    "derive",
    "what is",
    "what are",
    "what will",
    "what value",
    "what kind",
    "what must",
    "what capacitor",
    "what inductor",
    "what resistor",
    "what l",
    "what c",
    "what r",
    "how much",
    "how many",
    "how long",
    "how far",
    "how fast",
    "how high",
    "express",
    "state the",
)

# Weaker cues — these words frequently appear as relative pronouns or
# adverbial modifiers ("a charge ... which is placed at M"). We only
# use them when no PRIMARY_CUES were found.
SECONDARY_CUES = ("which",)

# Combined for backward compatibility with callers that import CUES.
CUES = PRIMARY_CUES + SECONDARY_CUES

# Cue patterns that hint at a numeric target without a leading verb.
# Used when neither primary nor secondary CUES gives positions but the
# sentence is still a question. We back up a window before the soft cue
# match so the question phrase itself stays inside search_text.
SOFT_CUE_PATTERNS = (
    r"\bvalue\s+of\b",
    r"\bis\s+needed\b",
    r"\bis\s+required\b",
    r"\bshould\s+be\b",
    r"\bis\s+at\s+its\s+maximum\b",
)
SOFT_CUE_BACK_WINDOW = 60  # characters before the soft-cue match to keep


def detect_target(problem_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect the unknown quantity symbol and expected unit from question phrases."""
    lowered = problem_text.lower()
    # Strategy:
    #   1) Strong cues (find/calculate/what is/...): start search_text at the
    #      FIRST occurrence so a single problem with one cue selects it
    #      unambiguously. Using rfind here previously broke patterns like
    #      "Find the force ... a charge q0 ... which is placed at M" where the
    #      relative pronoun "which" appears after the real cue.
    #   2) Secondary cues (which): only consulted when no primary cue exists.
    #   3) Soft cues (value of / is needed / ...): only when 1) and 2) failed.
    #   4) Last resort: window before final '?'.
    primary_positions = [lowered.find(cue) for cue in PRIMARY_CUES if cue in lowered]
    primary_positions = [pos for pos in primary_positions if pos >= 0]
    if primary_positions:
        search_text = lowered[min(primary_positions):]
    else:
        secondary_positions = [lowered.find(cue) for cue in SECONDARY_CUES if cue in lowered]
        secondary_positions = [pos for pos in secondary_positions if pos >= 0]
        if secondary_positions:
            search_text = lowered[min(secondary_positions):]
        else:
            soft_positions: List[int] = []
            for pattern in SOFT_CUE_PATTERNS:
                for match in re.finditer(pattern, lowered):
                    soft_positions.append(match.start())
            if soft_positions:
                anchor = max(soft_positions)
                search_text = lowered[max(0, anchor - SOFT_CUE_BACK_WINDOW):]
            elif "?" in lowered:
                qmark = lowered.rfind("?")
                search_text = lowered[max(0, qmark - 120): qmark + 1]
            else:
                return None, None

    if "how many times" in search_text or "times greater" in search_text or "times smaller" in search_text:
        return "ratio", None

    # Implicit "how long / how far / how high / how fast" — the question
    # itself names the dimension via the adverb, no noun cue follows.
    if re.search(r"\bhow\s+long\b", search_text, re.IGNORECASE):
        return "t", "s"
    if re.search(r"\bhow\s+far\b", search_text, re.IGNORECASE):
        # "how far ... above the ground" is a height; otherwise distance.
        if "above the ground" in search_text or "above the floor" in search_text:
            return "h", "m"
        return "d", "m"
    if re.search(r"\bhow\s+high\b", search_text, re.IGNORECASE):
        return "h", "m"
    if re.search(r"\bhow\s+fast\b", search_text, re.IGNORECASE):
        return "v", "m/s"

    if re.search(r"\bfind\s+C['′]|\bwhat\s+is\s+C['′]", problem_text, re.IGNORECASE):
        return "C_after", "F"

    # Domain-aware short-circuits: when the question itself names "P / C / L"
    # etc., return the dimensionally-correct target. Order matters; more
    # specific phrasing wins.
    if re.search(r"\b(?:find|calculate|determine|what\s+is)\s+p\b", search_text, re.IGNORECASE):
        return "P", "W"
    if re.search(r"\b(?:find|calculate|determine|what\s+is)\s+c\b", search_text, re.IGNORECASE):
        return "C_cap", "F"
    if "direction" in search_text or "orientation" in search_text:
        return "theta", "rad"
    if re.search(r"\benergy\b(?:\s*\([^)]*\))?\s+stored\b", search_text) and "capacitor" in lowered:
        return "U_cap", "J"
    if re.search(r"\b(?:stored\s+)?magnetic(?:\s+field)?\s+energy\b", search_text):
        return "U_B", "J"
    if re.search(r"\belectric(?:\s+field)?\s+energy\b", search_text):
        return "U_E", "J"
    if "maximum magnetic energy" in search_text:
        return "U_B", "J"

    if "fraction" in search_text and "magnetic field" in search_text:
        return "magnetic_energy_fraction", None
    if "fraction" in search_text and "electric field" in search_text:
        return "electric_energy_fraction", None

    # "Find/Calculate/Determine/Compute/What is/What value <single-letter>"
    # The 'r' case must return R (resistance) when the question is "what is R?"
    # in a circuit context, not the lowercase 'r' (length). When the context
    # is clearly geometric (radius/distance), the loop further down still
    # picks the right phrase first.
    find_symbol = re.search(
        r"\b(?:find|calculate|determine|compute|what\s+is|what\s+are|"
        r"what\s+value\s+of)\s+"
        r"(q\d*|r\d*|i\d*|v\d*|u_l|u_c|u_r|ul|uc|ur|u\d*|l\d*|c\d*|f_?0?|"
        r"z_l|z_c|zl|zc|z|b|e|p|t|cos[\u03c6\u03d5]?|cosphi)\b",
        search_text,
    )
    if find_symbol:
        symbol = find_symbol.group(1)
        if symbol.startswith("q"):
            return symbol, "C"
        if symbol.startswith("r"):
            # 'r' as a target in a circuit-style question almost always means
            # resistance R (Ω). In a geometric problem the TARGETS loop's
            # 'separation/radius/distance' phrases will have matched first.
            return "R", "Ω"
        if symbol.startswith("i"):
            return "I", "A"
        if symbol == "v" or re.fullmatch(r"v\d+", symbol):
            return "V", "V"
        if symbol in {"u", "u_l", "u_c", "u_r", "ul", "uc", "ur"} or re.fullmatch(r"u\d+", symbol):
            mapping = {
                "u_l": "U_L", "u_c": "U_C", "u_r": "U_R",
                "ul": "U_L", "uc": "U_C", "ur": "U_R",
            }
            return mapping.get(symbol, "V"), "V"
        if symbol == "l" or re.fullmatch(r"l\d+", symbol):
            return "L_ind", "H"
        if symbol == "c" or re.fullmatch(r"c\d+", symbol):
            return "C_cap", "F"
        if symbol in {"f", "f0", "f_0"}:
            return "f_res", "Hz"
        if symbol in {"z", "z_l", "z_c", "zl", "zc"}:
            mapping = {"z_l": "X_L", "z_c": "X_C", "zl": "X_L", "zc": "X_C"}
            return mapping.get(symbol, "Z"), "Ω"
        if symbol == "b":
            return "B", "T"
        if symbol == "e":
            return "E", "V/m"
        if symbol == "p":
            return "P", "W"
        if symbol == "t":
            return "t", "s"
        if symbol.startswith("cos"):
            return "power_factor", None

    value_symbol = re.search(
        r"\bvalue\s+of\s+"
        r"(R\d*|q\d*|q|k|I\d*|U_L|U_C|U_R|U\d*|L\d*|C\d*|Z_L|Z_C|Z|B)\b",
        search_text,
        re.IGNORECASE,
    )
    if value_symbol:
        symbol = value_symbol.group(1)
        sl = symbol.lower()
        if sl.startswith("r"):
            return symbol[0].upper() + symbol[1:], "Ω"
        if sl.startswith("q"):
            return symbol, "C"
        if sl.startswith("i"):
            return symbol if symbol[0].isupper() else symbol.upper(), "A"
        if sl in {"u_l", "u_c", "u_r"}:
            return {"u_l": "U_L", "u_c": "U_C", "u_r": "U_R"}[sl], "V"
        if sl.startswith("u"):
            return "V", "V"
        if sl.startswith("l"):
            return "L_ind", "H"
        if sl.startswith("c"):
            return "C_cap", "F"
        if sl in {"z", "z_l", "z_c"}:
            return {"z_l": "X_L", "z_c": "X_C"}.get(sl, "Z"), "Ω"
        if sl == "b":
            return "B", "T"
        return symbol, None

    if re.search(r"\berror (?:of|in)\s*R\b", search_text, re.IGNORECASE):
        return "delta_R", "Ω"

    # "What ... value of L / C is needed" — covers
    # "What value of C is needed to resonate ...", "what L is needed",
    # "what inductor L is needed", etc.
    needed_match = re.search(
        r"\bwhat\s+"
        r"(?:value\s+of\s+|capacitor\s+|inductor\s+|resistor\s+|frequency\s+)?"
        r"(L|C|R|f|U|V|I)(?:\s*[\w\d]*)?\s+"
        r"(?:is\s+(?:needed|required|chosen)|should\s+be(?:\s+chosen)?|must\s+be)\b",
        search_text,
        re.IGNORECASE,
    )
    # Reversed phrasing: "what must L be?", "what should C be?"
    if not needed_match:
        needed_match = re.search(
            r"\bwhat\s+(?:must|should)\s+(L|C|R|f|U|V|I)\s+be\b",
            search_text,
            re.IGNORECASE,
        )
    if needed_match:
        sym = needed_match.group(1).upper()
        mapping = {
            "L": ("L_ind", "H"),
            "C": ("C_cap", "F"),
            "R": ("R", "Ω"),
            "F": ("f_res", "Hz"),
            "U": ("V", "V"),
            "V": ("V", "V"),
            "I": ("I", "A"),
        }
        if sym in mapping:
            return mapping[sym]

    # "What capacitor is needed", "What inductor is needed" — no symbol given.
    appliance_match = re.search(
        r"\bwhat\s+(capacitor|inductor|resistor|frequency)\b"
        r"(?:\s+\w+){0,3}\s+(?:is\s+(?:needed|required|chosen)|should\s+be)\b",
        search_text,
        re.IGNORECASE,
    )
    if appliance_match:
        word = appliance_match.group(1).lower()
        mapping = {
            "capacitor": ("C_cap", "F"),
            "inductor": ("L_ind", "H"),
            "resistor": ("R", "Ω"),
            "frequency": ("f_res", "Hz"),
        }
        return mapping[word]

    # "how many meters / kilometers / seconds / ..." — implicit measurement query.
    how_many_unit = re.search(
        r"\bhow\s+many\s+(meters|metres|kilometers|kilometres|"
        r"seconds|minutes|hours|grams|kilograms|joules|watts|"
        r"volts|amps|amperes|ohms|hertz)\b",
        search_text,
        re.IGNORECASE,
    )
    if how_many_unit:
        word = how_many_unit.group(1).lower()
        unit_mapping = {
            "meters": ("d", "m"), "metres": ("d", "m"),
            "kilometers": ("d", "km"), "kilometres": ("d", "km"),
            "seconds": ("t", "s"), "minutes": ("t", "min"), "hours": ("t", "h"),
            "grams": ("m", "g"), "kilograms": ("m", "kg"),
            "joules": ("E_energy", "J"), "watts": ("P", "W"),
            "volts": ("V", "V"), "amps": ("I", "A"), "amperes": ("I", "A"),
            "ohms": ("R", "Ω"), "hertz": ("f", "Hz"),
        }
        # If the question mentions 'above the ground' it's a height.
        if word in {"meters", "metres"} and "above the ground" in lowered:
            return "h", "m"
        return unit_mapping[word]

    # "U_L?" / "I?" / "Z?" — bare-symbol question (often Vietnamese textbook style).
    bare_symbol = re.search(
        r"(?:^|\.\s*|\?\s*)(U_?L|U_?C|U_?R|U|I|V|R|L|C|Z_?L|Z_?C|Z|B|P|f_?0?)\s*\?",
        problem_text,
    )
    if bare_symbol:
        sym = bare_symbol.group(1).replace("_", "").upper()
        bare_map = {
            "UL": ("U_L", "V"), "UC": ("U_C", "V"), "UR": ("U_R", "V"),
            "U": ("V", "V"), "V": ("V", "V"),
            "I": ("I", "A"), "R": ("R", "Ω"),
            "L": ("L_ind", "H"), "C": ("C_cap", "F"),
            "ZL": ("X_L", "Ω"), "ZC": ("X_C", "Ω"), "Z": ("Z", "Ω"),
            "B": ("B", "T"), "P": ("P", "W"),
            "F": ("f_res", "Hz"), "F0": ("f_res", "Hz"),
        }
        if sym in bare_map:
            return bare_map[sym]

    for phrase, symbol, unit in sorted(TARGETS, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", search_text, re.IGNORECASE):
            if phrase == "speed" and "wave" in lowered:
                return "v_wave", "m/s"
            return symbol, unit
    return None, None