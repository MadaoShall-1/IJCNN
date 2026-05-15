"""Question-type triage for Stage 0.

Determines whether a problem is a numeric calculation, a Yes/No boolean check,
or a symbolic/relational derivation. Used as a verifier gate so that non-numeric
problems are not penalized for missing a numeric step plan.

Design notes:
- Deterministic only (regex + keyword phrases). No ML, no LLM.
- Default is numeric_calc; we only flip to a non-numeric type when there is
  affirmative evidence in the question text. False positives are worse than
  false negatives here, because misclassifying a numeric problem as boolean
  would silently skip the numeric verifier.
- Triggers were chosen by inspecting actual dataset answers
  (Physics_Problems_Text_Only.csv). Re-evaluate when adding new corpora.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


QUESTION_TYPE_NUMERIC = "numeric_calc"
QUESTION_TYPE_BOOLEAN = "boolean_check"
QUESTION_TYPE_SYMBOLIC = "symbolic_derivation"
QUESTION_TYPE_UNKNOWN = "unknown"

# Boolean-check signals.
# These are phrases that appear in questions whose expected answer is Yes/No.
# Curated from CH-prefix and DDT354-style items in Physics_Problems_Text_Only.csv.
# Each entry: (regex, label).
BOOLEAN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "does the circuit experience resonance" / "does X happen" / "does resonance occur"
    # Restricted to action verbs to avoid matching "how does X compare" (which is symbolic).
    (re.compile(r"\bdoes\s+(?:(?:the|this|that|a)\s+\w+(?:\s+\w+){0,3}\s+)?(?:\w+\s+)?(experience|occur|happen|exist|undergo|reach|achieve|exhibit|show|have)\b", re.IGNORECASE), "does_X_action"),
    # "does (adjective) resonance/oscillation/equilibrium occur"  e.g. "does electrical resonance occur"
    (re.compile(r"\bdoes\s+\w+\s+(resonance|oscillation|equilibrium|saturation|breakdown)\s+\w*\s*(occur|happen)\b", re.IGNORECASE), "does_phenomenon_occur"),
    # "determine if X" / "determine whether"
    (re.compile(r"\bdetermine\s+(if|whether)\b", re.IGNORECASE), "determine_if"),
    # "Is X in resonance" — only at sentence start (case-sensitive) so we don't match
    # mid-sentence phrases like "the resonance frequency".
    (re.compile(r"(?:^|[.?!]\s+)Is\s+(the|this|it)\s+(circuit|system|capacitor|inductor)?\s*(in|at|under)?\s*resonan(t|ce)\b"), "is_in_resonance"),
    # "is X lost / conserved / present / valid / correct / same / true / equal"
    (re.compile(r"\bis\s+(the|this)\s+\w+(\s+\w+){0,5}\s+(lost|conserved|present|valid|correct|same|true|equal)\??", re.IGNORECASE), "is_the_X_predicate"),
    # Question starts with "Is " followed by a noun phrase and ending with ?
    # Captures CH-style "Is the circuit in resonance at f=70 Hz?"
    # Critical: must be case-SENSITIVE "Is" at start of a sentence/clause, not
    # "what is the..." mid-sentence. We anchor on sentence boundary + capital I.
    (re.compile(r"(?:^|[.?!]\s+)Is\s+(the|this|it|a|an)\s+[^.?!]{3,80}\?"), "is_X_question"),
    # "will resonance occur" / "will the X occur"
    (re.compile(r"\bwill\s+(resonance|the\s+\w+)\s+(occur|happen|change|increase|decrease|reach|exhibit)\b", re.IGNORECASE), "will_X_occur"),
    # "can X be Y"
    (re.compile(r"\bcan\s+(the|this|a)\s+\w+\s+be\b", re.IGNORECASE), "can_X_be"),
    # Explicit yes/no framing
    (re.compile(r"\b(yes\s*or\s*no|true\s*or\s*false)\b", re.IGNORECASE), "yes_or_no_explicit"),
]

# Symbolic-derivation signals.
# Questions that ask for a relationship, ratio, or expression rather than a number.
SYMBOLIC_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "expression for X"
    (re.compile(r"\bexpression\s+for\b", re.IGNORECASE), "expression_for"),
    # "in terms of"
    (re.compile(r"\bin\s+terms\s+of\b", re.IGNORECASE), "in_terms_of"),
    # "relationship between X and Y" / "relation between"
    (re.compile(r"\brelation(?:ship)?\s+between\b", re.IGNORECASE), "relationship_between"),
    # "compare X and Y" / "compare X with Y"
    (re.compile(r"\bcompare\s+\w+\s+(and|with|to)\b", re.IGNORECASE), "compare_X_and_Y"),
    # "derive X" / "derive the expression"
    (re.compile(r"\bderive\s+(the|an?)\b", re.IGNORECASE), "derive"),
    # "show that" / "prove that"  (mathematical derivation, no numeric answer)
    (re.compile(r"\b(show|prove)\s+that\b", re.IGNORECASE), "show_or_prove_that"),
    # "ratio between X and Y" / "what is the ratio of X to Y"
    (re.compile(r"\bratio\s+(between|of)\b", re.IGNORECASE), "ratio_of"),
    # "equation of motion" / "equation describing"
    (re.compile(r"\bequation\s+(of\s+motion|describing|that\s+governs)\b", re.IGNORECASE), "equation_of"),
    # "which of the following" — multiple choice that we treat as symbolic for now
    (re.compile(r"\bwhich\s+of\s+the\s+following\b", re.IGNORECASE), "which_of_the_following"),
    # "which statement is" / "which statement(s)"
    (re.compile(r"\bwhich\s+statement", re.IGNORECASE), "which_statement"),
    # ---------- New: conceptual / qualitative question patterns ----------
    # "How does X change/compare/depend ..." — asks for a relationship, not a value
    (re.compile(r"\bhow\s+(?:does|do|will|can)\s+(?:the\s+)?\w+(?:\s+\w+){0,4}\s+(?:change|compare|depend|behave|vary|differ|affect|relate|move)\b", re.IGNORECASE), "how_does_X_change"),
    # "how bright/dim/dark will the bulb be" — qualitative comparison
    (re.compile(r"\bhow\s+(?:bright|dim|dark|hot|cold|strong|weak|fast|slow)\b", re.IGNORECASE), "how_qualitative"),
    # "what happens to X if/when ..." — qualitative outcome
    (re.compile(r"\bwhat\s+happens\s+to\b", re.IGNORECASE), "what_happens_to"),
    # "where is the energy stored" / "where does X go" — asks for location/component
    (re.compile(r"\bwhere\s+(?:is|are|does|do|will)\b", re.IGNORECASE), "where_is"),
    # "when will X be zero" / "when does X reach maximum" — asks for condition, not value
    (re.compile(r"\bwhen\s+(?:will|does|is)\s+\w+(?:\s+\w+){0,4}\s+(?:be|become|reach|equal|occur|happen|change)\b", re.IGNORECASE), "when_will_X_be"),
    # "What form of energy" / "what kind of"
    (re.compile(r"\bwhat\s+(?:form|kind|type|sort|nature)\s+of\b", re.IGNORECASE), "what_form_of"),
    # "what is the shape of the graph" / "what does the graph look like"
    (re.compile(r"\b(?:shape\s+of\s+the\s+graph|graph\s+(?:representing|of))\b", re.IGNORECASE), "graph_shape"),
    # "what is the formula for" / "write the formula"
    (re.compile(r"\b(?:what\s+is\s+the\s+formula|write\s+(?:down\s+)?the\s+formula|state\s+the\s+formula)\s+for\b", re.IGNORECASE), "formula_for"),
    # "Which energy is at its maximum" / "which quantity"
    (re.compile(r"\bwhich\s+(?:energy|quantity|component|element|form|charge|capacitor|inductor|resistor)\b", re.IGNORECASE), "which_quantity"),
    # "If X happens, how/what" with no numeric measure — typically qualitative
    (re.compile(r"\bif\s+(?:the\s+)?\w+(?:\s+\w+){0,5}\s+(?:increases|decreases|doubles|halves|triples|change[sd]?)\s*,?\s+(?:how|what\s+happens|what\s+will\s+happen)\b", re.IGNORECASE), "if_X_changes"),
    # "describe the X" / "explain the X"
    (re.compile(r"\b(?:describe|explain|discuss|state)\s+(?:the|how|why)\b", re.IGNORECASE), "describe_explain"),
    # "What are the units/dimensions of X"
    (re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:units?|dimensions?)\s+of\b", re.IGNORECASE), "units_of"),
    # "Why is X" / "Why does X"
    (re.compile(r"\bwhy\s+(?:is|are|does|do|did|will|would)\b", re.IGNORECASE), "why_is"),
]

# Numeric-affirmation signals.
# When these appear, they strongly indicate a numeric question even if a weak
# boolean/symbolic signal also matches. Used to override low-confidence
# non-numeric classifications.
NUMERIC_OVERRIDE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcalculate\s+(the\s+)?\w+", re.IGNORECASE),
    re.compile(r"\bcompute\s+(the\s+)?\w+", re.IGNORECASE),
    # "Find the X" — accept an optional adjective before the target noun
    re.compile(r"\bfind\s+(?:the\s+)?(?:\w+\s+){0,2}(value|magnitude|distance|speed|velocity|force|energy|charge|current|voltage|power|frequency|period|wavelength|mass|time|acceleration|resistance|capacitance|inductance|temperature|pressure|momentum|impulse|work|amplitude|displacement|angle|tension|height|length|area|volume|density|intensity|flux)\b", re.IGNORECASE),
    re.compile(r"\bdetermine\s+(?:the\s+)?(?:\w+\s+){0,2}(value|magnitude|distance|speed|velocity|force|energy|charge|current|voltage|power|frequency|period|wavelength|mass|time|acceleration|resistance|capacitance|inductance|amplitude|displacement|angle|tension|intensity)\b", re.IGNORECASE),
    # "What is the X" — accept up to 2 modifiers between 'the' and the target noun
    re.compile(r"\bwhat\s+is\s+the\s+(?:\w+\s+){0,2}(value|magnitude|distance|speed|velocity|force|energy|charge|current|voltage|power|frequency|period|wavelength|mass|time|acceleration|resistance|capacitance|inductance|amplitude|displacement|angle|tension|intensity|flux|strength|number)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(many|much|long|far|fast)\b", re.IGNORECASE),
    # Explicit numeric-answer signposts: "round to N decimal place(s)", "in grams/meters/..."
    re.compile(r"\bround\s+(?:the\s+(?:answer|result|value))?\s*(?:to|off)\s+(?:one|two|three|four|\d+)\s+decimal", re.IGNORECASE),
    # "what percentage (%) of X is Y" — has a numeric answer.
    re.compile(r"\bwhat\s+percentage\s*(?:\(\s*%?\s*\))?\s+of\b", re.IGNORECASE),
    # "by what factor must X change" / "by what factor is X" — numeric factor
    re.compile(r"\bby\s+what\s+factor\b", re.IGNORECASE),
]


def classify_question_type(problem_text: str) -> Dict[str, object]:
    """Classify a problem's question type.

    Returns a dict with keys:
        question_type: one of the QUESTION_TYPE_* constants
        question_type_confidence: float in [0.0, 1.0]
        question_type_triggers: list of trigger labels that fired
    """
    if not isinstance(problem_text, str) or not problem_text.strip():
        return {
            "question_type": QUESTION_TYPE_UNKNOWN,
            "question_type_confidence": 0.0,
            "question_type_triggers": [],
        }

    text = problem_text.strip()

    # Collect raw signal hits
    boolean_hits = [label for pattern, label in BOOLEAN_PATTERNS if pattern.search(text)]
    symbolic_hits = [label for pattern, label in SYMBOLIC_PATTERNS if pattern.search(text)]
    numeric_overrides = [True for pattern in NUMERIC_OVERRIDE_PATTERNS if pattern.search(text)]

    has_numeric_override = bool(numeric_overrides)

    # Decision logic:
    # 1. Both numeric and non-numeric signals fire -> compound problem.
    #    Treat as numeric_calc because (a) the dataset shows numeric subparts dominate,
    #    and (b) a numeric verifier with high confidence is more useful than a
    #    skipped check.
    # 2. Only boolean signals fire -> boolean_check.
    # 3. Only symbolic signals fire -> symbolic_derivation.
    # 4. Both boolean and symbolic but no numeric override -> prefer the stronger
    #    (more distinctive) signal. Symbolic patterns are more specific by
    #    construction, so symbolic wins ties.
    # 5. Nothing fires -> numeric_calc default (most common in dataset).

    if has_numeric_override and (boolean_hits or symbolic_hits):
        return {
            "question_type": QUESTION_TYPE_NUMERIC,
            "question_type_confidence": 0.7,
            "question_type_triggers": ["numeric_override"] + boolean_hits + symbolic_hits,
        }

    if boolean_hits and not symbolic_hits:
        confidence = min(0.95, 0.6 + 0.1 * len(boolean_hits))
        return {
            "question_type": QUESTION_TYPE_BOOLEAN,
            "question_type_confidence": confidence,
            "question_type_triggers": boolean_hits,
        }

    if symbolic_hits and not boolean_hits:
        confidence = min(0.95, 0.6 + 0.1 * len(symbolic_hits))
        return {
            "question_type": QUESTION_TYPE_SYMBOLIC,
            "question_type_confidence": confidence,
            "question_type_triggers": symbolic_hits,
        }

    if boolean_hits and symbolic_hits:
        # Symbolic patterns are more specific; prefer them on ties.
        return {
            "question_type": QUESTION_TYPE_SYMBOLIC,
            "question_type_confidence": 0.55,
            "question_type_triggers": symbolic_hits + boolean_hits,
        }

    # Default
    return {
        "question_type": QUESTION_TYPE_NUMERIC,
        "question_type_confidence": 0.8,
        "question_type_triggers": [],
    }