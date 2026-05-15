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
    (re.compile(r"\bwhen\s+(?:will|does|is)\s+\w+(?:\s+\w+){0,8}\s+(?:be|become|reach|equal|occur|happen|change)\b", re.IGNORECASE), "when_will_X_be"),
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
    # ---------- Round 2: more conceptual / definitional question patterns ----------
    # "what is the SI unit of X" / "what is the unit of X"
    (re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:SI\s+)?units?\s+of\b", re.IGNORECASE), "unit_of_X"),
    # "what (factor / quantity / quantities) does X depend on"
    (re.compile(r"\bwhat\s+(?:factor|factors|quantity|quantities|parameter|parameters|variables?)\s+(?:does|do)\b", re.IGNORECASE), "what_does_X_depend_on"),
    # "what factor does X not depend on" — explicit negation
    (re.compile(r"\bwhat\s+(?:factor|factors|quantity|quantities)\s+(?:does|do)\s+\w+\s+(?:not\s+)?depend\b", re.IGNORECASE), "what_factor_not_depend"),
    # "what (form|type|kind|nature) of (energy|motion|...) is X"
    # already covered by what_form_of, but add "in what form is X stored"
    (re.compile(r"\bin\s+what\s+form\b", re.IGNORECASE), "in_what_form"),
    # "what does X depend (linearly) on"
    (re.compile(r"\bwhat\s+(?:does|do)\s+(?:the\s+)?\w+(?:\s+\w+){0,5}\s+depend\s+(?:linearly\s+)?on\b", re.IGNORECASE), "what_does_X_depend_on_2"),
    # "where is the X stored" / "where is the energy" already covered by where_is.
    # "when does X reach (a) maximum/minimum" — conceptual question about timing condition
    (re.compile(r"\bwhen\s+(?:does|do|will|is)\s+(?:the\s+)?\w+(?:\s+\w+){0,5}\s+(?:reach|attain|become|equal)\s+(?:its\s+)?(?:maximum|minimum|zero|max\b|min\b)\b", re.IGNORECASE), "when_X_max_min"),
    # "what is the X of/in an ideal/perfect Y" — usually definitional ("what is the
    # current in an ideal LC circuit when ...?", "what are the characteristics
    # of the magnetic field inside an ideal solenoid?")
    (re.compile(r"\bwhat\s+(?:are|is)\s+the\s+(?:characteristics?|properties|nature)\s+of\b", re.IGNORECASE), "characteristics_of_X"),
    # "How to calculate X" / "How do I calculate X" — asks for a method, not a value
    (re.compile(r"\bhow\s+(?:to|do\s+(?:i|we|you))\s+(?:calculate|compute|find|determine|measure|solve)\b", re.IGNORECASE), "how_to_calculate"),
    # "is there a formula" / "what is the formula" — variant of formula_for
    (re.compile(r"\bis\s+there\s+(?:an?\s+)?formula\b", re.IGNORECASE), "is_there_a_formula"),
    # Multiple-choice option markers — "A.", "B.", "C.", "D." on separate lines
    # or after a colon. If at least 2 are present, the question is multiple choice.
    (re.compile(r"(?:^|\n)\s*[A-D]\s*[\.\)]\s+\w", re.MULTILINE), "multiple_choice_options"),
    # "Find the expression/formula for V"
    (re.compile(r"\bfind\s+(?:the|an?)\s+(?:expression|formula|equation)\s+for\b", re.IGNORECASE), "find_expression"),
    # "specify the requirements" / "state the formula" — instructional, no numeric answer
    (re.compile(r"\b(?:state|specify|provide|give|list|write)\s+(?:the|down\s+the)\s+(?:formula|requirements?|conditions?|expression|relationship)\b", re.IGNORECASE), "state_the_formula"),
    # "what kind of electric field" — concept variant
    (re.compile(r"\bwhat\s+kind\s+of\s+(?:electric|magnetic|gravitational)\s+field\b", re.IGNORECASE), "what_kind_of_field"),
    # "what does this indicate about" — interpretive
    (re.compile(r"\bwhat\s+does\s+this\s+indicate\s+about\b", re.IGNORECASE), "what_does_this_indicate"),
    # "may I ask" / "please state" / "could you tell me" — instructional opener,
    # usually attached to a concept question.
    (re.compile(r"\b(?:may\s+i\s+ask|please\s+state|please\s+tell|could\s+you\s+tell|please\s+specify)\b", re.IGNORECASE), "instructional_opener"),
    # ---------- Round 3: LC-oscillation conceptual phrasings ----------
    # "When does the X reach maximum/minimum/zero" — qualitative timing
    (re.compile(r"\bwhen\s+does\s+(?:the\s+)?\w+(?:\s+\w+){0,8}\s+(?:reach|attain|become|happen|occur|peak)\b", re.IGNORECASE), "when_does_X_reach"),
    # "When the X is N/M of the total" — fractional energy partition question.
    # Tight: numeric fraction + 'of (the) total' is a clear conceptual setup.
    (re.compile(r"\bwhen\s+(?:the\s+)?\w+(?:\s+\w+){0,6}\s+(?:is|equals)\s+(?:half|a\s+third|a\s+quarter|\d+/\d+|⅓|¼|½|¾|⅔)\s+of\s+(?:the\s+)?total\b", re.IGNORECASE), "when_X_is_fraction_of_total"),
    # "Where is the X entirely/fully/completely stored" — definitional location
    (re.compile(r"\bwhere\s+(?:is|are)\s+(?:the\s+)?\w+(?:\s+\w+){0,4}\s+(?:entirely|fully|completely|exclusively)\s*stored\b", re.IGNORECASE), "where_is_entirely_stored"),
    # "What happens when X" — qualitative outcome
    (re.compile(r"\bwhat\s+happens\s+(?:in|when)\b", re.IGNORECASE), "what_happens_when"),
    # "What fraction is the X" — fraction question with no numeric inputs
    (re.compile(r"\bwhat\s+fraction\s+(?:is|of|are)\b", re.IGNORECASE), "what_fraction_is"),
    # "What appears in the closed circuit when..." — qualitative phenomenon
    (re.compile(r"\bwhat\s+(?:appears|develops|forms|emerges|arises)\s+in\b", re.IGNORECASE), "what_appears_in"),
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

# Strong symbolic-override patterns: these almost always indicate a conceptual
# question, EVEN when a weak numeric pattern fires. Carefully curated; phrases
# that have unambiguous symbolic intent.
STRONG_SYMBOLIC_OVERRIDE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "How to calculate X" / "How do I find X" — methodology question
    (re.compile(r"\bhow\s+(?:to|do\s+(?:i|we|you))\s+(?:calculate|compute|find|determine|measure|solve)\b", re.IGNORECASE), "strong_how_to"),
    # "what is the unit of X" / "SI unit of X" — definitional
    (re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:SI\s+)?units?\s+of\b", re.IGNORECASE), "strong_unit_of"),
    # "what is the formula for X" / "state the formula"
    (re.compile(r"\b(?:what\s+is\s+the\s+formula|state\s+the\s+formula|write\s+(?:down\s+)?the\s+formula)\s+for\b", re.IGNORECASE), "strong_formula_for"),
    # "in what form" — definitional
    (re.compile(r"\bin\s+what\s+form\b", re.IGNORECASE), "strong_in_what_form"),
    # "where is X (stored|located|concentrated)"
    (re.compile(r"\bwhere\s+(?:is|are)\s+(?:the\s+)?\w+(?:\s+\w+){0,3}\s+(?:stored|located|concentrated|present|found|placed)\b", re.IGNORECASE), "strong_where_is_stored"),
    # "what quantity does X depend on"
    (re.compile(r"\bwhat\s+(?:factor|factors|quantity|quantities)\s+(?:does|do)\b", re.IGNORECASE), "strong_what_depend_on"),
    # Note: multiple_choice intentionally NOT here — many multi-choice problems
    # have numeric answers (e.g., "what time will they meet? A. 8 AM B. 9 AM...").
    # Multi-choice remains in regular SYMBOLIC_PATTERNS so it competes with
    # weak numeric override on equal footing.
]

# answer, even when a weak symbolic pattern also fires. These OVERRIDE any
# competing symbolic signal.
# Weak/lexical overrides above (e.g. "what is the X") trigger numeric only
# when no symbolic pattern is strong enough — symbolic patterns are evaluated
# alongside and can win.
STRONG_NUMERIC_OVERRIDE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bround\s+(?:the\s+(?:answer|result|value))?\s*(?:to|off)\s+(?:one|two|three|four|\d+)\s+decimal", re.IGNORECASE),
    re.compile(r"\bwhat\s+percentage\s*(?:\(\s*%?\s*\))?\s+of\b", re.IGNORECASE),
    re.compile(r"\bby\s+what\s+factor\b", re.IGNORECASE),
    # Numeric-tagged "calculate"/"compute" with explicit unit follow-up:
    # "Calculate the force in newtons", "Compute the energy in joules"
    re.compile(r"\b(?:calculate|compute)\b.*\b(?:in\s+(?:newtons?|joules?|coulombs?|amperes?|volts?|ohms?|watts?|hertz|seconds?|meters?|kilograms?|tesla|farads?))\b", re.IGNORECASE),
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
    strong_symbolic_hits = [label for pattern, label in STRONG_SYMBOLIC_OVERRIDE_PATTERNS if pattern.search(text)]
    weak_numeric = [True for pattern in NUMERIC_OVERRIDE_PATTERNS if pattern.search(text)]
    strong_numeric = [True for pattern in STRONG_NUMERIC_OVERRIDE_PATTERNS if pattern.search(text)]

    has_weak_numeric = bool(weak_numeric)
    has_strong_numeric = bool(strong_numeric)
    has_strong_symbolic = bool(strong_symbolic_hits)

    # Decision logic, in priority order:
    # 1. Strong numeric override wins unconditionally.
    # 2. Strong symbolic override wins over weak numeric (e.g. "how to
    #    calculate" wins over the bare "calculate" that NUMERIC_OVERRIDE
    #    matches inside it).
    # 3. No symbolic/boolean hits: default numeric_calc.
    # 4. Boolean + symbolic both present: prefer symbolic.
    # 5. Weak numeric vs single non-numeric hit: weak numeric wins.
    # 6. Weak numeric vs ≥2 non-numeric hits: non-numeric wins.

    if has_strong_numeric:
        return {
            "question_type": QUESTION_TYPE_NUMERIC,
            "question_type_confidence": 0.85,
            "question_type_triggers": ["strong_numeric_override"] + boolean_hits + symbolic_hits + strong_symbolic_hits,
        }

    if has_strong_symbolic and not has_strong_numeric:
        return {
            "question_type": QUESTION_TYPE_SYMBOLIC,
            "question_type_confidence": 0.85,
            "question_type_triggers": strong_symbolic_hits + symbolic_hits + boolean_hits,
        }

    if boolean_hits and symbolic_hits:
        return {
            "question_type": QUESTION_TYPE_SYMBOLIC,
            "question_type_confidence": 0.7,
            "question_type_triggers": symbolic_hits + boolean_hits,
        }

    if has_weak_numeric and (boolean_hits or symbolic_hits):
        non_numeric_hits = symbolic_hits + boolean_hits
        if len(non_numeric_hits) >= 2:
            chosen = QUESTION_TYPE_SYMBOLIC if symbolic_hits else QUESTION_TYPE_BOOLEAN
            return {
                "question_type": chosen,
                "question_type_confidence": 0.65,
                "question_type_triggers": non_numeric_hits,
            }
        return {
            "question_type": QUESTION_TYPE_NUMERIC,
            "question_type_confidence": 0.7,
            "question_type_triggers": ["numeric_override"] + non_numeric_hits,
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

    # Default
    return {
        "question_type": QUESTION_TYPE_NUMERIC,
        "question_type_confidence": 0.8,
        "question_type_triggers": [],
    }