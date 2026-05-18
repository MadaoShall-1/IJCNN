"""DSPy Signatures and Modules for the Type 1 reasoning pipeline.

Phase 0 (MVP) implementation: all question formats use LLM-only reasoning.
Z3 and Prover9 formal solver paths are declared in the routing layer but are
not yet active; questions assigned SolverRoute.Z3 / PROVER9 fall through to
the LLM path until Phase 1 integrates the formal solvers.

Configure the DSPy language model before instantiating any module:

    import dspy
    lm = dspy.LM(
        model="openai/qwen3-8b",       # model name as served by vLLM
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",               # vLLM does not require a real key
    )
    dspy.configure(lm=lm)

See design §9.1 for vLLM serving setup and §3.2 for SolverConfig.
"""

from __future__ import annotations

import re
from typing import List, Optional

import dspy

from .schemas import (
    QuestionFormat,
    SolverRoute,
    Type1ParseObject,
    Type1Question,
    Type1Response,
)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

class MCQReasoning(dspy.Signature):
    """Answer a multiple-choice question by reasoning step-by-step from premises.

    The answer must be exactly one letter: A, B, C, or D.
    """

    premises: list[str] = dspy.InputField(
        desc="Natural language premises (facts and rules to reason from)"
    )
    question: str = dspy.InputField(
        desc="The multiple-choice question text, including all labeled options"
    )
    answer: str = dspy.OutputField(
        desc="The single letter of the correct option (A, B, C, or D)"
    )
    explanation: str = dspy.OutputField(
        desc="Clear natural language justification that references specific premises"
    )


class YesNoReasoning(dspy.Signature):
    """Determine whether a conclusion follows from the given premises.

    The answer must be exactly one word: Yes, No, or Uncertain.
    - Yes: the conclusion must hold given all premises.
    - No: the conclusion is contradicted by the premises.
    - Uncertain: the premises neither prove nor disprove the conclusion.
    """

    premises: list[str] = dspy.InputField(
        desc="Natural language premises (facts and rules to reason from)"
    )
    question: str = dspy.InputField(
        desc="The question asking whether a conclusion follows from the premises"
    )
    answer: str = dspy.OutputField(
        desc="Exactly one of: Yes, No, or Uncertain"
    )
    explanation: str = dspy.OutputField(
        desc="Clear natural language justification that references specific premises"
    )


class OpenEndedReasoning(dspy.Signature):
    """Answer an open-ended question by reasoning from the given premises."""

    premises: list[str] = dspy.InputField(
        desc="Natural language premises (facts and rules to reason from)"
    )
    question: str = dspy.InputField(
        desc="The open-ended question to answer"
    )
    answer: str = dspy.OutputField(
        desc="A concise, direct answer derived from the premises"
    )
    explanation: str = dspy.OutputField(
        desc="Clear natural language justification that references specific premises"
    )


class PremiseFOLFormalization(dspy.Signature):
    """Convert natural language premises into first-order logic (FOL) expressions.

    Each input premise should produce exactly one FOL expression using standard
    notation: ForAll, Exists, ∧ (and), ∨ (or), → (implies), ¬ (not).
    """

    premises_nl: list[str] = dspy.InputField(
        desc="Natural language premises to formalize, one per list entry"
    )
    premises_fol: list[str] = dspy.OutputField(
        desc="FOL representations in the same order as the input premises"
    )


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------

class MCQReasoner(dspy.Module):
    """Reason over premises to answer a multiple-choice question."""

    def __init__(self) -> None:
        self.cot = dspy.ChainOfThought(MCQReasoning)

    def forward(self, premises: List[str], question: str) -> dspy.Prediction:
        return self.cot(premises=premises, question=question)


class YesNoReasoner(dspy.Module):
    """Reason over premises to answer a Yes/No/Uncertain question."""

    def __init__(self) -> None:
        self.cot = dspy.ChainOfThought(YesNoReasoning)

    def forward(self, premises: List[str], question: str) -> dspy.Prediction:
        return self.cot(premises=premises, question=question)


class OpenEndedReasoner(dspy.Module):
    """Reason over premises to answer an open-ended question."""

    def __init__(self) -> None:
        self.cot = dspy.ChainOfThought(OpenEndedReasoning)

    def forward(self, premises: List[str], question: str) -> dspy.Prediction:
        return self.cot(premises=premises, question=question)


class FOLFormalizer(dspy.Module):
    """Convert natural language premises to FOL expressions."""

    def __init__(self) -> None:
        self.predict = dspy.Predict(PremiseFOLFormalization)

    def forward(self, premises_nl: List[str]) -> dspy.Prediction:
        return self.predict(premises_nl=premises_nl)


# ---------------------------------------------------------------------------
# Main Type 1 Solver
# ---------------------------------------------------------------------------

class Type1Solver(dspy.Module):
    """Phase 0 Type 1 solver.

    Routes each question to the appropriate DSPy reasoner, generates
    FOL formalizations of the premises (unless already provided), and
    assembles Type1Response objects matching the competition API schema.

    Z3 and Prover9 routes are not yet active (Phase 1).  Questions with
    SolverRoute.Z3 or SolverRoute.PROVER9 fall through to the LLM path.

    Args:
        generate_fol: If True, run FOLFormalizer when premises_fol is absent.
            Adds one LLM call per request.  Set False to reduce latency.
    """

    def __init__(self, generate_fol: bool = True) -> None:
        self.mcq_reasoner = MCQReasoner()
        self.yes_no_reasoner = YesNoReasoner()
        self.open_ended_reasoner = OpenEndedReasoner()
        self.fol_formalizer = FOLFormalizer() if generate_fol else None
        self.generate_fol = generate_fol

    def forward(self, parse_obj: Type1ParseObject) -> List[Type1Response]:
        """Solve all questions in a Type1ParseObject.

        Returns one Type1Response per question in the same order as
        ``parse_obj.questions``.

        Typical usage (evaluation — single question per call):

            solver = Type1Solver()
            [response] = solver(parse_obj)   # parse_obj has one question
            api_payload = response.to_dict()
        """
        fol_premises = self._get_fol_premises(parse_obj)
        return [
            self._solve_one(
                premises=parse_obj.premises_nl,
                question=q,
                fol_premises=fol_premises,
            )
            for q in parse_obj.questions
        ]

    def solve_one(
        self,
        parse_obj: Type1ParseObject,
        question_idx: int = 0,
    ) -> Type1Response:
        """Convenience method: solve a single question by index."""
        fol_premises = self._get_fol_premises(parse_obj)
        return self._solve_one(
            premises=parse_obj.premises_nl,
            question=parse_obj.questions[question_idx],
            fol_premises=fol_premises,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fol_premises(self, parse_obj: Type1ParseObject) -> List[str]:
        if parse_obj.premises_fol:
            return list(parse_obj.premises_fol)
        if self.generate_fol and parse_obj.premises_nl:
            result = self.fol_formalizer(premises_nl=parse_obj.premises_nl)
            fol = getattr(result, "premises_fol", None)
            if isinstance(fol, list):
                return fol
        return []

    def _solve_one(
        self,
        premises: List[str],
        question: Type1Question,
        fol_premises: List[str],
    ) -> Type1Response:
        pred = self._call_reasoner(premises, question)
        cot_steps = _extract_cot_steps(pred)
        confidence = _compute_confidence(question, pred)

        return Type1Response(
            answer=_clean_answer(pred.answer, question.format),
            explanation=pred.explanation,
            fol=fol_premises,
            cot=cot_steps,
            premises=list(premises),
            confidence=confidence,
        )

    def _call_reasoner(
        self,
        premises: List[str],
        question: Type1Question,
    ) -> dspy.Prediction:
        # Phase 0: Z3/Prover9 routes fall through to LLM.
        # Phase 1 will intercept these routes before reaching this method.
        if question.format == QuestionFormat.MCQ:
            return self.mcq_reasoner(premises=premises, question=question.text)
        elif question.format == QuestionFormat.YES_NO_UNCERTAIN:
            return self.yes_no_reasoner(premises=premises, question=question.text)
        else:
            return self.open_ended_reasoner(premises=premises, question=question.text)


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

_YES_NO_VALID = {"yes", "no", "uncertain"}
_MCQ_LETTER_RE = re.compile(r"\b([A-D])\b")


def _clean_answer(raw: str, fmt: QuestionFormat) -> str:
    """Normalise the raw LLM answer string to the expected competition format."""
    raw = raw.strip()
    if fmt == QuestionFormat.MCQ:
        # Extract just the letter if the LLM returned "A. text" or "The answer is B".
        m = _MCQ_LETTER_RE.search(raw)
        return m.group(1).upper() if m else raw.split()[0].upper()
    if fmt == QuestionFormat.YES_NO_UNCERTAIN:
        lower = raw.split()[0].lower().rstrip(".,;")
        return lower.capitalize() if lower in _YES_NO_VALID else raw
    return raw


def _extract_cot_steps(pred: dspy.Prediction) -> List[str]:
    """Extract chain-of-thought steps from a DSPy prediction.

    ChainOfThought adds a ``reasoning`` field before declared outputs.
    Split on numbered lines or double-newlines to get a step list.
    """
    reasoning: Optional[str] = getattr(pred, "reasoning", None)
    if not reasoning:
        return []
    # Split on "Step N:" or numbered-list patterns first.
    steps = re.split(r"(?:^|\n)\s*(?:Step\s+\d+[:.)]|\d+[.)]\s+)", reasoning)
    steps = [s.strip() for s in steps if s.strip()]
    # Fall back to paragraph splitting if no step markers found.
    if len(steps) <= 1:
        steps = [p.strip() for p in re.split(r"\n{2,}", reasoning) if p.strip()]
    return steps


def _compute_confidence(
    question: Type1Question,
    pred: dspy.Prediction,
) -> Optional[float]:
    """Return a confidence score carried on the prediction, or None.

    Phase 0: always None (no verifier adapter yet).
    Phase 1: the verifier module populates this after its pass.
    The design §3.1 specifies confidence = 1.0 for formal-solver results
    and verifier-adjusted (default 0.9 agree / 0.4 disagree) for LLM paths.
    """
    return None


# ---------------------------------------------------------------------------
# Type 1 Verifier — Phase 1 (design §3.1 Step 4)
# ---------------------------------------------------------------------------

class Type1Verification(dspy.Signature):
    """Independently verify whether a proposed answer follows from the premises.

    You are a verifier, not an answer generator.  Your task is to critically
    evaluate whether the proposed answer is correct given the premises,
    WITHOUT generating a new answer of your own.

    Return AGREE if the proposed answer is well-supported by the premises.
    Return DISAGREE if the proposed answer is incorrect or unsupported.
    """

    premises: list[str] = dspy.InputField(
        desc="Natural language premises (the facts/rules to reason from)"
    )
    question: str = dspy.InputField(
        desc="The original question"
    )
    proposed_answer: str = dspy.InputField(
        desc="The answer produced by the primary reasoner"
    )
    proposed_explanation: str = dspy.InputField(
        desc="The explanation produced by the primary reasoner"
    )
    verdict: str = dspy.OutputField(
        desc="Exactly one of: AGREE or DISAGREE"
    )
    verifier_reasoning: str = dspy.OutputField(
        desc="Brief justification of your verdict, citing specific premises"
    )


class Type1Verifier(dspy.Module):
    """Second-pass independent review of a Type 1 answer (design §3.1 Step 4).

    Used for MCQ and open-ended questions when ``SolverConfig.type1_verify``
    is True and Tier 1 timeout has not been exceeded.

    On DISAGREE: confidence is set to 0.4 and the disagreement is noted in
    the explanation.  The original answer is NOT changed — the verifier
    flags uncertainty but does not override, since it may also be wrong.
    On AGREE: confidence is set to 0.9.
    """

    # Confidence values match design §3.1 verifier-adjusted specification
    CONFIDENCE_AGREE = 0.9
    CONFIDENCE_DISAGREE = 0.4

    def __init__(self) -> None:
        self.verify = dspy.ChainOfThought(Type1Verification)

    def forward(
        self,
        premises: List[str],
        question: str,
        proposed_answer: str,
        proposed_explanation: str,
    ) -> dspy.Prediction:
        """Run the verifier pass and return verdict + adjusted confidence.

        Returns a Prediction with:
          verdict    – "AGREE" or "DISAGREE"
          confidence – 0.9 (agree) or 0.4 (disagree)
          note       – appended to explanation if disagree
        """
        pred = self.verify(
            premises=premises,
            question=question,
            proposed_answer=proposed_answer,
            proposed_explanation=proposed_explanation,
        )

        verdict = _normalise_verdict(pred.verdict)
        confidence = (
            self.CONFIDENCE_AGREE
            if verdict == "AGREE"
            else self.CONFIDENCE_DISAGREE
        )
        note = (
            ""
            if verdict == "AGREE"
            else (
                f" [Verifier note: independent review found this answer uncertain. "
                f"Verifier reasoning: {pred.verifier_reasoning}]"
            )
        )
        return dspy.Prediction(
            verdict=verdict,
            confidence=confidence,
            note=note,
        )


def _normalise_verdict(raw: str) -> str:
    """Normalise verifier output to 'AGREE' or 'DISAGREE'."""
    upper = raw.strip().upper()
    if "AGREE" in upper and "DISAGREE" not in upper:
        return "AGREE"
    if "DISAGREE" in upper:
        return "DISAGREE"
    # Unknown output — treat as AGREE to avoid spurious confidence penalties
    return "AGREE"
