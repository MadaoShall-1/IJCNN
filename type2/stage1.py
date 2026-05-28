"""Type 2 Stage 1: Formula & Premise Retrieval (design §5 Stage 1).

Three-tier variable canonicalization cascade (design §5 Stage 1 step 2):
  Tier 1 — Regex lookup (CANONICAL_MAP): deterministic, no I/O.
  Tier 2 — Embedding similarity (sentence-transformers): guarded import;
            skipped gracefully when sentence-transformers is not installed.
  Tier 3 — LLM: Phase 2; not implemented here.

Retrieval pipeline per formula_application step:
  1. Topic filter  — narrow library by domain/subdomain.
  2. Canonicalize  — map problem variable names to canonical quantity names.
  3. Quantity match — score by overlap with canonical_quantity_names.
  4. Embedding fallback — when < 2 candidates score above threshold.
  5. Path construction — return beam_n ranked FormulaSet objects.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from parser.schemas import ProblemParseObject

from .schemas import FormulaEntry, FormulaSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding availability guard
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    import numpy as _np
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _np = None                   # type: ignore[assignment]
    _EMBEDDINGS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIBRARY_PATH = Path(__file__).parent / "formula_library.json"
_QUANTITY_MATCH_THRESHOLD = 0.3   # minimum overlap score to include a candidate
_EMBEDDING_THRESHOLD = 0.6        # cosine similarity threshold for Tier 2
_EMBEDDING_MODEL = "BAAI/bge-small-en"

# ---------------------------------------------------------------------------
# Tier 1 — Regex canonicalization table (design §5 Stage 1 step 2)
# ---------------------------------------------------------------------------

# Each entry is (regex_pattern, canonical_quantity_name).
# Patterns are tested in order; first match wins.
# Lower entries are more specific short-symbol overrides.

CANONICAL_MAP: List[Tuple[str, str]] = [
    # ── Descriptive multi-word patterns (matched before short symbols) ────
    (r".*(velocity|speed).*",                    "velocity"),
    (r".*(acceleration).*",                      "acceleration"),
    (r".*(mass).*",                              "mass"),
    (r".*(weight).*",                            "force"),       # weight = mg, unit N
    (r".*(height|altitude).*",                   "displacement"),
    (r".*(distance|displacement|position|separation).*", "displacement"),
    (r".*(force|thrust).*",                      "force"),
    (r".*(temperature|thermal).*",               "temperature"),
    (r".*(pressure).*",                          "pressure"),
    (r".*(energy|work).*",                       "energy"),
    (r".*(current|ampere).*",                    "electric_current"),
    (r".*(voltage|potential|emf|electromotive).*","electric_potential"),
    (r".*(resistance|impedance).*",              "resistance"),
    (r".*(frequency|wavelength).*",              "frequency"),
    (r".*(angle|theta).*",                       "angle"),
    (r".*(time|duration).*",                     "time"),
    (r".*(power).*",                             "power"),
    (r".*(charge|coulomb).*",                    "electric_charge"),
    (r".*(capacitance|capacitor).*",             "capacitance"),
    (r".*(electric.*field|field.*strength).*",   "electric_field"),
    (r".*(flux).*",                              "electric_flux"),
    (r".*(area).*",                              "area"),
    (r".*(permittivity).*",                      "permittivity"),
    (r".*(inductance|inductor).*",               "inductance"),
    (r".*(time.?constant|tau).*",                "time_constant"),
    (r".*(momentum).*",                          "momentum"),
    # ── Indexed short symbols (R1, R2, C1, C2, V1, I1 …) ─────────────────
    (r"^[Rr]\d+$",  "resistance"),
    (r"^[Cc]\d+$",  "capacitance"),
    (r"^[Vv]\d+$",  "electric_potential"),
    (r"^[Ii]\d+$",  "electric_current"),
    # ── Compound short symbols ────────────────────────────────────────────
    (r"^[Rr]_[a-zA-Z0-9]+$", "resistance"),
    (r"^[Cc]_[a-zA-Z0-9]+$", "capacitance"),
    (r"^[Vv]_[a-zA-Z0-9]+$", "electric_potential"),
    (r"^[Ii]_[a-zA-Z0-9]+$", "electric_current"),
    # ── Single-character conventional physics symbols ─────────────────────
    (r"^[Vv]$",   "electric_potential"),
    (r"^[Ii]$",   "electric_current"),
    (r"^[Rr]$",   "resistance"),
    (r"^[Pp]$",   "power"),
    (r"^[Qq]$",   "electric_charge"),
    (r"^[Cc]$",   "capacitance"),
    (r"^[Mm]$",   "mass"),
    (r"^[Ff]$",   "force"),
    (r"^[Ee]$",   "electric_field"),
    (r"^[Uu]$",   "energy"),
    (r"^[Ww]$",   "energy"),
    (r"^[Tt]$",   "time"),
    (r"^[Aa]$",   "area"),
    (r"^tau$",    "time_constant"),
    (r"^[Kk]$",   "energy"),               # KE abbreviation
    (r"^[Hh]$",   "displacement"),         # height
    (r"^[Dd]$",   "displacement"),         # distance
]

# Pre-compiled patterns for O(1) per-call lookup
_COMPILED_CANONICAL_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in CANONICAL_MAP
]


def _tokens(text: object) -> Set[str]:
    """Tokenize formula/template text for lightweight lexical scoring."""
    tokens: Set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", str(text or "")):
        token = raw.lower().strip("_")
        if not token:
            continue
        tokens.add(token)

        # Expand parser-style symbols so F_13, C_cap, and k_e can match
        # library symbols such as F, C, and k_e without requiring exact names.
        for part in re.split(r"_+", token):
            if part:
                tokens.add(part)
                alpha_prefix = re.match(r"[a-z]+", part)
                if alpha_prefix:
                    tokens.add(alpha_prefix.group(0))

        alpha_prefix = re.match(r"[a-z]+", token)
        if alpha_prefix:
            tokens.add(alpha_prefix.group(0))

    return tokens


def _overlap_score(query_tokens: Set[str], candidate_tokens: Set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _normalize_name(name: str) -> str:
    """Lowercase, strip whitespace, collapse separators to underscore."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _formula_similarity(a: str, b: str) -> float:
    """Similarity between two formula strings (0..1).

    Strips whitespace and compares symbolic tokens.  Returns 1.0 on exact
    match, partial credit for high token overlap.
    """
    if not a or not b:
        return 0.0
    norm_a = re.sub(r"\s+", "", a.lower())
    norm_b = re.sub(r"\s+", "", b.lower())
    if norm_a == norm_b:
        return 1.0
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    return jaccard


def canonicalize_variable(name: str) -> Optional[str]:
    """Map a problem variable name to a canonical quantity name (Tier 1).

    Returns the canonical name string, or ``None`` if no regex matches.
    """
    for pattern, canonical in _COMPILED_CANONICAL_MAP:
        if pattern.match(name):
            return canonical
    return None


def detect_collisions(
    var_to_canonical: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Find pairs of distinct variable names that map to the same canonical.

    Returns a list of ``(name_a, name_b, canonical)`` triples.
    The caller decides how to resolve collisions (log warning, LLM call, etc.).
    """
    canonical_to_first: Dict[str, str] = {}
    collisions: List[Tuple[str, str, str]] = []
    for name, canonical in var_to_canonical.items():
        if canonical in canonical_to_first:
            collisions.append((canonical_to_first[canonical], name, canonical))
        else:
            canonical_to_first[canonical] = name
    return collisions


# ---------------------------------------------------------------------------
# Formula library loader
# ---------------------------------------------------------------------------

def load_library(path: Optional[Path] = None) -> List[FormulaEntry]:
    """Load and parse the formula library JSON file."""
    target = path or _LIBRARY_PATH
    with open(target, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [FormulaEntry.from_dict(entry) for entry in raw]


# ---------------------------------------------------------------------------
# FormulaRetriever
# ---------------------------------------------------------------------------

class FormulaRetriever:
    """Stage 1 retrieval: given a ProblemParseObject, return ranked formula sets.

    Parameters
    ----------
    library_path:
        Path to the formula library JSON.  Defaults to the bundled
        ``formula_library.json`` in this package directory.
    embedding_model:
        Sentence-transformers model name for Tier 2 canonicalization.
        Only used when sentence-transformers is installed.  Defaults to
        ``BAAI/bge-small-en``.
    """

    def __init__(
        self,
        library_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        path = Path(library_path) if library_path else _LIBRARY_PATH
        self._library = load_library(path)
        self._embed_model = None
        self._library_embeddings = None

        if _EMBEDDINGS_AVAILABLE:
            model_name = embedding_model or _EMBEDDING_MODEL
            try:
                self._embed_model = _SentenceTransformer(model_name)
                # Pre-compute embeddings for all formula text fields
                texts = [e.text for e in self._library]
                self._library_embeddings = self._embed_model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                )
                logger.info("Loaded embedding model '%s' for Stage 1 Tier 2.", model_name)
            except Exception as exc:
                logger.warning(
                    "Could not load embedding model '%s': %s. Tier 2 disabled.", model_name, exc
                )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        parse_obj: ProblemParseObject,
        beam_n: int = 3,
    ) -> List[FormulaSet]:
        """Return up to *beam_n* ranked formula sets for the given parse object.

        Each FormulaSet maps step_id → FormulaEntry for every
        ``formula_application`` step in the plan.  A step with no matching
        formula gets ``None`` (Stage 2 falls back to LLM-only for that step).
        """
        formula_steps = [
            s for s in parse_obj.step_plan
            if isinstance(s, dict) and s.get("type") == "formula_application"
        ]

        if not formula_steps:
            logger.info("No formula_application steps in plan; returning empty formula set.")
            return [FormulaSet(formulas={}, retrieval_confidence=0.0, path_index=0)]

        # ── Topic filter (applied once for the whole problem) ─────────────
        # Expand sub_domains with step-level template_names so the topic
        # filter doesn't exclude entries the parser explicitly referenced.
        step_templates = {
            _normalize_name(s.get("template_name", ""))
            for s in formula_steps
            if s.get("template_name")
        }
        expanded_subs = list(parse_obj.sub_domains) + [
            t for t in step_templates if t not in {_normalize_name(s) for s in parse_obj.sub_domains}
        ]
        topic_candidates = self._topic_filter(parse_obj.domains, expanded_subs)

        # ── Collect all variable names across formula_application steps ───
        all_vars: Set[str] = set()
        for step in formula_steps:
            all_vars.update(step.get("input_var", {}).keys())
            all_vars.update(step.get("output_var", {}).keys())

        # ── Tier 1: canonicalize all variable names ───────────────────────
        var_to_canonical: Dict[str, str] = {}
        for name in all_vars:
            canonical = canonicalize_variable(name)
            if canonical:
                var_to_canonical[name] = canonical
            else:
                logger.debug(
                    "No Tier-1 canonical for variable '%s'; embedding/LLM tier needed.", name
                )

        # ── Collision detection ───────────────────────────────────────────
        for name_a, name_b, canonical in detect_collisions(var_to_canonical):
            logger.warning(
                "Variable collision: '%s' and '%s' both canonicalize to '%s'. "
                "LLM disambiguation (Phase 2) skipped; keeping both mappings.",
                name_a, name_b, canonical,
            )

        # ── Per-step candidate scoring ────────────────────────────────────
        step_candidates: Dict[str, List[Tuple[FormulaEntry, float]]] = {}

        for step in formula_steps:
            step_id = step["step_id"]
            step_vars: Set[str] = set()
            step_vars.update(step.get("input_var", {}).keys())
            step_vars.update(step.get("output_var", {}).keys())

            canonical_vars = {
                var_to_canonical[v] for v in step_vars if v in var_to_canonical
            }

            scored = self._score_candidates(
                topic_candidates,
                canonical_vars,
                step_vars,
                formula_name=step.get("formula_name", ""),
                template_name=step.get("template_name", ""),
                step_goal=step.get("goal", ""),
            )
            good = [(e, s) for e, s in scored if s >= _QUANTITY_MATCH_THRESHOLD]

            # Tier 2: embedding fallback when few candidates pass the threshold
            if len(good) < 2 and self._embed_model is not None:
                embed_results = self._embedding_fallback(
                    problem_text=parse_obj.problem_text,
                    step_goal=step.get("goal", ""),
                )
                existing_ids = {e.id for e, _ in good}
                for entry, score in embed_results:
                    if entry.id not in existing_ids:
                        good.append((entry, score))

            if not good:
                logger.warning(
                    "No formula found for step '%s' (goal: '%s'). "
                    "Stage 2 will attempt LLM-only for this step.",
                    step_id, step.get("goal", ""),
                )

            step_candidates[step_id] = sorted(good, key=lambda x: -x[1])

        step_ids = [s["step_id"] for s in formula_steps]
        return self._build_formula_sets(step_candidates, step_ids, beam_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topic_filter(
        self,
        domains: List[str],
        sub_domains: List[str],
    ) -> List[FormulaEntry]:
        """Return library entries whose topic/subtopic matches any listed domain.

        If domains is ``["unknown"]`` or empty, the filter is skipped and the
        full library is returned (graceful degradation for unseen domains).
        """
        if not domains or domains == ["unknown"]:
            return list(self._library)

        domain_set = {d.lower() for d in domains}
        sub_set = {s.lower() for s in sub_domains}

        filtered = [
            e for e in self._library
            if e.topic.lower() in domain_set or e.subtopic.lower() in sub_set
        ]

        if not filtered:
            # No topic match — fall through to full library (quantity match will narrow it)
            logger.debug(
                "Topic filter produced no matches for domains=%s; using full library.", domains
            )
            return list(self._library)

        return filtered

    def _score_candidates(
        self,
        candidates: List[FormulaEntry],
        canonical_vars: Set[str],
        direct_vars: Set[str],
        formula_name: str = "",
        template_name: str = "",
        step_goal: str = "",
    ) -> List[Tuple[FormulaEntry, float]]:
        """Score formula entries against a step's variables and parser hints.

        The parser's ``formula_name`` and ``template_name`` are treated as
        strong identity signals that dominate when they match a library
        entry's formula string or subtopic.  Canonical quantity overlap acts
        as a softer fallback that disambiguates remaining ties.
        """
        norm_template = _normalize_name(template_name)
        formula_tokens = _tokens(formula_name)
        template_tokens = _tokens(template_name)
        goal_tokens = _tokens(step_goal)
        scored = []
        for entry in candidates:
            entry_canonicals = set(entry.canonical_quantity_names)
            entry_symbols = set(entry.target_quantities)

            # ── 1. Name match: template_name vs entry subtopic / id ──────
            # Strongest signal — parser already identified the formula type.
            norm_subtopic = _normalize_name(entry.subtopic)
            norm_id = _normalize_name(entry.id)
            if norm_template and (
                norm_template == norm_subtopic
                or norm_subtopic in norm_template
                or norm_template in norm_subtopic
                or norm_template == norm_id
            ):
                name_match_score = 3.0
            else:
                name_match_score = 0.0

            # ── 2. Formula string similarity ─────────────────────────────
            formula_sim = _formula_similarity(formula_name, entry.formula)
            formula_str_score = 3.0 * formula_sim

            # ── 3. Token overlap (softer formula/template matching) ──────
            entry_formula_tokens = _tokens(f"{entry.formula} {entry.sympy_expr} {entry.text}")
            formula_token_score = _overlap_score(formula_tokens, entry_formula_tokens)
            entry_topic_tokens = _tokens(
                f"{entry.topic} {entry.subtopic} {entry.text} {entry.premise_text}"
            )
            template_token_score = _overlap_score(template_tokens, entry_topic_tokens)
            goal_score = _overlap_score(goal_tokens, entry_topic_tokens)

            # ── 4. Canonical quantity overlap (F1) ───────────────────────
            intersection = len(entry_canonicals & canonical_vars)
            if intersection > 0 and entry_canonicals and canonical_vars:
                precision = intersection / len(entry_canonicals)
                recall = intersection / len(canonical_vars)
                canonical_score = 2 * precision * recall / (precision + recall)
            else:
                canonical_score = 0.0

            # ── 5. Direct symbol bonus ───────────────────────────────────
            direct_overlap = len(entry_symbols & direct_vars)
            direct_score = 0.2 * direct_overlap / max(len(direct_vars), 1)

            # ── Combined score ───────────────────────────────────────────
            # Name/formula identity signals dominate; canonical overlap is
            # the tiebreaker for entries the parser didn't name exactly.
            has_strong_signal = name_match_score > 0 or formula_str_score >= 2.0

            score = (
                name_match_score
                + formula_str_score
                + 0.8 * formula_token_score
                + 0.5 * template_token_score
                + 0.2 * goal_score
                + (0.5 if has_strong_signal else 1.0) * canonical_score
                + direct_score
            )

            if score > 0:
                scored.append((entry, score))

        return scored

    def _embedding_fallback(
        self,
        problem_text: str,
        step_goal: str,
    ) -> List[Tuple[FormulaEntry, float]]:
        """Tier 2: embedding similarity search over formula text fields.

        Returns entries whose cosine similarity to the query exceeds
        ``_EMBEDDING_THRESHOLD``, sorted descending by similarity.
        """
        if self._embed_model is None or self._library_embeddings is None:
            return []

        query = f"{step_goal}. {problem_text}"
        try:
            query_emb = self._embed_model.encode(
                [query], convert_to_numpy=True, show_progress_bar=False
            )[0]
            # Cosine similarity
            norms = _np.linalg.norm(self._library_embeddings, axis=1)
            q_norm = _np.linalg.norm(query_emb)
            if q_norm == 0 or _np.any(norms == 0):
                return []
            sims = (self._library_embeddings @ query_emb) / (norms * q_norm)
            results = [
                (self._library[i], float(sims[i]))
                for i in range(len(self._library))
                if sims[i] >= _EMBEDDING_THRESHOLD
            ]
            return sorted(results, key=lambda x: -x[1])
        except Exception as exc:
            logger.warning("Embedding search failed: %s", exc)
            return []

    def _build_formula_sets(
        self,
        step_candidates: Dict[str, List[Tuple[FormulaEntry, float]]],
        step_ids: List[str],
        beam_n: int,
    ) -> List[FormulaSet]:
        """Construct up to *beam_n* non-duplicate formula sets.

        Path 0 = best candidate for every step.
        Paths 1+ = vary one step at a time (the step with the most alternatives
        first), keeping all other steps at their best candidate.  This gives
        maximum diversity for the beam search without combinatorial explosion.
        """
        if not step_ids:
            return [FormulaSet(formulas={}, retrieval_confidence=0.0, path_index=0)]

        per_step = {
            sid: sorted(step_candidates.get(sid, []), key=lambda x: -x[1])
            for sid in step_ids
        }

        def _make_set(formulas: Dict[str, Optional[Tuple[FormulaEntry, float]]], idx: int) -> FormulaSet:
            resolved: Dict[str, Optional[FormulaEntry]] = {}
            scores = []
            for sid in step_ids:
                pair = formulas.get(sid)
                if pair is not None:
                    resolved[sid] = pair[0]
                    scores.append(pair[1])
                else:
                    resolved[sid] = None
            confidence = sum(scores) / len(step_ids) if step_ids else 0.0
            confidence = max(0.0, min(1.0, confidence))
            return FormulaSet(formulas=resolved, retrieval_confidence=confidence, path_index=idx)

        def _id_signature(formula_set: FormulaSet) -> Tuple[Optional[str], ...]:
            return tuple(
                (v.id if v else None) for _, v in sorted(formula_set.formulas.items())
            )

        # Base path: best candidate (index 0) per step
        base: Dict[str, Optional[Tuple[FormulaEntry, float]]] = {
            sid: per_step[sid][0] if per_step[sid] else None
            for sid in step_ids
        }
        seen_sigs = set()
        formula_sets: List[FormulaSet] = []

        base_set = _make_set(base, 0)
        formula_sets.append(base_set)
        seen_sigs.add(_id_signature(base_set))

        # Sort steps by number of alternatives (most ambiguous first)
        steps_by_alternatives = sorted(
            step_ids,
            key=lambda sid: -len(per_step[sid]),
        )

        for step_id in steps_by_alternatives:
            if len(formula_sets) >= beam_n:
                break
            candidates = per_step[step_id]
            for alt_idx in range(1, len(candidates)):
                if len(formula_sets) >= beam_n:
                    break
                variant = {**base, step_id: candidates[alt_idx]}
                candidate_set = _make_set(variant, len(formula_sets))
                sig = _id_signature(candidate_set)
                if sig not in seen_sigs:
                    formula_sets.append(candidate_set)
                    seen_sigs.add(sig)

        return formula_sets
