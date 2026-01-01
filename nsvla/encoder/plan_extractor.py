"""Episode-level plan extraction  p = P(x, o_0)  (NS-VLA §4.1, Eq. 4).

Two routes, first-then-fallback (paper §4.1 lines 108-110):
  1. a **deterministic grammar parser** on the instruction string (this file's
     rule engine, covering the templated LIBERO instructions), invoked first;
  2. a **VLM fallback** (frozen Qwen3-VL-2B, strict-JSON contract, App. A) invoked
     *only* when the grammar cannot resolve a clause. Interface + prompt constant +
     output schema validator live here; the actual VLM call is a stub (needs GPU).

The plan is extracted ONCE at episode start and stays fixed (Eq. 4). This module
is the highest-risk, per-benchmark work: every new benchmark redoes the
keyword map / argument parsing. Keep the rules here and data-driven.

Design of the LIBERO clause splitter (the crux — naive `and`-splitting breaks):
`and` inside LIBERO instructions has four roles, and only one is a clause break:
  (a) clause separator  -> "... and put ...", "... and close it"   (followed by a VERB)
  (b) spatial relation  -> "between the plate and the ramekin"      (followed by a noun)
  (c) conjoined objects -> "both the soup and the butter"           ("both ...")
  (d) color compound    -> "the yellow and white mug"               (followed by a noun)
We split on `and` **only when the next token is a verb** (role a); roles b/c/d
stay inside their clause, and conjoined-object clauses (role c) are expanded to
one primitive per object afterwards.
"""
from __future__ import annotations

import json
import re
from typing import Any

from nsvla.primitives.vocab import PAD_OP, Plan, Primitive, PrimitiveVocab, default_vocab

# ---------------------------------------------------------------------------
# Appendix A: strict-JSON system prompt for the VLM fallback route.
# ---------------------------------------------------------------------------
VLM_SYSTEM_PROMPT = """You are the robot's brain.
Given a natural-language instruction and an observation, output a plan as STRICT JSON.
Output MUST be a JSON array. Each step MUST be a JSON object with keys:
- "op"   (action)
- "args" (object)
- "support" (support)
Do not include comments, trailing commas, or any extra fields.
Return JSON only --- no markdown, no explanations."""

# Verbs that may open a new clause (used for the verb-gated `and` split, role a).
_VERB_LOOKAHEAD = r"(?:pick|put|place|open|close|turn|push|grasp|lift|slide|move)"

# Clause splitter: unconditional separators (comma/semicolon/then/after that) OR
# an ` and ` immediately followed by a clause-opening verb.
_SPLIT = re.compile(
    r"\s*(?:,|;|\bafter that\b|\band then\b|\bthen\b)\s*"
    r"|\s+\band\b\s+(?=" + _VERB_LOOKAHEAD + r"\b)",
    re.I,
)

# Leading verb -> op family. Longest phrases first; first hit wins.
_VERB_OP = [
    ("pick up", "pick"),
    ("pick", "pick"),
    ("grasp", "pick"),
    ("lift", "pick"),
    ("turn on", "turn_on"),
    ("turn off", "close"),
    ("put", "_place_family"),
    ("place", "_place_family"),
    ("open", "open"),
    ("close", "close"),
    ("push", "push_to"),
    ("slide", "push_to"),
    ("move", "push_to"),
]

# Preposition -> op for the put/place family. Earliest-position match wins; on a
# positional tie the longest match wins (so "on the top of" beats bare "on").
_PLACE_PREPS = [
    (re.compile(r"\bto the (?:right|left|front|back|rear) of\b", re.I), "place_rel"),
    (re.compile(r"\binside\b", re.I), "place_in"),
    (re.compile(r"\binto\b", re.I), "place_in"),
    (re.compile(r"\bin\b", re.I), "place_in"),
    (re.compile(r"\bon the top of\b", re.I), "place_on"),
    (re.compile(r"\bon top of\b", re.I), "place_on"),
    (re.compile(r"\bonto\b", re.I), "place_on"),
    (re.compile(r"\bon\b", re.I), "place_on"),
]

_PUSH_PREP = re.compile(r"\bto (?:the )?", re.I)
_PRONOUNS = {"it", "them", "it.", "one"}
_ARTICLES = ("the ", "a ", "an ", "both ")


def _strip_article(np_text: str) -> str:
    s = np_text.strip().rstrip(".")
    low = s.lower()
    for art in _ARTICLES:
        if low.startswith(art):
            return s[len(art):].strip()
    return s


def _split_verb(clause: str) -> tuple[str, str] | None:
    """Return (op_family, remainder) for the leading verb, or None if unmatched."""
    c = clause.strip().lower()
    for phrase, op in _VERB_OP:
        if c == phrase or c.startswith(phrase + " "):
            return op, clause.strip()[len(phrase):].strip()
    return None


def _first_prep(text: str) -> tuple[int, int, str] | None:
    """Earliest (start, end, op) preposition match in a put/place clause."""
    best: tuple[int, int, str] | None = None
    for pat, op in _PLACE_PREPS:
        m = pat.search(text)
        if m is None:
            continue
        if best is None or m.start() < best[0] or (m.start() == best[0] and m.end() > best[1]):
            best = (m.start(), m.end(), op)
    return best


def _parse_clause(
    clause: str,
    vocab: PrimitiveVocab,
    prev_object: str | None,
    prev_support: str | None,
) -> list[Primitive]:
    """Parse one clause into >=1 primitives. Empty list => unresolved by the grammar."""
    split = _split_verb(clause)
    if split is None:
        return []
    op_family, remainder = split

    # --- pick / turn_on / open / close: object = whole remainder, no support ----
    if op_family in ("pick", "turn_on", "open", "close"):
        obj = _strip_article(remainder) if remainder else (prev_object or "")
        if obj.lower() in _PRONOUNS or not obj:
            # "close it" -> the last container (support), else the last object.
            obj = prev_support or prev_object or ""
        if not vocab.has(op_family):
            return []
        return [Primitive(op_family, object=obj or None)]

    # --- push / slide / move -> push_to: object before " to ...", support after -
    if op_family == "push_to":
        m = _PUSH_PREP.search(remainder)
        if m is not None:
            obj = _strip_article(remainder[: m.start()])
            sup = _strip_article(remainder[m.end():])
        else:
            obj, sup = _strip_article(remainder), None
        if not vocab.has("push_to"):
            return []
        return [Primitive("push_to", object=obj or None, support=sup or None)]

    # --- put / place family: preposition decides place_on / place_in / place_rel -
    prep = _first_prep(remainder)
    if prep is None:
        # bare "put X" with no preposition -> default place_on (rare / unseen)
        op, obj_region, sup_region = "place_on", remainder.strip(), ""
    else:
        start, end, op = prep
        obj_region = remainder[:start].strip()
        sup_region = remainder[end:].strip()
    if not vocab.has(op) or not vocab.has("pick"):
        return []

    sup = _strip_article(sup_region) if sup_region else None
    if sup is not None and sup.lower() in _PRONOUNS:
        sup = prev_object  # "on it" -> the object named in the previous clause
    if not sup_region and op == "place_in":
        sup = prev_object or prev_support  # "put the bowl inside" -> previous container

    # Operation-level expansion (Fig. 4a): every transport clause maps a manipulated
    # object to TWO primitives, pick(object) + place_*(object, support). A conjoined
    # "both X and Y" (role c) expands per object. A PRONOUN object ("place it ...")
    # was already grasped by a prior pick clause, so it stays place-only (this keeps
    # the explicit two-verb "pick up X and place it ..." sentences at pick + place).
    prims: list[Primitive] = []
    for o in _expand_conjoined(obj_region):
        is_pronoun = o.strip().lower() in _PRONOUNS or _strip_article(o).lower() in _PRONOUNS
        o_clean = _strip_article(o)
        if o_clean.lower() in _PRONOUNS or not o_clean:
            o_clean = prev_object or ""
        if not is_pronoun:
            prims.append(Primitive("pick", object=o_clean or None))
        prims.append(Primitive(op, object=o_clean or None, support=sup))
    return prims


def _singularize(noun: str) -> str:
    """Naive plural->singular for the "both <plural noun>" case ("moka pots" -> "moka pot")."""
    toks = noun.split()
    if toks and len(toks[-1]) > 1 and toks[-1].lower().endswith("s"):
        toks[-1] = toks[-1][:-1]
    return " ".join(toks)


def _expand_conjoined(obj_region: str) -> list[str]:
    """Expand a conjoined / plural object region into one entry per manipulated object.

    Two "both" forms, both producing TWO objects (Fig. 4a: each object is its own
    transport pair pick+place_*):
      * "both X and Y"      -> [X, Y]                     (role c, distinct objects);
      * "both <plural noun>" -> [<singular>, <singular>]  (two rounds, SAME object name;
        e.g. "both moka pots" -> ["moka pot", "moka pot"]). The two identical names
        are disambiguated downstream by grounding (option: same string; alt would be
        "moka pot #1/#2" — we keep the same string here).
    Everything else passes through unchanged.
    """
    low = obj_region.lower()
    if low.startswith("both "):
        body = obj_region[len("both "):].strip()
        if re.search(r"\band\b", body, re.I):
            return [p.strip() for p in re.split(r"\s+\band\b\s+", body, flags=re.I) if p.strip()]
        singular = _singularize(body)
        return [singular, singular]   # "both <plural noun>" -> two transport rounds
    return [obj_region]


def extract_plan(
    instruction: str,
    vocab: PrimitiveVocab | None = None,
    max_len: int = 6,
    vlm: Any = None,
) -> Plan:
    """instruction -> fixed episode plan p (Eq. 4).

    Grammar parser first; the ``vlm`` fallback (if given) resolves clauses the
    grammar cannot. Unresolved clauses with no VLM become ``<pad>`` slots.
    """
    vocab = vocab or default_vocab()
    clauses = [c for c in _SPLIT.split(instruction.strip()) if c and c.strip()]
    prims: list[Primitive] = []
    prev_object: str | None = None
    prev_support: str | None = None
    for clause in clauses:
        parsed = _parse_clause(clause, vocab, prev_object, prev_support)
        if not parsed:
            if vlm is not None:
                parsed = _vlm_fallback_clause(clause, vocab, vlm)
            else:
                parsed = [Primitive(PAD_OP)]  # unresolved -> pad (pointer skips it)
        for p in parsed:
            if p.is_real() and p.object and p.object.lower() not in _PRONOUNS:
                prev_object = p.object
            if p.is_real() and p.support and p.support.lower() not in _PRONOUNS:
                prev_support = p.support
            prims.append(p)
    if not prims:
        prims = [Primitive(PAD_OP)]
    return Plan(prims[:max_len], max_len=max_len)


def is_fully_resolved(plan: Plan) -> bool:
    """True iff every slot is a real primitive (no unresolved ``<pad>``)."""
    return len(plan.primitives) >= 1 and all(p.is_real() for p in plan.primitives)


# ---------------------------------------------------------------------------
# VLM fallback route (App. A). Prompt + schema validation implemented; call stub.
# ---------------------------------------------------------------------------
def build_vlm_messages(instruction: str, observation_note: str = "img and state") -> list[dict]:
    """Assemble the strict-JSON chat messages for the Qwen3-VL-2B fallback (App. A)."""
    user = json.dumps(
        [{"task": instruction, "observation": observation_note}], ensure_ascii=False
    )
    return [
        {"role": "system", "content": VLM_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_vlm_plan(
    raw: str | list, vocab: PrimitiveVocab | None = None, max_len: int = 6
) -> Plan:
    """Validate a VLM strict-JSON response against the plan schema (App. A) -> Plan.

    Raises ``ValueError`` on any contract violation: non-array root, missing/extra
    keys, unknown op, or non-string arguments. Enforcing this keeps the downstream
    symbolic decoding stable (paper §A: "reduces formatting variance").
    """
    vocab = vocab or default_vocab()
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, list):
        raise ValueError("VLM plan must be a JSON array")
    prims: list[Primitive] = []
    for i, step in enumerate(data):
        if not isinstance(step, dict):
            raise ValueError(f"step {i} is not a JSON object")
        if "op" not in step:
            raise ValueError(f"step {i} missing required key 'op'")
        extra = set(step) - {"op", "args", "support"}
        if extra:
            raise ValueError(f"step {i} has forbidden extra fields: {sorted(extra)}")
        op = step["op"]
        if not isinstance(op, str) or not vocab.has(op):
            raise ValueError(f"step {i} has unknown op: {op!r}")
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"step {i} 'args' must be an object")
        obj = args.get("object")
        sup = args.get("support", step.get("support"))
        for name, val in (("object", obj), ("support", sup)):
            if val is not None and not isinstance(val, str):
                raise ValueError(f"step {i} arg '{name}' must be a string or absent")
        prims.append(Primitive(op, object=obj, support=sup))
    if not prims:
        raise ValueError("VLM plan is empty")
    return Plan(prims[:max_len], max_len=max_len)


def _vlm_fallback_clause(clause: str, vocab: PrimitiveVocab, vlm: Any) -> list[Primitive]:
    """Resolve one clause with the frozen VLM: prompt, decode, validate against the schema.

    Reached only for a clause the grammar cannot parse, which on the templated LIBERO
    instructions never happens; it is the escape hatch for free-form instructions.
    """
    raise NotImplementedError(
        "the VLM plan fallback needs a loaded encoder: build the messages with "
        "build_vlm_messages(clause), decode under the strict-JSON contract, then "
        "validate_vlm_plan(...)"
    )
