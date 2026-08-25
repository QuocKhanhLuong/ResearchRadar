"""Deterministic-first intent routing for personal research chat.

Routing order (first confident match wins):

1. Empty / whitespace-only text -> ``CONVERSATIONAL`` (no backend work).
2. Greeting, small talk, or a meta question about the bot -> ``CONVERSATIONAL``.
3. Explicit ``project_hint`` metadata -> ``PROJECT_RESEARCH``.
4. First-person durable statement ("I prefer ...", "I don't want to pursue
   ...", "My goal is ...") -> ``CONVERSATIONAL`` with ``needs_user_memory``
   set so ``MemoryCapturePolicy`` later decides what to store.
5. First-person memory question ("what do I ...", "my interests",
   "remember", "what am I working on") -> ``PERSONAL_MEMORY``.
6. Textual "my project <name>" reference -> ``PROJECT_RESEARCH``.
7. Research intent ("find ...", "recent papers", "compare X and Y",
   "state of the art", "survey", "literature") -> ``RESEARCH_STORED`` with
   ``allows_live_discovery=True``.

**Ambiguity condition for the optional LLM assist.** The deterministic pass is
"genuinely ambiguous" if and only if *no* rule above fired and the message
carries at least three word tokens — i.e. the deterministic fallback of
``CONVERSATIONAL`` is a pure guess for a substantive message. Empty text,
greetings, small talk, meta questions, durable statements, memory questions,
project references, and explicit research intents all fire a rule and
therefore NEVER reach the assist. When the assist is absent, fails, times out,
or returns an invalid payload, the deterministic decision stands; the failure
is logged at debug level without echoing user content.

``RESEARCH_LIVE`` is never returned here. The router only grants permission
(``allows_live_discovery``); ``ChatService`` decides whether live discovery
runs and whether the turn becomes ``RESEARCH_LIVE``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from research_radar.chat.models import ChatMode, ChatRequest
from research_radar.reader.llm.base import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

_ASSIST_TIMEOUT_SECONDS: float = 4.0
_MIN_ASSIST_TOKENS = 3
_MAX_ASSIST_TOPIC_CHARS = 300


class RoutingAssistResponse(BaseModel):
    """Small structured reply consumed by the optional LLM routing assist."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["conversational", "personal_memory", "research"] = "conversational"
    topic: str = ""


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Outcome of routing one chat turn into execution capabilities."""

    mode: ChatMode
    needs_user_memory: bool
    needs_stored_research: bool
    allows_live_discovery: bool
    search_query: str  # normalized retrieval query ("" when not research)
    project_hint: str | None = None


# ---------------------------------------------------------------------------
# Deterministic cue patterns, grouped per rule in documented evaluation order.
# ---------------------------------------------------------------------------

_GREETING_VOCAB = frozenset(
    {
        "hi", "hello", "hey", "heya", "hiya", "howdy", "yo", "sup",
        "thanks", "thank", "thx", "ty", "tysm",
        "good", "morning", "afternoon", "evening", "night",
        "bot", "radar", "there", "everyone", "folks", "guys", "all", "yall",
        "again", "pls", "plz", "please", "ok", "okay", "okie",
        "cool", "nice", "great", "awesome", "perfect", "amazing",
        "lol", "lmao", "haha", "hehe", "hmm", "hm",
        "yes", "yeah", "yep", "no", "nope", "nah", "sure", "fine",
        "whats", "hows", "up", "doing", "going", "it",
    }
)
_SMALLTALK_PHRASES = frozenset(
    {
        "how are you", "how are you doing", "hows it going",
        "whats up", "what is up", "wassup",
        "who are you", "who r u", "what are you", "what are you doing",
        "what can you do", "what do you do", "how do you work",
        "are you there", "you there", "you alive", "anyone there",
        "help", "help me", "help pls", "help please",
        "good bot", "thanks bot", "thank you bot", "nice bot",
        "good morning", "good afternoon", "good evening", "good night",
        "thank you", "many thanks", "thanks a lot",
    }
)
_META_QUESTION_PREFIXES = (
    "who are you",
    "what are you",
    "what can you do",
    "what do you do",
    "how do you work",
    "are you a",
)
_MAX_SMALLTALK_TOKENS = 6
_MAX_META_QUESTION_TOKENS = 8

_DURABLE_STATEMENT_RES = (
    re.compile(
        r"^\s*i\s+(?:really\s+|generally\s+|usually\s+|typically\s+)?prefer\b", re.I
    ),
    re.compile(
        r"^\s*i\s+(?:do\s*n[o']t|never)\s+want\s+to\s+"
        r"(?:pursue|work|explore|see|use|follow)\b",
        re.I,
    ),
    re.compile(r"^\s*my\s+goal\s+is\b", re.I),
)

_MEMORY_QUESTION_RES = (
    re.compile(r"\bwhat\s+do\s+i\b", re.I),
    re.compile(
        r"\bwhat\s+(?:research\s+)?(?:areas?|topics?|fields?|directions?)\s+do\s+i\b", re.I
    ),
    # First-person possessives, allowing one qualifier between "my" and the
    # noun so "my research interests" and "my previous preferences" are caught
    # as readily as the bare "my interests".
    re.compile(
        r"\bmy\s+(?:\w+\s+){0,2}?"
        r"(?:interests?|preferences?|goals?|priorities|constraints?"
        r"|directions?|topics?|areas?|decisions?)\b",
        re.I,
    ),
    re.compile(r"\bremember\b", re.I),
    re.compile(r"\bwhat\s+am\s+i\s+working\s+on\b", re.I),
    re.compile(r"\bwhat\s+do\s+i\s+(?:care\s+about|like|prefer)\b", re.I),
    re.compile(r"\bwhat\s+do\s+you\s+know\s+about\s+(?:me|my)\b", re.I),
)

_MY_PROJECT_RE = re.compile(
    r"\bmy\s+project\s+(?:is\s+|called\s+)?(?P<name>[a-z0-9][\w .\-]{0,60})", re.I
)
_PROJECT_NAME_CUT_RE = re.compile(r"[.,;!?]")

_RESEARCH_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+|pls\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
    r"(?:find|search(?:\s+for)?|look\s+up|show\s+me|list|fetch|pull\s+up)\b",
    re.I,
)
_RESEARCH_CUE_RES = (
    re.compile(
        r"\b(?:recent|latest|newest|new|current)\s+"
        r"(?:papers|work|research|publications|articles|advances|developments|literature)\b",
        re.I,
    ),
    re.compile(r"\bpapers?\s+(?:on|about|for|regarding|covering)\b", re.I),
    re.compile(r"\bany\s+(?:papers|work|research|publications|literature)\b", re.I),
    re.compile(r"\bliterature\b", re.I),
    re.compile(r"\bsurvey\b|\breview\s+of\b", re.I),
    re.compile(r"\bstate\s+of\s+the\s+art\b|\bsota\b", re.I),
    re.compile(r"\b(?:prior|related)\s+work\b", re.I),
    re.compile(r"\bcompare\b[^?.!]*\b(?:and|with|versus|vs\.?|to|against)\b", re.I),
    re.compile(r"\b(?:differences?|similarities?)\s+between\b", re.I),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:new|recent)\s+(?:in|on|with|across)\b", re.I),
    re.compile(r"\bwho\s+(?:is\s+)?(?:working|publishing)\s+on\b", re.I),
)

# Leading-frame patterns used ONLY to normalize the retrieval query; route
# detection never depends on these. Longest / most specific frames first.
_QUERY_POLITE_RE = re.compile(r"^(?:please|pls)\s+", re.I)
_QUERY_AUX_RE = re.compile(r"^(?:can|could|would)\s+you\s+(?:please\s+)?", re.I)
_QUERY_INTERJECTION_RE = re.compile(r"^(?:hey|hi|hello|ok|okay|so|also|now)\b[,.!]?\s*", re.I)
_QUERY_FRAME_RES = (
    re.compile(
        r"^what(?:'s|\s+is|\s+are)\s+(?:the\s+)?(?:recent|latest|current|new)\s+"
        r"(?:work|research|papers|publications|articles|literature)"
        r"\s+(?:on|about|in|regarding)\b\s*",
        re.I,
    ),
    re.compile(
        r"^(?:find\s+me|search\s+for|look\s+up|show\s+me|tell\s+me\s+about"
        r"|give\s+me|get\s+me|pull\s+up|find|search|list|fetch|compare|any)\b\s*",
        re.I,
    ),
    re.compile(
        r"^(?:the\s+)?(?:recent|latest|newest|new|current)\s+"
        r"(?:work|papers|research|publications|articles|advances|developments|literature)"
        r"\s+(?:on|about|for|regarding|in)\b\s*",
        re.I,
    ),
    re.compile(
        r"^(?:the\s+)?(?:work|papers|research|publications|articles|literature"
        r"|info(?:rmation)?)\s+(?:on|about|for|regarding|in)\b\s*",
        re.I,
    ),
    re.compile(r"^everything\s+(?:we\s+have\s+)?(?:about|on)\b\s*", re.I),
    re.compile(r"^more\s+(?:about|on)\b\s*", re.I),
    re.compile(r"^(?:related\s+to|regarding|about)\b\s*", re.I),
)
_QUERY_TRAILING_PUNCT_RE = re.compile(r"[\s.!?,;:\u2026]+$")


def _normalize_text(text: str) -> str:
    """Lowercase, drop apostrophes and non-alphanumerics, collapse spaces."""

    lowered = text.casefold().replace("'", "")
    return " ".join(re.findall(r"[a-z0-9]+", lowered))


def _token_count(text: str) -> int:
    """Count word tokens in free text."""

    return len(re.findall(r"[a-z0-9]+", text.casefold()))


def _is_small_talk(text: str) -> bool:
    """Return True when the whole message is greeting, small talk, or meta."""

    normalized = _normalize_text(text)
    if not normalized:
        return True
    if normalized in _SMALLTALK_PHRASES:
        return True
    tokens = normalized.split()
    if len(tokens) <= _MAX_SMALLTALK_TOKENS and all(t in _GREETING_VOCAB for t in tokens):
        return True
    stripped = text.strip().casefold().rstrip("?!. ")
    for prefix in _META_QUESTION_PREFIXES:
        if stripped.startswith(prefix) and len(tokens) <= _MAX_META_QUESTION_TOKENS:
            return True
    return False


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    """Return True when any pattern matches anywhere in the text."""

    return any(pattern.search(text) for pattern in patterns)


def _extract_project_name(text: str) -> str | None:
    """Extract the project name from a textual 'my project X' reference."""

    found = _MY_PROJECT_RE.search(text)
    if found is None:
        return None
    cut = _PROJECT_NAME_CUT_RE.split(found.group("name").strip(), maxsplit=1)[0]
    name = " ".join(cut.split())[:60]
    return name or None


def _strip_one_frame(text: str) -> str | None:
    """Remove one leading conversational frame; None when no frame matches."""

    for pattern in _QUERY_FRAME_RES:
        stripped = pattern.sub("", text, count=1).strip()
        if stripped and stripped != text:
            return stripped
    return None


def normalize_search_query(text: str) -> str:
    """Strip leading conversational framing and keep the substantive topic.

    Removes polite prefixes ("please", "can you"), leading imperatives
    ("find", "search for", "look up", "show me", "compare", "what is the
    recent work on", ...), connector frames ("recent work on", "papers
    about", ...), surrounding quotes, and trailing punctuation, then
    collapses whitespace. Returns "" only when nothing substantive remains.
    """

    current = text.strip().strip("\"'")
    for _ in range(3):
        stripped = _QUERY_INTERJECTION_RE.sub("", current, count=1).strip()
        if not stripped or stripped == current:
            break
        current = stripped
    for pattern in (_QUERY_POLITE_RE, _QUERY_AUX_RE):
        while True:
            stripped = pattern.sub("", current, count=1).strip()
            if not stripped or stripped == current:
                break
            current = stripped
    for _ in range(6):
        stripped = _strip_one_frame(current)
        if stripped is None:
            break
        current = stripped
    current = _QUERY_TRAILING_PUNCT_RE.sub("", current)
    return " ".join(current.split())


def _fallback_decision() -> RouteDecision:
    """Build the conservative CONVERSATIONAL decision used as default."""

    return RouteDecision(
        mode=ChatMode.CONVERSATIONAL,
        needs_user_memory=False,
        needs_stored_research=False,
        allows_live_discovery=False,
        search_query="",
    )


def _route_deterministic(request: ChatRequest) -> tuple[RouteDecision, bool]:
    """Apply the authoritative rule set.

    Returns the decision plus whether any rule fired. An unfired decision is
    the conservative CONVERSATIONAL guess and is the only state eligible for
    the optional LLM assist.
    """

    text = request.text
    if not text or not text.strip():
        return _fallback_decision(), True

    if _is_small_talk(text):
        return _fallback_decision(), True

    if request.project_hint:
        return (
            RouteDecision(
                mode=ChatMode.PROJECT_RESEARCH,
                needs_user_memory=False,
                needs_stored_research=True,
                allows_live_discovery=False,
                search_query=normalize_search_query(text),
                project_hint=request.project_hint,
            ),
            True,
        )

    if any(pattern.match(text) for pattern in _DURABLE_STATEMENT_RES):
        return (
            RouteDecision(
                mode=ChatMode.CONVERSATIONAL,
                needs_user_memory=True,
                needs_stored_research=False,
                allows_live_discovery=False,
                search_query="",
            ),
            True,
        )

    # A message can carry both signals ("find papers about my research
    # interests"). Research intent then wins the mode, but the personal signal
    # still turns memory retrieval on so the answer can be framed by context.
    memory_question = _matches_any(_MEMORY_QUESTION_RES, text)
    research_intent = bool(
        _RESEARCH_IMPERATIVE_RE.match(text) or _matches_any(_RESEARCH_CUE_RES, text)
    )

    if memory_question and not research_intent:
        return (
            RouteDecision(
                mode=ChatMode.PERSONAL_MEMORY,
                needs_user_memory=True,
                needs_stored_research=False,
                allows_live_discovery=False,
                search_query="",
            ),
            True,
        )

    project_name = _extract_project_name(text)
    if project_name is not None:
        return (
            RouteDecision(
                mode=ChatMode.PROJECT_RESEARCH,
                needs_user_memory=False,
                needs_stored_research=True,
                allows_live_discovery=False,
                search_query=normalize_search_query(text),
                project_hint=project_name,
            ),
            True,
        )

    if research_intent:
        return (
            RouteDecision(
                mode=ChatMode.RESEARCH_STORED,
                needs_user_memory=memory_question,
                needs_stored_research=True,
                allows_live_discovery=True,
                search_query=normalize_search_query(text),
            ),
            True,
        )

    return _fallback_decision(), False


def _assist_messages(request: ChatRequest) -> list[LLMMessage]:
    """Build the bounded classification prompt for the routing assist."""

    system_rules = (
        "You classify one user chat message for a private single-user research "
        "assistant. Reply ONLY with JSON matching the schema "
        '{"mode": "conversational" | "personal_memory" | "research", '
        '"topic": "<topic phrase>"}. '
        "Greetings, thanks, small talk, and questions about the bot are "
        "'conversational' with an empty topic. Questions about the user's own "
        "interests, preferences, goals, or history are 'personal_memory'. "
        "Requests to find, survey, summarize, or compare scientific literature "
        "or technical topics are 'research'; put the substantive topic phrase, "
        "without commands like 'find', into topic."
    )
    return [
        LLMMessage(role="system", content=system_rules),
        LLMMessage(role="user", content=f"Message: {request.text.strip()}"),
    ]


def _decision_from_assist(
    request: ChatRequest, assist: RoutingAssistResponse
) -> RouteDecision | None:
    """Map a valid assist payload to a decision; None keeps the fallback."""

    if assist.mode == "research":
        topic = assist.topic.strip()[:_MAX_ASSIST_TOPIC_CHARS]
        query = normalize_search_query(topic) if topic else normalize_search_query(request.text)
        if not query:
            return None
        return RouteDecision(
            mode=ChatMode.RESEARCH_STORED,
            needs_user_memory=False,
            needs_stored_research=True,
            allows_live_discovery=True,
            search_query=query,
        )
    if assist.mode == "personal_memory":
        return RouteDecision(
            mode=ChatMode.PERSONAL_MEMORY,
            needs_user_memory=True,
            needs_stored_research=False,
            allows_live_discovery=False,
            search_query="",
        )
    return None


class ChatRouter:
    """Classify chat turns deterministically, with optional LLM refinement.

    The router NEVER raises: every provider failure, timeout, or invalid
    response silently falls back to the deterministic decision (logged at
    debug level, sanitized).
    """

    def __init__(self, *, llm_provider: LLMProvider | None = None) -> None:
        self._llm_provider = llm_provider

    async def _assist(self, request: ChatRequest) -> RouteDecision | None:
        """Run the optional refinement pass; never raises, never echoes."""

        provider = self._llm_provider
        if provider is None:
            return None
        try:
            assist = await asyncio.wait_for(
                provider.generate_structured(_assist_messages(request), RoutingAssistResponse),
                timeout=_ASSIST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.debug(
                "Chat routing assist skipped (%s); using deterministic decision.",
                type(exc).__name__,
            )
            return None
        return _decision_from_assist(request, assist)

    async def route(self, request: ChatRequest) -> RouteDecision:
        """Route one chat turn to a :class:`RouteDecision`; never raises."""

        try:
            decision, fired = _route_deterministic(request)
        except Exception as exc:
            logger.debug(
                "Deterministic chat routing degraded (%s); using fallback.",
                type(exc).__name__,
            )
            return _fallback_decision()
        if self._llm_provider is not None and not fired:
            if _token_count(request.text) >= _MIN_ASSIST_TOKENS:
                refined = await self._assist(request)
                if refined is not None:
                    return refined
        return decision
