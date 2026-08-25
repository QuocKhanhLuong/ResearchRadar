"""Capture policy deciding which user messages become personal memory.

The policy is deterministic: no LLM call, no network, no logging. Given a
user message it returns a :class:`CaptureDecision` saying whether the text
should be persisted as user memory, into which :class:`MemoryClass` bucket
it falls, a short safe-to-log ``reason``, and a ``redacted_text`` that is
safe to persist (empty whenever the decision is not to store).

Rejection precedence inside :meth:`MemoryCapturePolicy.evaluate_user_message`:

1. capture disabled,
2. empty text,
3. secret detected (whole message rejected, never echoed),
4. stack traces,
5. shell/log dumps,
6. raw PDF-style text dumps,
7. non-durable chatter (greetings/thanks/acks, bare questions, too-short),
8. classification; ambiguous but clearly durable first-person statements
   fall back to ``MemoryClass.PREFERENCE`` instead of being dropped.

Assistant output is never stored, unconditionally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_radar.memory.models import MemoryClass
from research_radar.memory.secrets import contains_secret, redact_secrets


@dataclass(frozen=True, slots=True)
class CaptureDecision:
    """Outcome of evaluating one message against the capture policy."""

    should_store: bool
    memory_class: MemoryClass | None
    reason: str  # short, safe-to-log; never echoes rejected content
    redacted_text: str  # the text safe to persist ("" when should_store is False)


REASON_DISABLED = "capture_disabled"
REASON_EMPTY = "empty_text"
REASON_SECRET = "secret_detected"
REASON_STACK_TRACE = "stack_trace"
REASON_SHELL_LOG_DUMP = "shell_or_log_dump"
REASON_PDF_DUMP = "pdf_text_dump"
REASON_CHATTER = "non_durable_chatter"
REASON_QUESTION = "question_not_durable"
REASON_TOO_SHORT = "message_too_short"
REASON_NOT_DURABLE = "not_durable"
REASON_ASSISTANT_NEVER = "assistant_output_never_stored"


# --- shared text-shape detectors -------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and trim."""
    return " ".join(text.split())


_FIRST_PERSON_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_'])(?:i|i'm|i've|i'd|i'll|i am|i have|i had|i was"
    r"|i will|i would|my|me|mine|myself|we|we're|we've|we'll|our|ours|us)"
    r"(?![A-Za-z0-9_'])"
)


def _has_first_person(text: str) -> bool:
    """True when the text speaks in the first person."""
    return bool(_FIRST_PERSON_RE.search(text))


_TRACEBACK_HEADER_RE = re.compile(r"^\s*Traceback \(most recent call last\):?\s*$")
_TRACEBACK_FRAME_RE = re.compile(r'^\s*File "[^"]+", line \d+')
_EXCEPTION_NAME_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt|Exit)\b"
)
_EXCEPTION_TAIL_RE = re.compile(
    r"(?:^|[\s.])[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt|Exit)"
    r"\s*:?\s*$"
)


def _is_stack_trace(text: str) -> bool:
    """Detect Python-style tracebacks pasted into the message."""
    lines = text.splitlines()
    exception_lines = 0
    for line in lines:
        if _TRACEBACK_HEADER_RE.match(line) or _TRACEBACK_FRAME_RE.match(line):
            return True
        stripped = line.strip()
        started = _EXCEPTION_NAME_RE.match(stripped)
        if (started and (started.end() == len(stripped) or stripped[started.end()] == ":")) or (
            stripped and _EXCEPTION_TAIL_RE.search(stripped)
        ):
            exception_lines += 1
    return exception_lines >= 3


_TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\[\d{1,2}:\d{2}(?::\d{2})?\]"
    r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"|\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}"
    r"|\d{1,2}:\d{2}:\d{2})"
)
_SHELL_PREFIXES = ("$ ", ">", "#")


def _is_shell_or_log_dump(text: str) -> bool:
    """Detect blocks of shell commands, prompts, or timestamped log lines."""
    flagged = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_SHELL_PREFIXES) or _TIMESTAMP_PREFIX_RE.match(line):
            flagged += 1
    return flagged >= 3


_PDF_DUMP_MIN_CHARS = 2000


def _is_pdf_style_dump(text: str) -> bool:
    """Detect a very long single paragraph with no first-person marker."""
    if len(text.strip()) <= _PDF_DUMP_MIN_CHARS:
        return False
    if "\n\n" in text.strip():
        return False
    return not _has_first_person(text)


_CHATTER_WORDS = frozenset(
    {
        "a", "all", "awesome", "bot", "bye", "cool", "copy", "do", "done",
        "everyone", "folks", "gn", "gm", "good", "got", "great", "guys",
        "hello", "hey", "hi", "hiya", "howdy", "it", "kk", "later", "lmao",
        "lol", "lots", "man", "many", "me", "much", "nice", "np", "ok",
        "okay", "on", "oops", "perfect", "please", "roger", "sorry", "sup",
        "sweet", "team", "thanks", "that", "there", "tho", "then", "thx",
        "ty", "tysm", "very", "welcome", "will", "woah", "wow", "yeah",
        "yes", "yep", "yo", "you", "your", "yup",
    }
)
_CHATTER_PHRASES = frozenset(
    {
        "good morning", "good afternoon", "good evening", "good night",
        "thank you", "thanks a lot", "thanks so much", "much appreciated",
        "got it", "will do", "on it", "copy that", "sounds good",
        "sounds great", "no worries", "you're welcome", "see you",
        "see you later", "talk soon", "go ahead", "carry on", "as you wish",
    }
)


def _is_pure_chatter(text: str) -> bool:
    """Detect messages that are only a greeting, thanks, or acknowledgement."""
    lowered = text.lower()
    if lowered in _CHATTER_PHRASES:
        return True
    tokens = [token.strip(".,!?…;:\"'()[]") for token in lowered.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return True
    return all(token in _CHATTER_WORDS for token in tokens)


# --- durable-statement classifiers -----------------------------------------
# Order matters: the most specific intent wins. Checked top to bottom.


def _phrase_re(*phrases: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex matching any of the given phrases."""
    return re.compile(r"(?i)" + "|".join(re.escape(p) for p in phrases))


_REJECTED_IDEA_RE = _phrase_re(
    "don't want to pursue", "do not want to pursue", "not going to pursue",
    "no longer interested in", "not pursuing", "stop working on",
    "stopped working on", "abandon", "abandoning", "kill off", "scrap",
)
_DROP_WORD_RE = re.compile(r"(?i)\bdrop(?:ping|ped|s)?\b")

_PROJECT_DECISION_RE = _phrase_re(
    "i decided", "we decided", "i've decided", "we've decided", "decided to",
    "going with", "went with", "settled on", "settled for", "we're using",
    "we are using", "we'll use", "i'm going with",
)

_WORKFLOW_RE = _phrase_re(
    "always run tests", "run tests first", "tests first", "always use",
    "always include", "keep commits small", "small commits",
    "commit messages should", "conventional commits", "before committing",
    "when you finish", "make sure to", "be sure to", "don't forget to",
    "never push", "never commit", "no emoji", "in plain english",
    "bullet points", "one PR per",
)

_TOOL_TOKENS = (
    "pytest", "ruff", "mypy", "black", "isort", "uv", "pip", "poetry",
    "conda", "docker", "kubernetes", "git", "github", "gitlab", "sqlite",
    "postgresql", "postgres", "mysql", "redis", "mongodb", "react", "vue",
    "svelte", "next.js", "node", "deno", "bun", "typescript", "javascript",
    "python", "rust", "golang", "kotlin", "swift", "pydantic", "fastapi",
    "flask", "django", "sqlalchemy", "httpx", "discord.py", "numpy",
    "pandas", "polars", "torch", "tensorflow", "jax", "sklearn",
    "scikit-learn", "matplotlib", "seaborn", "plotly", "jupyter", "vscode",
    "vim", "neovim", "emacs", "zsh", "bash", "tmux", "make", "cmake",
    "cargo", "npm", "yarn", "pnpm", "homebrew", "latex", "overleaf",
    "zotero", "obsidian",
)
_TOOL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(t) for t in _TOOL_TOKENS) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PREFERENCE_VERB_RE = re.compile(
    r"(?i)\b(?:prefer|prefers|preferred|like|likes|liked|love|loves|loved"
    r"|hate|hates|hated|enjoy|enjoys|favor|favour|favourite|favorite|avoid"
    r"|rather|stick with|sticking with|switch to|switching to|switched to"
    r"|migrate to|migrating to|migrated to|move to|moving to|moved to"
    r"|instead of)\b"
)

_RESEARCH_DIRECTION_RE = re.compile(
    r"(?i)(?:\bnext\b[^.!?]{0,60}\b(?:explore|investigate|dig into|look into)\b"
    r"|\b(?:planning|plan|going)\s+to\s+(?:look\s+at|look\s+into|explore|examine|review)\b"
    r"|\bwant\s+to\s+explore\b[^.!?]{0,40}\bnext\b)"
)

_GOAL_RE = _phrase_re(
    "my goal is", "my goals are", "my aim is", "i want to", "i wanna",
    "i'd like to", "i would like to", "i'm trying to", "i am trying to",
    "aiming to", "we're trying to", "i hope to",
)

_CONSTRAINT_RE = _phrase_re(
    "i only have", "i've only got", "i can't use", "i cannot use",
    "can't afford", "limited to", "must not", "budget", "deadline",
    "due date", "no more than", "at most",
)

_FUTURE_TIME_RE = re.compile(
    r"(?i)\b(?:tomorrow|tonight|soon|shortly"
    r"|(?:this|next|coming)\s+(?:week|weekend|month|quarter|year"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|by\s+(?:end\s+of\s+(?:the\s+)?)?(?:week|month|year|day)"
    r"|in\s+\d+\s+(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)"
    r"|(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|q[1-4]\s*['’]?\s*\d{2,4}"
    r"|eod|eow)\b"
)

_RESEARCH_INTEREST_RE = _phrase_re(
    "i'm interested in", "i am interested in", "i'm researching",
    "i am researching", "i research", "i work on", "i'm working on",
    "i work with", "my research focuses on", "my focus is on",
    "i care about", "i study",
)

_PREFERENCE_RE = _phrase_re(
    "i prefer", "i'd rather", "i would rather", "i like", "i don't like",
    "i hate", "i usually", "i generally", "i typically", "i love", "i enjoy",
)

_FALLBACK_CUE_RE = re.compile(
    r"(?i)\b(?:prefer|want|need|plan|decid|goal|aim|trying|working on"
    r"|interested|care about|usually|rather|always|never|keep|struggle"
    r"|struggling|noticed?|realized?|hate|love|avoid|favourite|favorite"
    r"|wish|hope|intend|going to|found)\b"
)


def _classify(text: str) -> MemoryClass | None:
    """Map a durable statement to its MemoryClass, or None if unclassifiable."""
    lowered = text.lower()

    def matches(pattern: re.Pattern[str]) -> bool:
        return bool(pattern.search(lowered))

    if matches(_REJECTED_IDEA_RE) or matches(_DROP_WORD_RE):
        return MemoryClass.REJECTED_IDEA
    if matches(_PROJECT_DECISION_RE):
        return MemoryClass.PROJECT_DECISION
    if matches(_WORKFLOW_RE):
        return MemoryClass.WORKFLOW_PREFERENCE
    if matches(_TOOL_TOKEN_RE) and matches(_PREFERENCE_VERB_RE):
        return MemoryClass.TOOL_PREFERENCE
    if matches(_RESEARCH_DIRECTION_RE):
        return MemoryClass.RESEARCH_DIRECTION
    if matches(_GOAL_RE):
        return MemoryClass.GOAL
    if matches(_CONSTRAINT_RE):
        return MemoryClass.CONSTRAINT
    if matches(_FUTURE_TIME_RE) and matches(_FIRST_PERSON_RE):
        return MemoryClass.TEMPORAL_PLAN
    if matches(_RESEARCH_INTEREST_RE):
        return MemoryClass.RESEARCH_INTEREST
    if matches(_PREFERENCE_RE):
        return MemoryClass.PREFERENCE
    return None


# --- policy ----------------------------------------------------------------


class MemoryCapturePolicy:
    """Decides which Discord user messages may be persisted as memory."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Create a policy; ``enabled=False`` rejects every message."""
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether memory capture is currently switched on."""
        return self._enabled

    def evaluate_user_message(self, text: str) -> CaptureDecision:
        """Evaluate one user message and decide whether it becomes memory.

        The returned reason is a fixed, short label that is safe to log; it
        never includes the message text or any matched secret substring.
        """
        stripped = text.strip()
        if not self._enabled:
            return CaptureDecision(False, None, REASON_DISABLED, "")
        if not stripped:
            return CaptureDecision(False, None, REASON_EMPTY, "")
        if contains_secret(stripped):
            return CaptureDecision(False, None, REASON_SECRET, "")

        if _is_stack_trace(stripped):
            return CaptureDecision(False, None, REASON_STACK_TRACE, "")
        if _is_shell_or_log_dump(stripped):
            return CaptureDecision(False, None, REASON_SHELL_LOG_DUMP, "")
        if _is_pdf_style_dump(stripped):
            return CaptureDecision(False, None, REASON_PDF_DUMP, "")
        if _is_pure_chatter(stripped):
            return CaptureDecision(False, None, REASON_CHATTER, "")

        classified = _classify(stripped)
        is_question = stripped.endswith("?")
        if is_question and classified is None:
            return CaptureDecision(False, None, REASON_QUESTION, "")
        if classified is not None:
            return self._accept(classified, stripped)
        if len(stripped) < 15:
            return CaptureDecision(False, None, REASON_TOO_SHORT, "")

        # Ambiguous but clearly durable first-person statements fall back to
        # PREFERENCE rather than being dropped.
        if _has_first_person(stripped) and _FALLBACK_CUE_RE.search(stripped):
            return self._accept(MemoryClass.PREFERENCE, stripped)
        return CaptureDecision(False, None, REASON_NOT_DURABLE, "")

    def evaluate_assistant_message(self, text: str) -> CaptureDecision:
        """Always rejects: assistant output must never become user memory."""
        return CaptureDecision(False, None, REASON_ASSISTANT_NEVER, "")

    def _accept(self, memory_class: MemoryClass, stripped: str) -> CaptureDecision:
        """Build an accepting decision with defensively redacted text."""
        redacted = redact_secrets(_normalize_whitespace(stripped))
        return CaptureDecision(True, memory_class, f"accepted_{memory_class.value}", redacted)
