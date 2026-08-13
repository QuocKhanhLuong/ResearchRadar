"""Small, domain-oriented exception hierarchy for user-facing failures."""


class ResearchRadarError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ResearchRadarError):
    """Raised when required runtime configuration is unavailable."""


class ProviderUnavailableError(ResearchRadarError):
    """Raised when a scholarly provider cannot complete a request."""


class PaperNotFoundError(ResearchRadarError):
    """Raised when a requested paper cannot be resolved."""


class PaperParseError(ResearchRadarError):
    """Raised when paper text cannot be extracted reliably."""


class LLMUnavailableError(ResearchRadarError):
    """Raised when configured model inference is unavailable."""


class LLMResponseError(ResearchRadarError):
    """Raised when a reachable model returns invalid structured data."""
