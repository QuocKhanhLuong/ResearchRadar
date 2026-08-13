"""Application entry point. Discord startup is added by the bot shell."""

from __future__ import annotations

import logging

from research_radar.config import get_settings
from research_radar.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Load configuration without requiring optional provider credentials."""

    configure_logging()
    settings = get_settings()
    logger.info("ResearchRadar configuration loaded (database=%s).", settings.database_url)


if __name__ == "__main__":
    main()
