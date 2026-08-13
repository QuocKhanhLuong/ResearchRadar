# ResearchRadar

ResearchRadar is a private, single-user research assistant. Discord is its interface;
research discovery, paper reading, memory, and scheduled monitoring live in a small
Python application independent of Discord.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
ruff check .
```

The current bootstrap has no required API credentials. A Discord token is required
only when launching the Discord bot.

## Design principles

- One user, one process, and SQLite by default.
- Provider-neutral paper and model abstractions.
- Deterministic discovery, deduplication, and ranking before LLM assistance.
- Evidence and provenance are retained whenever possible.

