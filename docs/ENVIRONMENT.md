# Environment & Configuration

All configuration is read from environment variables (optionally via a `.env`
file; copy `.env.example` to start). Every variable has a safe default except
`DISCORD_TOKEN`, which is only required to launch the bot. Empty values are
ignored (`env_ignore_empty`), so `VAR=` behaves exactly like an unset variable.
Secrets are never required in the file for offline or mock operation.

## Feature matrix

| Feature | Required envs | Optional envs | Offline / unset behaviour |
| --- | --- | --- | --- |
| Discord bot launch | `DISCORD_TOKEN` | `DISCORD_GUILD_ID`, `DISCORD_CHANNEL_ID`, `HTTP_TIMEOUT_SECONDS` | Bot simply does not start without the token; everything else runs headless. |
| Discord mention chat | — (bot running) | `DISCORD_CHAT_ON_MENTION`, `DISCORD_ALLOWED_CHANNEL_IDS`, `DISCORD_OWNER_USER_ID`, `DISCORD_DM_CHAT` | Mention chat is on by default in every channel; unset allow-list means no channel restriction and no owner filter. |
| Discord DM chat | — (bot running) | `DISCORD_DM_CHAT` | Enabled by default; set `false` to reject DMs at admission. |
| SQLite storage | — | `DATABASE_URL` | Defaults to `sqlite:///data/research_radar.db`; created on first use. |
| Artifact cache | — | `ARTIFACT_ROOT` | Defaults to `data/artifacts`; directories are created lazily on first write. |
| OpenAlex | — | `OPENALEX_EMAIL`, `OPENALEX_API_KEY` | Works anonymously with polite rate limits; failures degrade gracefully to other providers. |
| Semantic Scholar | — | `SEMANTIC_SCHOLAR_API_KEY` | Anonymous access is rate-limited; provider errors are logged sanitized and skipped. |
| arXiv | — | — | Fully public API; no key ever needed. |
| LLM synthesis | `LLM_PROVIDER` for non-mock use | `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `HTTP_TIMEOUT_SECONDS` | `LLM_PROVIDER=mock` works fully offline; remote providers fail gracefully to a safe degraded answer. |
| Local embeddings | — | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `disabled` by default; lexical retrieval still works. Model downloads lazily on first real use. Requires the `embeddings` extra. |
| Pinecone semantic index | — | `SEMANTIC_INDEX`, `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_NAMESPACE` | `disabled` by default; when enabled but unreachable, retrieval degrades to lexical-only. Never canonical data. Requires the `pinecone` extra. |
| Personal memory (Graphiti/Kuzu) | — | `USER_MEMORY_BACKEND`, `USER_MEMORY_DB_PATH`, `USER_MEMORY_GROUP_ID`, `USER_MEMORY_MAX_RESULTS`, `USER_MEMORY_CAPTURE` | `disabled` by default: no memory reads or writes, chat remains fully functional. See below before enabling. |
| Bounded live discovery | — | `CHAT_LIVE_DISCOVERY_LIMIT`, `INGESTION_METADATA_LIMIT` | Discovery runs only when stored evidence is insufficient; capped at 12 results per chat turn (default 10). |
| Watch scheduler | — | `WATCH_SCAN_HOURS`, `TIMEZONE` | Defaults to a scan every 6 hours in `Asia/Bangkok`. |
| Digest scheduler | — | `DIGEST_HOUR`, `TIMEZONE` | Defaults to 08:00 in `Asia/Bangkok`. |

## Discord Developer Portal setup

In the [Discord Developer Portal](https://discord.com/developers/applications),
for your application:

- Enable **Presence Intent**: not needed.
- Enable **Server Members Intent**: not needed.
- Enable **Message Content Intent**: NOT required. The privileged Message
  Content intent is deliberately not requested because Discord delivers message
  content for messages that @mention the app and for DMs sent to the app,
  which are exactly the surfaces this feature uses.
- Invite the bot with the `bot` scope and permissions to read/send messages in
  the target channels (`Send Messages`, `Read Message History`, `View Channels`).

## Personal memory backend

`USER_MEMORY_BACKEND` accepts only `disabled` or `graphiti`
(case-insensitive). The Graphiti backend uses an embedded Kuzu database under
`USER_MEMORY_DB_PATH`.

- Enabling it requires the optional dependency group:
  `pip install -e ".[memory]"`.
- If the package is missing or the backend fails, the store degrades to
  disabled behaviour with a sanitized log line; chat never breaks because of
  memory.
- Leaving `USER_MEMORY_BACKEND` unset or `disabled` keeps chat fully
  functional — personal memory is advisory context only, never scientific
  evidence.
