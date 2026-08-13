# ResearchRadar Agent Rules

1. ResearchRadar is single-user.
2. Do not build SaaS infrastructure.
3. No user registration.
4. No RBAC.
5. No tenant isolation.
6. No Redis unless explicitly required later.
7. No Celery.
8. No Kubernetes.
9. No microservices for V1.
10. Prefer one Python process.
11. SQLite is the default database.
12. Discord handlers must stay thin.
13. Research/domain logic must not import Discord.
14. Scholarly APIs must sit behind provider abstractions.
15. Provider-specific JSON must never leak into domain models.
16. LLM inference must sit behind an `LLMProvider` abstraction.
17. Do not couple the application directly to Qwen, MLX, Hugging Face, OpenAI, or another model vendor.
18. Never commit secrets.
19. `.env` must be ignored.
20. `.env.example` contains placeholders only.
21. Prefer deterministic algorithms before LLM calls.
22. Every external HTTP request needs a timeout.
23. External API failures must fail gracefully.
24. Tests must mock external APIs.
25. Run lint and tests after every implementation phase.
26. Do not equate "not retrieved" with "does not exist".
27. Research claims must preserve provenance/evidence where possible.
28. Do not implement future features simply because directories exist for them.
29. Keep modules small and responsibilities explicit.
30. Avoid speculative abstractions that have no current use.

