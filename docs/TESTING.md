# ResearchRadar Testing & Verification Guide

This guide details the complete local testing, smoke-testing, and Discord slash-command verification workflow for ResearchRadar.

---

## 1. Local Environment Setup & Execution

### 1.1 Create Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1.2 Initialize Local SQLite Database & Seed Demo Corpus
```bash
# Seed deterministic MRI Robustness project, PaperCards, 5 gap types, and Critic reviews:
python scripts/seed_demo_research_memory.py --db-url sqlite:///data/research_radar.db
```

### 1.3 Run Unit Tests & Linter
```bash
# Run pytest test suite:
pytest

# Run ruff code quality checks:
ruff check .
```

### 1.4 Run Local Smoke Test
```bash
# Run deterministic end-to-end smoke tests (no external network or LLM required):
python scripts/smoke_test_research_radar.py --db-url sqlite:///data/research_radar.db
```

---

## 2. Discord Bot Configuration & Launch

### 2.1 Configure Environment Variables
Copy `.env.example` to `.env` (ensure `.env` remains git-ignored):
```bash
cp .env.example .env
```

Set the required environment variables in `.env`:
```ini
DISCORD_BOT_TOKEN="your_discord_bot_token_here"
DISCORD_GUILD_ID="optional_guild_id_for_instant_slash_sync"
DATABASE_URL="sqlite:///data/research_radar.db"

# Optional Remote OpenAI-Compatible LLM for /ask and /read synthesis:
LLM_PROVIDER="remote"
LLM_BASE_URL="https://api.openai.com/v1"
LLM_API_KEY="your_api_key_here"
LLM_MODEL="gpt-4o-mini"
```

### 2.2 Start the Discord Bot
```bash
python -m research_radar.bot
```

---

## 3. Manual Discord Slash Command Verification

Run the following slash commands in your test Discord server to verify end-to-end functionality:

### 3.1 Health Check
```
/ping
```
*Expected Result*: Returns `Pong! Latency: XXms`.

---

### 3.2 Project Memory Inspection
```
/project-list
```
*Expected Result*: Lists stored projects including `MRI Robustness`.

```
/project-show project:MRI Robustness
```
*Expected Result*: Renders an embed showing:
- Project Goal: `study robustness of MRI reconstruction under scanner/domain shift`
- Hypotheses & Constraints
- Rejected Ideas: `Pure GAN reconstruction due to hallucinated lesions and training instability`
- Linked Papers (`seed`, `supporting`, `relevant`, `conflicting`, `background`)
- Linked Gaps (`active`, `interesting`)

---

### 3.3 Project-Aware Question Answering (/ask)

#### Query 1: Unresolved Robustness Issues
```
/ask question:What robustness issues are unresolved? project:MRI Robustness
```
*Expected Behavior*:
- Retrieves relevant project papers (`p-spectral-mri`, `p-domain-adapt`, `p-spectral-degrade`).
- Embeds project constraints and hypotheses.
- Mentions unresolved questions without making forbidden global literature claims (e.g. framing as `"Within the papers currently stored in ResearchRadar..."`).
- Does NOT recommend the rejected idea (`Pure GAN`) as a new solution.
- Citations contain only valid allowed paper IDs.

#### Query 2: Method Transfer Feasibility
```
/ask question:Which methods may transfer to scanner-shift robustness? project:MRI Robustness
```
*Expected Behavior*:
- Highlights `Patch-Based Wavelet Normalization` transferred from CT domain to MRI scanner shift robustness.
- Lists structured tasks (`reconstruction [observed]`) and modalities (`MRI [observed]`, `CT [observed]`).

---

### 3.4 Gap Engine Slash Commands

Run all 5 supported gap types for topic `"MRI reconstruction robustness"`:

#### 1. Explicit Gap Mining
```
/gap topic:MRI reconstruction robustness type:explicit
```
*Expected Result*: Analyzes corpus, retrieves candidate explicit gaps (e.g., real-time multi-coil diffusion under scanner shift) with latest Critic reviews (`preserved`).

#### 2. Coverage Gap Mining
```
/gap topic:MRI reconstruction robustness type:coverage
```
*Expected Result*: Surfaces an evidence-bounded coverage gap from the demo corpus (e.g., under-representation of low-field 0.55T scanners) without claiming global absence.

#### 3. Evaluation Gap Mining
```
/gap topic:MRI reconstruction robustness type:evaluation
```
*Expected Result*: Identifies un-evaluated condition pairs (e.g., concurrent motion artifacts and B0 frequency drift).

#### 4. Contradiction Gap Mining
```
/gap topic:MRI reconstruction robustness type:contradiction
```
*Expected Result*: Surfaces contradictory claims between `p-spectral-mri` (error reduction) and `p-spectral-degrade` (detail loss at 7T low SNR).

#### 5. Method Transfer Gap Mining
```
/gap topic:MRI reconstruction robustness type:method_transfer
```
*Expected Result*: Formulates transfer hypothesis from CT wavelet normalization to MRI scanner-shift robustness with attributable evidence fields.

---

## 4. Key Guardrails & Behavior Verification Checklist

- [x] **Project Scoping**: Project-linked papers receive deterministic boosts only when lexically relevant; unrelated project papers never crowd out search slots.
- [x] **Source-ID Safety**: LLM citations are strictly validated against `AskContext.allowed_paper_ids` and `allowed_gap_ids`. Hallucinated IDs are discarded.
- [x] **Rejected Ideas Safe Guarding**: Project rejected ideas are explicitly identified as project history and never recommended as novel suggestions.
- [x] **Status Integrity**: Resolved or rejected candidate gaps are explicitly marked as past status and not presented as active open gaps.
- [x] **Safe Error Handling**: Internal exceptions are logged and replaced with user-friendly messages rather than exposing raw tracebacks.
- [x] **Language Safety**: Forbidden global novelty claims (*"No one has studied"*, *"This is the first"*) are systematically rewritten or prevented.
- [x] **Five Gap Types Supported**:
  - `explicit`
  - `coverage`
  - `evaluation`
  - `contradiction`
  - `method_transfer`

---

## 5. Ready for Discord Test Gate

READY FOR DISCORD TEST IF:
- [x] pytest passes (`pytest`)
- [x] ruff passes (`ruff check .`)
- [x] demo seed runs twice (`python scripts/seed_demo_research_memory.py`)
- [x] smoke test passes with 8/8 scenarios (`python scripts/smoke_test_research_radar.py`)
- [x] all 5 gap types verified (`explicit`, `coverage`, `evaluation`, `contradiction`, `method_transfer`)
- [x] GitHub Actions CI green
