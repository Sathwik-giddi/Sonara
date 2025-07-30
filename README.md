# Sonara

**"Hey Sonara."** A sonara is a cloud — and this assistant's brain lives in a router over free cloud tiers (including NVIDIA NIM, pun intended), with local models for speech and offline mode.

Windows-first, router-first personal voice assistant. Runs at $0 forever. Formerly working-titled Sonara.

**The blueprint:** `C:\Users\vaish\.gstack\projects\PA\vaish-unknown-design-20260730-200543.md` (approved 2026-07-30). This repo implements it milestone by milestone. Current milestone: **M0 → M1** (the pipeline speaks).

## Tonight's checklist (the assignment)

1. **Create free API keys yourself** (no credit card needed on any of them):
   - Groq: https://console.groq.com → API Keys
   - Google AI Studio: https://aistudio.google.com/apikey
   - NVIDIA: https://build.nvidia.com → nvapi- key (one-time trial credits)
   - OpenRouter: https://openrouter.ai/keys
2. `./setup.ps1` — installs the Python env via uv, downloads the Piper TTS voice, creates `.env`
3. Paste `GROQ_API_KEY` into `.env` (only key the smoke test needs)
4. `uv run scripts/smoke_test.py` — speak one sentence, get every stage timed

The number that matters: **end-of-speech → first audio**. M1 gate: p50 ≤ 2.0s. Final target: 1.5s.

## Layout

- `scripts/smoke_test.py` — M0 staged latency test (record → faster-whisper STT → Groq LLM → Piper TTS; Piper won the measured bake-off vs Kokoro on this CPU)
- `config/caps.yaml` — dated free-tier caps config, seed of the M2 quota ledger
- `setup.ps1` — one-time environment setup

## Milestones (from the design doc)

- **M1** — the pipeline speaks: Pipecat (or plan-B sounddevice loop), push-to-talk, latency logger. Gate: p50 ≤ 2.0s over 50 exchanges, zero self-interruptions over 20 playback exchanges.
- **M2** — the heart: LiteLLM router + quota ledger + burn table + degradation modes.
- **M3** — hands: MCP tool layer, four action packs (M3a: PC control + notes; M3b: web + media), safety gates.
- **M4** — presence: wake word, memory (FTS5), tray app + autostart.
- **M5** — ship-shape then senses: latency push to 1.5s, integration day, then vision + open-source prep.
