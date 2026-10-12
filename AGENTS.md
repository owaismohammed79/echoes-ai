# Echoes: Architecture & Agent Guardrails

## Core Strategy

Echoes uses a server-side generative model (Gemma 4 via Gemini API/AI Studio) to stream hyper-local, cited audio narratives to a thin mobile client. It enforces factual grounding through code, not just prompt instructions.

## STRICT AGENT DIRECTIVES

### 1. Factual Grounding & Fact Objects (The MOAT)

- **Data Aggregation:** The backend assembles `FactObject`s (`id`, `text`, `source_url`) from Wikipedia GeoSearch and SerpApi (used ONLY for named parks). OpenStreetMap is explicitly excluded from the MVP.
- **Ranking over Filtering:** Use the infrastructure deny-list (metro, station, constituency) as a negative ranking signal to push irrelevant results down, rather than a hard boolean filter that might drop valid adjacent history.
- **Enforcement by Validator:** Gemma MUST output structured sentences citing a `[[fact_id]]`. A deterministic backend validator strips any sentence that lacks a valid, in-context citation before it reaches the TTS engine.
- **Deterministic Refusal:** If zero qualifying facts are found, the system MUST automatically skip the Local Story prompt and route to Notice Mode. Do not ask the LLM to decide if it should refuse. Log all refusals.

### 2. Notice Mode & Safety Restrictions

- **Sensory Invitations Only:** Notice Mode must only issue invitations based on time (e.g., "If you are somewhere safe to pause, notice how the light falls on the nearest surface").
- **No Assertions:** NEVER state what is physically in front of the user (e.g., "Look at the bench"). Weather and time do not dictate physical surroundings.
- **No Directions:** NEVER output walking directions, route guidance, or road crossings.
- **Standard Disclaimer:** The backend must always append a static audio file: "Stay on marked pedestrian paths."

### 3. iOS Audio & Mobile Constraints

- **Synchronous Unlock:** The client MUST create and unlock the `AudioContext` and `<audio>` element synchronously inside the "Start Walk" tap handler, BEFORE any asynchronous network calls.
- **Privacy First:** Request geolocation permissions only _after_ the user taps Start. Round coordinates to 2-3 decimal places. Do not store location history.
- **Audio Delivery:** Prefetch a rolling queue of 2–3 short segments cached via the Cache API/IndexedDB. Do not pre-generate 15-minute scripts.

### 4. Cost, Abuse & Fallbacks

- **SerpApi Scope:** SerpApi is used ONLY for **Parks** (`app/services/parks.py`, the single `Parks nearby` query) and **Local News** (`app/services/news.py`, one query). Rivers, temples, statues and every other historical/geographical landmark stay with Wikipedia GeoSearch, which already handles them reliably. Never add a SerpApi query for them: one query per source per walk keeps the credits and the 3.0s latency budget intact.
- **Rate Limits:** Implement per-IP walk limits and a global daily spend cap for ElevenLabs and the LLM API.
- **Kill-Switch:** If budgets are exceeded, seamlessly fall back to a clearly labeled recorded demo mode.
- **TTS Fallback:** If ElevenLabs fails or rate-limits, fall back to the browser's native `speechSynthesis` API.

### 5. Core Contstraints to be followed all the time

- NEVER read .env files wherever present, always refer to the .env.example present.
- If any updations which require human intervention are needed, ask the user for it and then proceed.
- NEVER EVER perform non reversible and destructive commands which might cause huge loss to the project directly or indirectly
