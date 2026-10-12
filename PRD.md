# Echoes Product Requirements Document

## 1. Product Overview

Echoes is a voice-first walking companion powered by open-weight models. It streams factually verified local history or pivots to sensory mindfulness when no facts exist, requiring zero screen time after initiation.

## 2. Infrastructure & Stack

- **Frontend:** Next.js or Vite React PWA (Thin Client) hosted on Render.
- **Backend:** Python FastAPI hosted on Render.
- **LLM Engine:** Gemma 4 (Generative Model) hosted via Gemini API/AI Studio (or equivalent inference provider).
- **Voice Pipeline:** ElevenLabs (TTS streaming) + Groq Whisper (STT for Push-To-Talk).

## 3. Playback & Interaction Sequencing

1. **Initial Playback:** The app fetches local facts, validates sentences, and streams 2-3 short audio segments to the client queue.
2. **Push-to-Talk (PTT) Q&A:** Once playback is stable, enable a push-to-talk loop where users can ask questions about the audio they just heard, leveraging the existing STT/TTS integration.
3. **Barge-in:** True continuous barge-in is deferred post-MVP to reduce demo risk.

## 4. The DEV Submission Angle

The submission must center on one falsifiable claim: **Near a real landmark, Echoes narrate only cited facts; where it has none, it explicitly declines to invent a story.**

- **Proof:** Include a before-and-after refusal trace in the post.
- **Open-Source Value:** Emphasize that server-side open weights allow for inspectable prompts, strict code-level guardrails, swappable models, and a clear self-host path without forcing users to download gigabytes of data over cellular networks.
