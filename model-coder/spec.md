# Model Coder (As-Built Specification)

## Changelog

- 2026-03-14: Replaced original aspirational draft with as-built specification.
- 2026-03-14: Documented current GitHub Pages/static-host runtime approach, including COI service worker integration.
- 2026-03-14: Added implemented UX/accessibility features (theme toggle persistence, splitter, About modal, keyboard/focus behaviors).
- 2026-03-14: Captured actual run/session lifecycle behavior and OpenAI-compatible wrapper support as implemented in code.

## Overview

Model Coder is a browser-based educational sandbox for learning OpenAI-style Python coding patterns against a local LLM.

Users write Python code in a PyScript editor and execute it in a terminal pane. The app provides a local `openai`-compatible wrapper so students can practice common syntax for:

- `OpenAI().chat.completions.create(...)`
- `OpenAI().responses.create(...)`
- Streaming responses
- Async usage with `AsyncOpenAI`

The model runtime is local and in-browser via `wllama` with SmolLM2.

## Core Goals

- Run fully in-browser on static hosting (including GitHub Pages).
- Provide an OpenAI-like Python authoring experience without external API calls.
- Support interactive terminal workflows (`print`, `input`, loop-based chat samples).
- Provide beginner-friendly sample templates that can be edited and rerun quickly.

## Architecture

The implementation is split across these files:

- `index.html`: app shell, toolbar, editor/terminal containers, accessibility landmarks, About modal.
- `styles.css`: responsive layout, light/dark themes, terminal/editor styling, focus-visible and a11y styles, modal styles.
- `app.js`: UI controller, templates, run lifecycle, session cleanup, theme/splitter/accessibility behavior.
- `llm.js`: local model lifecycle and request handling through `wllama`.
- `nopenai.py`: Python-side OpenAI-compatible wrapper and stream abstractions.
- `coi-serviceworker.js`: COOP/COEP service-worker bootstrap for cross-origin isolation support on static hosting.

## Hosting and Runtime Model

- Designed for static hosting and GitHub Pages.
- Includes `coi-serviceworker.js` bootstrap in `index.html` to support cross-origin isolation patterns needed by advanced browser runtime features.
- Terminal execution uses PyScript script runners targeted at `terminal-container`.
- Runtime mode selection is managed in `app.js` by `shouldUseTerminalWorker()`.

## UI and Interaction Requirements

### Layout

- Top toolbar with:
  - sample picker
  - Run, Stop, Reset Layout, Theme toggle, About button, Retry Model button
- Status pills for runtime/model state
- Two-pane workspace:
  - editor pane (top)
  - terminal pane (bottom)
  - draggable splitter with keyboard support

### Theming

- Light and dark themes
- Theme preference persisted in `localStorage`
- Editor and terminal themed consistently with app shell

### Accessibility

- Labeled toolbar controls and landmarks
- `aria-live` status updates
- keyboard-resizable splitter
- skip link to workspace
- focus-visible styling for keyboard navigation
- About dialog with proper dialog semantics and keyboard close behavior

## Python API Compatibility (Implemented)

The wrapper exposed via `nopenai.py` supports:

- `OpenAI(base_url="http://localwllama", api_key="...")`
- `AsyncOpenAI(base_url="http://localwllama", api_key="...")`
- Chat Completions:
  - `chat.completions.create(model="smollm2", messages=[...], stream=False|True)`
- Responses API:
  - `responses.create(model="smollm2", input=..., instructions=..., previous_response_id=..., stream=False|True)`
- Synchronous and asynchronous stream iterators

Validation behavior:

- `base_url` must be `http://localwllama`
- `model` must be `smollm2`
- message role/content types are validated

## Local Model Runtime

- Uses `@wllama/wllama` from CDN.
- Loads model `ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF` (`smollm2-360m-instruct-q8_0.gguf`).
- Converts request message structures to ChatML prompt format.
- Supports full and streaming completion paths.
- Maintains response/session maps for continuation semantics and stream retrieval.

## Run Lifecycle Behavior

- Editor is used for authoring only; execution is routed through terminal run button.
- Running code injects `nopenai.py` source and aliases it as both `nopenai` and `openai`.
- Run sessions are tracked with run IDs to prevent stale async completion events from corrupting UI state.
- Stop and completion both reset active session state while preserving expected terminal behavior.
- Switching to a different sample clears terminal output and resets model session context.

## Built-in Samples

Current sample set:

- Blank Page
- Simple chat (ChatCompletions API)
- Simple chat (Responses API)
- Conversation Tracking (ChatCompletions API)
- Conversation Tracking (Responses API)
- Streaming (Responses API)
- Async chat (Responses API)
- Async streaming (Responses API)

## Privacy and Data Flow

- The app is designed for local, in-browser execution.
- User code, prompts, and outputs are processed locally in the browser runtime.
- No server-side inference pipeline is used by this app.

## Notes for Contributors

- This document is intentionally as-built, not aspirational.
- For implementation walkthrough and function-by-function flow, see `implementation_details.md`.
