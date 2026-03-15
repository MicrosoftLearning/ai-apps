# Model Coder Implementation Details

This document describes the current implementation and gives a code-trace map for debugging and maintenance.

## 1. File Map

- `index.html`
  - Declares the app shell, toolbar controls, status pills, editor/terminal panes, splitter, and About modal.
  - Loads `coi-serviceworker.js`, `styles.css`, PyScript runtime, then `app.js`.
- `styles.css`
  - Contains responsive layout, light/dark theme variables, editor/terminal styling, splitter behavior, focus-visible styles, and modal styles.
- `app.js`
  - Main UI/runtime controller: template loading, run lifecycle, terminal runner creation, run-id sync, theme/modal behavior, and accessibility helpers.
- `llm.js`
  - Local model runtime wrapper over `wllama`: initialization, request translation to ChatML, streaming chunk queues, session reset/hard reset.
- `nopenai.py`
  - Python OpenAI-compatible wrapper used inside PyScript execution, including sync/async APIs and bridge invocation logic.
- `coi-serviceworker.js`
  - COI bootstrap service worker for static hosting scenarios.

## 2. Startup Flow

### 2.1 Bootstrap sequence

1. Browser loads `index.html`.
2. `coi-serviceworker.js` executes first.
3. PyScript assets load.
4. `app.js` imports `llm.js` and runs `initializeApp()`.

### 2.2 Bridge registration (`llm.js`)

`llm.js` creates one `ModelCoderLLM` instance and exposes bridge methods:

- `modelCoderSetStatusListener`
- `modelCoderInit`
- `modelCoderSetActiveRunId`
- `modelCoderRequest`
- `modelCoderResetSession`
- `modelCoderHardResetSession`
- `modelCoderNextStreamChunk`

These are attached to `globalThis`, `window`, and `self`, and also grouped under `modelCoderBridge`.

### 2.3 App initialization (`app.js`)

`initializeApp()`:

1. Sets initial runtime/model status text.
2. Registers model status listener.
3. Registers `window.modelCoderMarkRunComplete` callback.
4. Applies saved theme, initializes splitter, wires UI handlers.
5. Initializes About dialog interactions.
6. Loads `nopenai.py` source into memory (`state.nopenaiSource`).
7. Waits for PyScript ready (event + fallback polling) and marks runtime ready.
8. Initializes local model through bridge `modelCoderInit`.

## 3. UI and State (`app.js`)

## 3.1 Core state

Key state fields:

- Readiness: `pyReady`, `terminalReady`, `modelReady`
- Run/session: `running`, `sessionActive`, `activeRunId`
- Template switching safety: `switchingTemplate`
- Theme/UI: `darkTheme`, `savedCode`, `selectedTemplate`
- Modal focus restore: `aboutReturnFocus`

`updateRunState()` disables Run when switching templates or when run/session prerequisites are not satisfied.

### 3.2 Template switching behavior

`loadSelectedTemplate()` is transactional:

1. Sets `switchingTemplate = true` and disables Run.
2. If sample changed:
   - clears terminal output (`clearTerminalOutput({ resetModel: false })`)
   - awaits `requestModelSessionReset({ hard: true })`
3. Loads snippet into editor.
4. Clears switching flag and re-enables controls.

This is intended to start the next sample from fresh model state.

### 3.3 Run-id synchronization

`syncActiveRunId()` pushes `state.activeRunId` to JS bridge (`modelCoderSetActiveRunId`).

It is called when:

- app initializes
- a run starts
- stop is pressed
- terminal output is cleared

## 4. Editor and Terminal Integration

### 4.1 Editor mode

- PyScript editor is used for authoring only.
- Native embedded run controls are hidden and app-level Run controls execution.

### 4.2 Terminal runner lifecycle

`launchTerminalScript(scriptCode, runId)`:

1. Replaces `terminal-container` with a fresh node.
2. Creates a unique per-run terminal target element: `terminal-run-${runId}`.
3. Removes stale tagged runner scripts: `script[type="py"][data-model-coder-runner="true"]`.
4. Creates a new runner script with:
   - `type="py"`
   - `terminal`
   - `worker` if `shouldUseTerminalWorker()`
   - `target=<unique run target id>`
   - `config` packages
5. Tags runner with `data-model-coder-runner="true"`.
6. Hooks completion events (`py:done`, `py:error`, `error`) to `completeActiveRun(runId)`.

This unique-target strategy reduces cross-run terminal worker/proxy interference.

### 4.3 Runtime mode decision

`shouldUseTerminalWorker()`:

- GitHub Pages host detection (`github.io` or `*.github.io`) returns worker mode.
- Otherwise, returns `window.crossOriginIsolated`.

## 5. Python Execution Wrapper (`buildExecutionCode` in `app.js`)

Every Run wraps user code with prelude logic:

1. Creates in-memory module from fetched `nopenai.py` source.
2. Injects `_MODELCODER_RUN_ID` into that module.
3. Registers aliases:
   - `sys.modules["nopenai"]`
   - `sys.modules["openai"]`
4. Wraps `builtins.input` as `__modelcoder_input` to normalize async-returning terminal input values.
5. Executes user code with `exec(__user_code, globals())`.
6. In `finally`, signals JS completion callback with run id.

This wrapper applies to all user-authored code, not only built-in samples.

## 6. Run Lifecycle and Cleanup

### 6.1 Run start (`runCurrentCode`)

1. Sets `running = true`.
2. Loads `nopenai.py` source if needed.
3. Generates new run id (`activeRunId + 1`), syncs it to model bridge.
4. Sets `sessionActive = true`.
5. Launches terminal runner script.

### 6.2 Completion (`completeActiveRun`)

- Validates callback run id equals current active run id.
- Removes runner scripts.
- Clears run/session flags.
- Requests soft model session reset.
- Leaves terminal output visible.

### 6.3 Manual stop (`stopActiveRun`)

- Increments run id, syncs run id, removes runners.
- Writes stop note to terminal.
- Clears run/session flags and requests soft reset.

### 6.4 Terminal clear (`clearTerminalOutput`)

- Increments run id and syncs run id.
- Removes runners and replaces terminal container.
- Optionally triggers soft reset (default true).

## 7. Model Runtime (`llm.js`)

### 7.1 Initialization and stability settings

`_loadModel()` uses:

- model: `ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF`
- file: `smollm2-360m-instruct-q8_0.gguf`
- `n_ctx: 2048`
- `n_threads: 1`

`WASM_PATHS` currently maps both single-thread and multi-thread keys to single-thread wasm for stability.

### 7.2 Request handling

`_requestInternal(payload)` supports:

- `chat.completions.create`
- `responses.create`

Flow:

1. validates payload and model constraints
2. validates run token (`payload.run_id`) via `_ensureActiveRun(...)`
3. builds ChatML prompt
4. executes full response path or streaming path

### 7.3 Run token gating

- `setActiveRunId(runId)` stores current active run id.
- `_ensureActiveRun(runId, context)` rejects stale calls from older runs.
- stream sessions store `requestedRunId`; chunk retrieval drops sessions that do not match active run.

This prevents stale scripts from previous samples from issuing new model requests/chunks.

### 7.4 Streaming

- `_createStreamSession(...)` initializes a queue-backed stream session.
- `_complete(...)` pushes text deltas.
- `nextStreamChunk(streamId, runId)` drains queue with run-id and session-version checks.

### 7.5 Reset semantics

Soft reset (`resetSession`):

- increments `sessionVersion`
- clears stream/response maps
- waits briefly for active generations to unwind
- clears KV cache

Hard reset (`hardResetSession`):

- runs soft reset
- clears current model instance references
- attempts engine cleanup (`dispose`, `destroy`, `unload`, etc. when available)
- re-initializes model

## 8. Python Wrapper and Bridge (`nopenai.py`)

### 8.1 API surface

- `OpenAI` and `AsyncOpenAI`
- `chat.completions.create(...)`
- `responses.create(...)`
- sync and async stream iterators

### 8.2 Run-id propagation

- `_current_run_id()` reads `_MODELCODER_RUN_ID` from module globals.
- `_request(...)` injects `run_id` into every model request payload.
- `_next_chunk(...)` passes run id with stream chunk polling.

### 8.3 Bridge invocation strategy

`_bridge_call(method_name, *args)`:

1. Enumerates candidate bridge objects across `pyscript.window`, `js`, `pyscript.sync`, and `js` globals (`globalThis/window/self/parent/top`).
2. Prefers `modelCoderBridge.<method>` when available.
3. Falls back to direct method calls.
4. Falls back to `.call(...)` style invocation.
5. Handles both sync and awaitable returns.

## 9. COI Service Worker (`coi-serviceworker.js`)

- Registers service worker when available in secure context.
- On first controller activation, triggers reload.
- Intercepts fetch and adds COI headers:
  - `Cross-Origin-Embedder-Policy: require-corp`
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Resource-Policy: cross-origin`

## 10. Accessibility and Keyboard UX

Implemented across `index.html`, `styles.css`, and `app.js`:

- semantic labels/landmarks
- `aria-live` status pills
- skip link
- keyboard splitter controls
- visible focus styles
- Escape handling:
  - close About modal
  - move focus out of editor to splitter (`enableEditorEscapeToTabOut()`)

## 11. Debug Trace Recipe

Use this sequence when troubleshooting run hangs or stale-state behavior:

1. `initializeApp()` in `app.js` for startup and listeners.
2. `runCurrentCode()` in `app.js` for run-id assignment and runner launch.
3. `buildExecutionCode()` in `app.js` for Python wrapper/run-id stamping.
4. `_request(...)` and `_bridge_call(...)` in `nopenai.py` for payload + bridge path.
5. `modelCoderRequest` and `_requestInternal(...)` in `llm.js` for model request execution.
6. `_ensureActiveRun(...)`, `_createStreamSession(...)`, and `nextStreamChunk(...)` in `llm.js` for stale-run filtering.
7. `loadSelectedTemplate()` + `requestModelSessionReset({ hard: true })` in `app.js` for sample-switch reset flow.
8. `hardResetSession()` in `llm.js` for full model reinitialization path.
