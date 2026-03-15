import "./llm.js";

const statusRuntime = document.getElementById("runtime-status");
const statusModel = document.getElementById("model-status");
const runBtn = document.getElementById("run-btn");
const stopBtn = document.getElementById("stop-btn");
const retryBtn = document.getElementById("retry-btn");
const themeBtn = document.getElementById("theme-btn");
const aboutBtn = document.getElementById("about-btn");
const savedIndicator = document.getElementById("saved-indicator");
const templateSelect = document.getElementById("template-select");
const resetLayoutBtn = document.getElementById("reset-layout-btn");
const paneSplitter = document.getElementById("pane-splitter");
const workspace = document.querySelector(".workspace");
const aboutModalBackdrop = document.getElementById("about-modal-backdrop");
const aboutModal = document.getElementById("about-modal");
const aboutCloseBtn = document.getElementById("about-close-btn");
const THEME_STORAGE_KEY = "model-coder-theme";
const HOSTNAME = String(window.location.hostname || "").toLowerCase();
const IS_GITHUB_PAGES = HOSTNAME === "github.io" || HOSTNAME.endsWith(".github.io");

const PY_PACKAGES = ["numpy", "pandas", "matplotlib", "scikit-learn"];

function shouldUseTerminalWorker() {
    // Terminal-style input requires worker mode in PyScript.
    // Cross-origin isolation is provided on GitHub Pages via the COI service worker.
    if (IS_GITHUB_PAGES) {
        return true;
    }
    return Boolean(window.crossOriginIsolated);
}

const TEMPLATE_SNIPPETS = {
    "blank-page": "",
    "simple-chat-chatcompletions": String.raw`# import namespace
from openai import OpenAI


def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        openai_client = OpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Loop until the user wants to quit
        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Get a response
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that answers questions and provides information."
                    },
                    {
                        "role": "user",
                        "content": input_text
                    }
                ]
            )
            print(completion.choices[0].message.content)
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
`,
    "simple-chat-responses": String.raw`# import namespace
from openai import OpenAI


def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        openai_client = OpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Loop until the user wants to quit
        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Get a response
            response = openai_client.responses.create(
                        model=model_name,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text
            )
            print(response.output_text)
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
`,
    "conversation-tracking-chatcompletions": String.raw`# import namespace
from openai import OpenAI


def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        openai_client = OpenAI(
            base_url=endpoint,
            api_key=key
        )

        # Initial messages
        conversation_messages=[
                    {
                        "role": "developer",
                        "content": "You are a helpful AI assistant that answers questions and provides information."
                    }
        ]
        
        # Loop until the user wants to quit
        print("Enter a prompt (or type 'quit' to exit)")
        while True:
            input_text = input('You: ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt:")
                continue

            # Add the user message
            conversation_messages.append({"role": "user", "content": input_text})

            # Get a response
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=conversation_messages
            )
            assistant_text = completion.choices[0].message.content
            print("Assistant:", assistant_text)
            conversation_messages.append({"role": "assistant", "content": assistant_text})
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
`,
    "conversation-tracking-responses": String.raw`# import namespace
from openai import OpenAI


def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        openai_client = OpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Track responses
        last_response_id = None

        # Loop until the user wants to quit
        print("Enter a prompt (or type 'quit' to exit)")
        while True:
            input_text = input('You: ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt:")
                continue

            # Get a response
            response = openai_client.responses.create(
                        model=model_name,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id
            )
            assistant_text = response.output_text
            print("Assistant:", assistant_text)
            last_response_id = response.id
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
`,
    "streaming-responses": String.raw`# import namespace
from openai import OpenAI


def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        openai_client = OpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Track responses
        last_response_id = None
        print("Enter a prompt (or type 'quit' to exit)")
        while True:
            input_text = input('You: ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt:")
                continue

            # Get a response
            stream = openai_client.responses.create(
                        model=model_name,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id,
                        stream=True
            )
            print("Assistant:")
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="")
                elif event.type == "response.completed":
                    last_response_id = event.response.id
            print()
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    main()
`,
    "async-chat": String.raw`# import namespaces for async
import asyncio
from openai import AsyncOpenAI

async def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize an async OpenAI client
        async_client = AsyncOpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Track responses
        last_response_id = None

        # Loop until the user wants to quit
        print("Enter a prompt (or type 'quit' to exit)")
        while True:
            input_text = input('You: ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt:")
                continue

            # Await an asynchronous response
            response = await async_client.responses.create(
                        model=model_name,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id
            )
            assistant_text = response.output_text
            print("Assistant:", assistant_text)
            last_response_id = response.id
            

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    asyncio.run(main())
`,
    "async-stream": String.raw`# import namespaces for async
import asyncio
from openai import AsyncOpenAI


async def main(): 

    try:
        # Configuration settings 
        endpoint = "http://localwllama"
        key = "key123"
        model_name = "smollm2"

        # Initialize the OpenAI client
        async_client = AsyncOpenAI(
            base_url=endpoint,
            api_key=key
        )
        
        # Track responses
        last_response_id = None

        # Loop until the user wants to quit
        print("Enter a prompt (or type 'quit' to exit)")
        while True:
            input_text = input('You: ')
            if input_text.lower() == "quit":
                print("Goodbye!")
                break
            if len(input_text) == 0:
                print("Please enter a prompt:")
                continue

            # Await an asynchronous stream response
            stream = await async_client.responses.create(
                        model=model_name,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id,
                        stream=True
            )
            print("Assistant:")
            async for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="")
                elif event.type == "response.completed":
                    last_response_id = event.response.id
            print()

    except Exception as ex:
        print(ex)

if __name__ == '__main__': 
    asyncio.run(main())
`
};

const state = {
    pyReady: false,
    terminalReady: false,
    modelReady: false,
    running: false,
    sessionActive: false,
    runtimeInitialized: false,
    darkTheme: false,
    savedCode: "",
    nopenaiSource: "",
    selectedTemplate: templateSelect?.value || "",
    activeRunId: 0,
    aboutReturnFocus: null,
    switchingTemplate: false,
};

let modelResetInFlight = Promise.resolve();
let terminalResizeRafId = 0;

function openAboutModal() {
    if (!aboutModalBackdrop || !aboutModal) {
        return;
    }

    state.aboutReturnFocus = document.activeElement;
    aboutModalBackdrop.hidden = false;
    document.body.style.overflow = "hidden";
    aboutModal.focus();
}

function closeAboutModal() {
    if (!aboutModalBackdrop) {
        return;
    }

    aboutModalBackdrop.hidden = true;
    document.body.style.overflow = "";

    if (state.aboutReturnFocus && typeof state.aboutReturnFocus.focus === "function") {
        state.aboutReturnFocus.focus();
    } else if (aboutBtn) {
        aboutBtn.focus();
    }

    state.aboutReturnFocus = null;
}

function initializeAboutModal() {
    if (!aboutBtn || !aboutModalBackdrop || !aboutModal) {
        return;
    }

    aboutBtn.addEventListener("click", openAboutModal);

    if (aboutCloseBtn) {
        aboutCloseBtn.addEventListener("click", closeAboutModal);
    }

    aboutModalBackdrop.addEventListener("click", (event) => {
        if (event.target === aboutModalBackdrop) {
            closeAboutModal();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !aboutModalBackdrop.hidden) {
            event.preventDefault();
            closeAboutModal();
        }
    });
}

function applyTheme(isDark) {
    state.darkTheme = Boolean(isDark);
    document.body.classList.toggle("dark-theme", state.darkTheme);

    if (themeBtn) {
        themeBtn.setAttribute("aria-pressed", state.darkTheme ? "true" : "false");
        themeBtn.classList.toggle("active", state.darkTheme);
    }

    try {
        localStorage.setItem(THEME_STORAGE_KEY, state.darkTheme ? "dark" : "light");
    } catch (_error) {
        // Ignore storage failures and continue using in-memory theme state.
    }

    applyEmbeddedEditorTheme();
}

function toggleTheme() {
    applyTheme(!state.darkTheme);
}

function getSavedThemePreference() {
    try {
        return localStorage.getItem(THEME_STORAGE_KEY);
    } catch (_error) {
        return null;
    }
}

function applyEmbeddedEditorTheme() {
    const editor = getEditor();
    if (!editor) {
        return;
    }

    editor.setAttribute("data-theme", state.darkTheme ? "dark" : "light");
    editor.style.backgroundColor = "";
    editor.style.color = "";

    const container = document.getElementById("editor-container");
    const styleId = "model-coder-embedded-theme";
    const roots = [];

    if (editor?.shadowRoot) {
        roots.push(editor.shadowRoot);
    }

    if (container) {
        for (const node of container.querySelectorAll("*")) {
            if (!node.shadowRoot) {
                continue;
            }
            if (node.shadowRoot.querySelector(".cm-editor")) {
                roots.push(node.shadowRoot);
            }
        }
    }

    if (roots.length === 0) {
        return;
    }

    for (const root of roots) {
        let styleEl = root.getElementById(styleId);
        if (!styleEl) {
            styleEl = document.createElement("style");
            styleEl.id = styleId;
            root.appendChild(styleEl);
        }

        if (!state.darkTheme) {
            styleEl.textContent = "";
            continue;
        }

        // Third-party fix: One Dark-inspired theme from @codemirror/theme-one-dark.
        styleEl.textContent = `
        .cm-editor,
        .cm-scroller,
        .cm-content,
        .cm-gutters {
            background-color: #282c34 !important;
            color: #abb2bf !important;
        }

        .cm-content {
            caret-color: #528bff !important;
        }

        .cm-gutters {
            background-color: #282c34 !important;
            color: #7d8799 !important;
            border: none !important;
        }

        .cm-panels {
            background-color: #21252b !important;
            color: #abb2bf !important;
        }

        .cm-panels.cm-panels-top {
            border-bottom: 2px solid #000 !important;
        }

        .cm-panels.cm-panels-bottom {
            border-top: 2px solid #000 !important;
        }

        .cm-editor .cm-cursor,
        .cm-editor .cm-dropCursor {
            border-left-color: #528bff !important;
        }

        .cm-editor .cm-activeLine,
        .cm-editor .cm-activeLineGutter {
            background-color: #2c313a !important;
        }

        .cm-editor .cm-selectionMatch {
            background-color: #aafe661a !important;
        }

        /* Keep selection styling explicit and centralized to avoid override conflicts. */
        .cm-editor .cm-selectionLayer {
            z-index: 2 !important;
            mix-blend-mode: normal !important;
        }

        .cm-editor .cm-selectionLayer .cm-selectionBackground,
        .cm-editor .cm-selectionBackground {
            background-color: rgba(82, 139, 255, 0.42) !important;
        }

        .cm-editor.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground,
        .cm-editor.cm-focused .cm-selectionBackground {
            background-color: rgba(82, 139, 255, 0.62) !important;
            outline: 1px solid rgba(173, 204, 255, 0.7) !important;
        }

        .cm-editor ::selection,
        .cm-editor *::selection,
        .cm-editor .cm-content::selection,
        .cm-editor .cm-line::selection {
            background-color: rgba(82, 139, 255, 0.62) !important;
            color: #f0f6fc !important;
        }

        .cm-editor .cm-searchMatch {
            background-color: #72a1ff59 !important;
            outline: 1px solid #457dff !important;
        }
        `;
    }
}

function queueEmbeddedEditorThemeSync() {
    let attempts = 0;
    const maxAttempts = 20;
    const timer = setInterval(() => {
        attempts += 1;
        applyEmbeddedEditorTheme();
        if (getEditor()?.shadowRoot || attempts >= maxAttempts) {
            clearInterval(timer);
        }
    }, 150);
}

function setPill(el, text, mode = "") {
    el.textContent = text;
    el.classList.remove("ready", "error");
    if (mode) {
        el.classList.add(mode);
    }
}

function updateRunState() {
    runBtn.disabled = !state.pyReady || !state.terminalReady || !state.modelReady || state.running || state.sessionActive || state.switchingTemplate;
    if (stopBtn) {
        stopBtn.disabled = !(state.sessionActive || state.running);
    }
}

function getEditor() {
    return document.getElementById("python-editor");
}

function getTerminal() {
    return document.getElementById("python-terminal");
}

function getActiveTerminalInstance() {
    const container = document.getElementById("terminal-container");
    const candidates = [];

    if (container) {
        candidates.push(container, ...container.querySelectorAll("*"));
    }

    // PyScript stores terminal instance on the runner script element.
    const runnerScripts = document.querySelectorAll('script[type="py"][data-model-coder-runner="true"]');
    candidates.push(...runnerScripts);

    for (const node of candidates) {
        const terminal = node?.terminal;
        if (terminal && typeof terminal.resize === "function") {
            return terminal;
        }
    }

    return null;
}

function fitTerminalUsingAddon(terminal) {
    const addonEntries = terminal?._addonManager?._addons;
    if (!Array.isArray(addonEntries)) {
        return false;
    }

    for (const entry of addonEntries) {
        const addon = entry?.instance || entry?.addon || entry;
        if (addon && typeof addon.fit === "function") {
            addon.fit();
            return true;
        }
    }

    return false;
}

function fitTerminalByMeasurement(terminal, container) {
    if (!terminal || !container) {
        return false;
    }

    const viewport = container.querySelector(".xterm-viewport") || container.querySelector(".xterm-screen");
    const rowSample = container.querySelector(".xterm-rows > div");
    const rowHeight = rowSample?.getBoundingClientRect().height || 0;
    const viewportHeight = viewport?.getBoundingClientRect().height || 0;

    if (!(rowHeight > 0) || !(viewportHeight > 0)) {
        return false;
    }

    const nextRows = Math.max(2, Math.floor(viewportHeight / rowHeight));
    const nextCols = Math.max(2, Number(terminal.cols) || 80);

    if (typeof terminal.resize === "function") {
        terminal.resize(nextCols, nextRows);
    }

    if (typeof terminal.refresh === "function") {
        terminal.refresh(0, Math.max(0, nextRows - 1));
    }

    return true;
}

function syncTerminalToPaneSize() {
    const container = document.getElementById("terminal-container");
    const terminal = getActiveTerminalInstance();
    if (!container || !terminal) {
        return;
    }

    if (fitTerminalUsingAddon(terminal)) {
        return;
    }

    if (fitTerminalByMeasurement(terminal, container)) {
        return;
    }

    // Last resort: some integrations only react to window resize.
    window.dispatchEvent(new Event("resize"));
}

function requestTerminalResizeSync() {
    if (terminalResizeRafId) {
        cancelAnimationFrame(terminalResizeRafId);
    }

    terminalResizeRafId = requestAnimationFrame(() => {
        terminalResizeRafId = 0;
        syncTerminalToPaneSize();
    });
}

function queueTerminalResizeSync() {
    let attempts = 0;
    const maxAttempts = 20;
    const timer = setInterval(() => {
        attempts += 1;
        requestTerminalResizeSync();
        if (getActiveTerminalInstance() || attempts >= maxAttempts) {
            clearInterval(timer);
        }
    }, 120);
}

function setPaneSizes(editorPx, terminalPx) {
    if (!workspace) {
        return;
    }
    workspace.style.setProperty("--editor-size", `${Math.round(editorPx)}px`);
    workspace.style.setProperty("--terminal-size", `${Math.round(terminalPx)}px`);
    requestTerminalResizeSync();
}

function resetPaneSizes() {
    if (!workspace) {
        return;
    }

    const splitterSize = 12;
    const rect = workspace.getBoundingClientRect();
    const available = rect.height - splitterSize;
    if (available <= 0) {
        return;
    }

    const half = available / 2;
    setPaneSizes(half, half);
}

function initializePaneSplitter() {
    if (!paneSplitter || !workspace) {
        return;
    }

    const minPane = 160;
    const splitterSize = 12;
    let dragging = false;

    const clampEditorHeight = (value, available) => {
        const maxEditor = Math.max(minPane, available - minPane);
        return Math.min(Math.max(value, minPane), maxEditor);
    };

    const applyFromClientY = (clientY) => {
        const rect = workspace.getBoundingClientRect();
        const available = rect.height - splitterSize;
        if (available <= minPane * 2) {
            return;
        }

        const nextEditor = clampEditorHeight(clientY - rect.top, available);
        const nextTerminal = available - nextEditor;
        setPaneSizes(nextEditor, nextTerminal);
    };

    const onPointerMove = (event) => {
        if (!dragging) {
            return;
        }
        applyFromClientY(event.clientY);
    };

    const stopDrag = () => {
        if (!dragging) {
            return;
        }
        dragging = false;
        paneSplitter.classList.remove("dragging");
    };

    paneSplitter.addEventListener("pointerdown", (event) => {
        dragging = true;
        paneSplitter.classList.add("dragging");
        paneSplitter.setPointerCapture(event.pointerId);
        applyFromClientY(event.clientY);
    });

    paneSplitter.addEventListener("pointermove", onPointerMove);
    paneSplitter.addEventListener("pointerup", stopDrag);
    paneSplitter.addEventListener("pointercancel", stopDrag);

    paneSplitter.addEventListener("keydown", (event) => {
        if (!workspace) {
            return;
        }

        const rect = workspace.getBoundingClientRect();
        const available = rect.height - splitterSize;
        if (available <= minPane * 2) {
            return;
        }

        const styles = getComputedStyle(workspace);
        const currentEditor = parseFloat(styles.getPropertyValue("--editor-size")) || (available / 2);
        const delta = event.shiftKey ? 40 : 20;
        let nextEditor = currentEditor;

        if (event.key === "ArrowUp") {
            nextEditor -= delta;
        } else if (event.key === "ArrowDown") {
            nextEditor += delta;
        } else {
            return;
        }

        event.preventDefault();
        nextEditor = clampEditorHeight(nextEditor, available);
        setPaneSizes(nextEditor, available - nextEditor);
    });

    resetPaneSizes();
}

function resetTerminalContainer() {
    const current = document.getElementById("terminal-container");
    if (!current) {
        throw new Error("Terminal container not found.");
    }

    const replacement = current.cloneNode(false);
    replacement.innerHTML = "";
    current.replaceWith(replacement);
    requestTerminalResizeSync();
    return replacement;
}

function requestModelSessionReset(options = {}) {
    const { hard = false } = options;
    const resetFnName = hard ? "modelCoderHardResetSession" : "modelCoderResetSession";
    const resetFn = window[resetFnName];

    if (typeof resetFn !== "function") {
        if (hard && typeof window.modelCoderResetSession === "function") {
            return requestModelSessionReset();
        }
        return Promise.resolve();
    }

    modelResetInFlight = modelResetInFlight
        .catch(() => {
            // Keep reset chain alive even if a previous reset failed.
        })
        .then(() => resetFn());

    return modelResetInFlight.catch((error) => {
        console.warn("Unable to reset model session.", error);
    });
}

function syncActiveRunId() {
    if (typeof window.modelCoderSetActiveRunId !== "function") {
        return;
    }

    try {
        window.modelCoderSetActiveRunId(state.activeRunId);
    } catch (error) {
        console.warn("Unable to sync active run id.", error);
    }
}

function launchTerminalScript(scriptCode, runId) {
    const terminalContainer = resetTerminalContainer();
    const runTargetId = `terminal-run-${runId}`;
    const runTarget = document.createElement("div");
    runTarget.id = runTargetId;
    runTarget.style.height = "100%";
    runTarget.style.minHeight = "0";
    terminalContainer.appendChild(runTarget);

    // Remove any stale terminal scripts still bound to this target.
    const staleRunners = document.querySelectorAll('script[type="py"][data-model-coder-runner="true"]');
    staleRunners.forEach((node) => node.remove());

    const runner = document.createElement("script");
    runner.id = "python-terminal-runner";
    runner.type = "py";
    runner.dataset.modelCoderRunner = "true";
    runner.setAttribute("terminal", "");
    if (shouldUseTerminalWorker()) {
        runner.setAttribute("worker", "");
    }
    runner.setAttribute("target", runTargetId);
    runner.setAttribute("config", JSON.stringify({ packages: PY_PACKAGES }));
    runner.textContent = scriptCode;

    const markCompleted = () => {
        completeActiveRun(runId);
    };

    // PyScript emits lifecycle events when the script finishes or errors.
    runner.addEventListener("py:done", markCompleted, { once: true });
    runner.addEventListener("py:error", markCompleted, { once: true });
    runner.addEventListener("error", markCompleted, { once: true });

    document.body.appendChild(runner);
    queueTerminalResizeSync();
}

function stopActiveRun(message = "Run stopped. You can load another template or run code again.") {
    state.activeRunId += 1;
    syncActiveRunId();

    const staleRunners = document.querySelectorAll('script[type="py"][data-model-coder-runner="true"]');
    staleRunners.forEach((node) => node.remove());

    const terminalContainer = resetTerminalContainer();
    const note = document.createElement("pre");
    note.textContent = message;
    note.style.margin = "0";
    note.style.padding = "12px";
    note.style.whiteSpace = "pre-wrap";
    terminalContainer.appendChild(note);

    state.sessionActive = false;
    state.running = false;
    void requestModelSessionReset();
    updateRunState();
}

function completeActiveRun(runId = state.activeRunId) {
    if (runId !== state.activeRunId) {
        return;
    }

    // Keep runner element so its terminal instance remains available for resizing.
    // It will be removed by the next run, stop, template switch, or clear action.

    state.sessionActive = false;
    state.running = false;
    void requestModelSessionReset();
    updateRunState();
    requestTerminalResizeSync();
}

function clearTerminalOutput(options = {}) {
    const { resetModel = true } = options;
    state.activeRunId += 1;
    syncActiveRunId();

    const staleRunners = document.querySelectorAll('script[type="py"][data-model-coder-runner="true"]');
    staleRunners.forEach((node) => node.remove());

    resetTerminalContainer();
    state.sessionActive = false;
    state.running = false;
    if (resetModel) {
        void requestModelSessionReset();
    }
    updateRunState();
}

function hasTerminalRunner() {
    return document.querySelector('script[type="py"][data-model-coder-runner="true"]') !== null;
}

function setEditorCode(value) {
    const editor = getEditor();
    if (!editor) {
        return false;
    }

    const code = String(value ?? "");
    let applied = false;

    if ("code" in editor) {
        try {
            editor.code = code;
            applied = true;
        } catch (_err) {
            // Fall through to textContent path.
        }
    }

    try {
        editor.textContent = code;
        applied = true;
    } catch (_err) {
        // Ignore.
    }

    return applied;
}

function readEditorCode() {
    const editor = getEditor();
    if (!editor) {
        return "";
    }

    if ("code" in editor && typeof editor.code === "string") {
        return editor.code;
    }

    return String(editor.textContent || "");
}

async function loadSelectedTemplate() {
    if (!templateSelect) {
        return;
    }

    const selected = templateSelect.value;
    if (!(selected in TEMPLATE_SNIPPETS)) {
        return;
    }
    const templateChanged = state.selectedTemplate !== selected;

    const snippet = TEMPLATE_SNIPPETS[selected];

    state.switchingTemplate = true;
    updateRunState();

    try {
        if (templateChanged) {
            clearTerminalOutput({ resetModel: false });
            await requestModelSessionReset({ hard: true });
        }

        const ok = setEditorCode(snippet);
        state.selectedTemplate = selected;
        savedIndicator.textContent = ok
            ? `Loaded template: ${selected}`
            : "Unable to load template into editor.";
    } finally {
        state.switchingTemplate = false;
        updateRunState();
    }
}

function initializeEditorEmpty() {
    const existing = readEditorCode();
    if (existing.trim()) {
        return;
    }

    setEditorCode("");
    savedIndicator.textContent = "Editor empty. Pick a template or start typing.";
}

function markRuntimeReady() {
    if (state.runtimeInitialized) {
        return;
    }

    state.runtimeInitialized = true;
    state.pyReady = true;
    state.terminalReady = true;
    const runtimeMode = shouldUseTerminalWorker() ? "worker" : "main-thread";
    setPill(statusRuntime, `PyScript runtime ready (${runtimeMode})`, "ready");
    setupEditorAsEditOnly();
    suppressNativeEditorRunButton();
    enableEditorEscapeToTabOut();
    queueEmbeddedEditorThemeSync();
    initializeEditorEmpty();
    updateRunState();
}

async function loadNopenaiSource() {
    const response = await fetch("./nopenai.py", { cache: "no-store" });
    if (!response.ok) {
        throw new Error("Failed to load nopenai.py");
    }
    state.nopenaiSource = await response.text();
}

function setupEditorAsEditOnly() {
    const editor = getEditor();
    if (!editor || typeof editor.handleEvent === "undefined") {
        return;
    }

    // The editor remains available for editing; execution is routed through the terminal run button.
    editor.handleEvent = () => false;
}

function suppressNativeEditorRunButton() {
    const container = document.getElementById("editor-container");
    if (!container) {
        return;
    }

    const hideRunControls = () => {
        const candidates = container.querySelectorAll("button, [role='button']");
        for (const node of candidates) {
            if (node.dataset.modelCoderRunHidden === "true") {
                continue;
            }

            const title = String(node.getAttribute("title") || "").toLowerCase();
            const label = String(node.getAttribute("aria-label") || "").toLowerCase();
            const className = String(node.className || "").toLowerCase();
            const text = String(node.textContent || "").trim().toLowerCase();

            const looksLikeRun =
                title.includes("run") ||
                label.includes("run") ||
                className.includes("run") ||
                text === "run" ||
                text === "▶" ||
                text === "►";

            if (looksLikeRun) {
                node.style.display = "none";
                node.setAttribute("aria-hidden", "true");
                node.dataset.modelCoderRunHidden = "true";
            }
        }
    };

    hideRunControls();

    if (container.dataset.runButtonObserverAttached === "true") {
        return;
    }

    const observer = new MutationObserver(() => {
        hideRunControls();
    });
    observer.observe(container, { childList: true, subtree: true });
    container.dataset.runButtonObserverAttached = "true";
}

function enableEditorEscapeToTabOut() {
    if (document.body.dataset.editorEscapeFocusBound === "true") {
        return;
    }

    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") {
            return;
        }

        const editor = getEditor();
        if (!editor) {
            return;
        }

        const active = document.activeElement;
        const hostFocused = active === editor || editor.contains(active);
        const shadowFocused = Boolean(editor.shadowRoot?.activeElement);
        if (!hostFocused && !shadowFocused) {
            return;
        }

        event.preventDefault();
        if (paneSplitter) {
            paneSplitter.focus();
            return;
        }

        const terminalContainer = document.getElementById("terminal-container");
        terminalContainer?.focus();
    });

    document.body.dataset.editorEscapeFocusBound = "true";
}

function buildExecutionCode(userCode, runId) {
    const serializedNopenai = JSON.stringify(state.nopenaiSource || "");
    const serializedUserCode = JSON.stringify(String(userCode || ""));
    const serializedRunId = Number.isFinite(runId) ? runId : -1;
    return `
import sys
import types
import builtins
import asyncio

module = types.ModuleType("nopenai")
__nopenai_source = ${serializedNopenai}
exec(__nopenai_source, module.__dict__)
module.__dict__["_MODELCODER_RUN_ID"] = ${serializedRunId}
sys.modules["nopenai"] = module
sys.modules["openai"] = module

__original_input = builtins.input

def __modelcoder_input(*args, **kwargs):
    value = __original_input(*args, **kwargs)
    if not hasattr(value, "__await__"):
        if value is None:
            return ""
        return str(value)

    try:
        import pyodide.webloop as __webloop
        runner = getattr(__webloop, "run_until_complete", None)
        if callable(runner):
            return runner(value)
    except Exception:
        pass

    try:
        return asyncio.run(value)
    except Exception:
        pass

    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return loop.run_until_complete(value)
    except Exception:
        pass

    # If resolution fails, return original object and allow user code to handle it.
    if value is None:
        return ""
    return str(value)

builtins.input = __modelcoder_input

globals()["__name__"] = "__main__"
__user_code = ${serializedUserCode}
__run_id = ${serializedRunId}
print("Model Coder ready. Running script...")
try:
    exec(__user_code, globals())
finally:
    try:
        import js
        if hasattr(js, "modelCoderMarkRunComplete"):
            js.modelCoderMarkRunComplete(__run_id)
        elif hasattr(js, "window") and hasattr(js.window, "modelCoderMarkRunComplete"):
            js.window.modelCoderMarkRunComplete(__run_id)
    except Exception:
        pass
`;
}

async function runCurrentCode() {
    const editor = getEditor();
    if (!editor) {
        return;
    }

    state.running = true;
    updateRunState();

    try {
        await loadNopenaiSource();

        const code = readEditorCode();
        if (!code.trim()) {
            const runId = state.activeRunId + 1;
            state.activeRunId = runId;
            syncActiveRunId();
            state.sessionActive = true;
            launchTerminalScript("print('Editor is empty. Load a template or type code first.')", runId);
            return;
        }

        state.savedCode = code;
        savedIndicator.textContent = `Saved and ran at ${new Date().toLocaleTimeString()}`;

        const runId = state.activeRunId + 1;
        state.activeRunId = runId;
        syncActiveRunId();
        state.sessionActive = true;
        launchTerminalScript(buildExecutionCode(code, runId), runId);
    } catch (error) {
        const msg = JSON.stringify(`Execution failed: ${String(error.message || error)}`);
        const runId = state.activeRunId + 1;
        state.activeRunId = runId;
        syncActiveRunId();
        state.sessionActive = true;
        launchTerminalScript(`print(${msg})`, runId);
        state.sessionActive = false;
    } finally {
        state.running = false;
        updateRunState();
    }
}

async function initializeModel() {
    state.modelReady = false;
    updateRunState();
    retryBtn.hidden = true;

    try {
        await window.modelCoderInit(3);
        state.modelReady = true;
        setPill(statusModel, "Model ready: smollm2", "ready");
    } catch (error) {
        setPill(statusModel, `Model failed: ${error.message}`, "error");
        retryBtn.hidden = false;
    }

    updateRunState();
}

async function initializeApp() {
    setPill(statusRuntime, "PyScript loading...");
    setPill(statusModel, "Model initializing...");

    window.modelCoderSetStatusListener(({ kind, message }) => {
        if (kind === "ready") {
            setPill(statusModel, message, "ready");
            return;
        }
        if (kind === "error") {
            setPill(statusModel, message, "error");
            return;
        }
        setPill(statusModel, message);
    });

    window.modelCoderMarkRunComplete = (runId) => {
        const parsedRunId = Number(runId);
        completeActiveRun(Number.isFinite(parsedRunId) ? parsedRunId : state.activeRunId);
    };

    syncActiveRunId();

    const savedTheme = getSavedThemePreference();
    applyTheme(savedTheme === "dark");

    initializePaneSplitter();
    window.addEventListener("resize", requestTerminalResizeSync);

    window.addEventListener("py:ready", markRuntimeReady, { once: true });

    runBtn.addEventListener("click", runCurrentCode);
    if (stopBtn) {
        stopBtn.addEventListener("click", () => {
            stopActiveRun();
        });
    }
    retryBtn.addEventListener("click", initializeModel);
    if (templateSelect) {
        templateSelect.addEventListener("change", () => {
            void loadSelectedTemplate();
        });
    }
    if (resetLayoutBtn) {
        resetLayoutBtn.addEventListener("click", resetPaneSizes);
    }
    if (themeBtn) {
        themeBtn.addEventListener("click", toggleTheme);
    }

    initializeAboutModal();

    await loadNopenaiSource();
    const pyReadyFallback = setInterval(() => {
        const editor = getEditor();
        if (editor) {
            markRuntimeReady();
            clearInterval(pyReadyFallback);
        }
    }, 300);

    setTimeout(() => clearInterval(pyReadyFallback), 15000);

    await initializeModel();
}

initializeApp().catch((error) => {
    setPill(statusRuntime, `Startup failed: ${error.message}`, "error");
    console.error(error);
});

