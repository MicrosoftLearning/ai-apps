const runBtn = document.getElementById("run-btn");
const examplesBtn = document.getElementById("examples-btn");
const examplesDropdown = document.getElementById("examples-dropdown");
const savedIndicator = document.getElementById("saved-indicator");
const paneSplitter = document.getElementById("pane-splitter");
const workspace = document.querySelector(".workspace");
const topbarControls = document.querySelector(".topbar-controls");
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

// Wait briefly for the COI service worker to take control (which triggers a reload).
// Resolves to whether the page is now cross-origin isolated.
function waitForCrossOriginIsolation(timeoutMs = 5000) {
    if (window.crossOriginIsolated) {
        return Promise.resolve(true);
    }

    if (!window.isSecureContext || !("serviceWorker" in navigator)) {
        return Promise.resolve(false);
    }

    return new Promise((resolve) => {
        const finish = () => {
            cleanup();
            resolve(Boolean(window.crossOriginIsolated));
        };

        const onControllerChange = () => {
            // The COI service worker reloads the page on controllerchange, so this
            // resolution path is mostly defensive in case the reload is suppressed.
            finish();
        };

        const timer = setTimeout(finish, timeoutMs);

        function cleanup() {
            clearTimeout(timer);
            navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
        }

        navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
    });
}

const TEMPLATE_SNIPPETS = {
    "blank-page": "",
    "hello-world": String.raw`# Hello World example
print("Hello, World!")
print("Welcome to Python!")
`,
    "user-input": String.raw`# User input example
name = input("What is your name? ")
print(f"Hello, {name}!")
age = input("How old are you? ")
print(f"You are {age} years old.")
`,
    "math-and-types": String.raw`# Math operations and type conversion

# --- Basic arithmetic ---
a = 15
b = 4

print("Addition:      ", a + b)
print("Subtraction:   ", a - b)
print("Multiplication:", a * b)
print("Division:      ", a / b)   # always returns a float
print("Floor division:", a // b)  # rounds down to nearest integer
print("Remainder:     ", a % b)   # modulus (remainder after division)
print("Exponent:      ", a ** b)  # a raised to the power of b

# --- Working with floats ---
print()
price = 9.99
quantity = 3
total = price * quantity
print(f"Unit price:  ${"$"}{price}")
print(f"Quantity:    {quantity}")
print(f"Total:       ${"$"}{total}")

# --- Converting between types ---
print()
raw = "42"
as_int = int(raw)        # string  → integer
as_float = float(raw)    # string  → float
back_to_str = str(as_int)  # integer → string

print("Original string: ", raw,        type(raw))
print("As integer:      ", as_int,     type(as_int))
print("As float:        ", as_float,   type(as_float))
print("Back to string:  ", back_to_str, type(back_to_str))

# --- Rounding ---
print()
pi = 3.14159265
print("pi =", pi)
print("Rounded to 2 dp:  ", round(pi, 2))
print("Rounded to 0 dp:  ", round(pi))
print("int() truncates:  ", int(pi))   # drops everything after decimal point
`,
    "loops-and-decisions": String.raw`# Loops and decisions

# --- if / elif / else ---
score = 78
print("Score:", score)

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print("Grade:", grade)

# --- for loop with range() ---
print()
print("Numbers 1 to 10:")
for number in range(1, 11):
    print(number, end=" ")
print()

# --- for loop with if / else inside ---
print()
print("Odd and even (1 –20):")
for number in range(1, 21):
    if number % 2 == 0:
        print(f"{number:>2} is even")
    else:
        print(f"{number:>2} is odd")

# --- while loop with a condition ---
print()
print("Countdown:")
countdown = 5
while countdown > 0:
    print(countdown)
    countdown -= 1
print("Go!")
`,
    "numpy-demo": String.raw`# NumPy array operations
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Array:", arr)
print("Mean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Standard deviation:", np.std(arr))

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print()
print("Matrix:")
print(matrix)
print("Transpose:")
print(matrix.T)
`,
    "pandas-demo": String.raw`# Pandas data analysis
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [25, 30, 35, 28],
    "Score": [88, 92, 78, 95]
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print()
print("Descriptive statistics:")
print(df.describe())
print()
print("Average score:", df["Score"].mean())
`
};

const state = {
    pyReady: false,
    terminalReady: false,
    running: false,
    sessionActive: false,
    runtimeInitialized: false,
    savedCode: "",
    selectedTemplate: "",
    activeRunId: 0,
    switchingTemplate: false,
};

let terminalResizeRafId = 0;

function isAppReady() {
    return state.pyReady && state.terminalReady;
}

function updateRunState() {
    const appReady = isAppReady();

    if (examplesBtn) {
        examplesBtn.disabled = !appReady || state.switchingTemplate;
    }

    if (topbarControls) {
        topbarControls.classList.toggle("ui-locked", !appReady);
    }

    if (workspace) {
        workspace.inert = !appReady;
        workspace.classList.toggle("ui-locked", !appReady);
        workspace.setAttribute("aria-busy", appReady ? "false" : "true");
    }

    if (paneSplitter) {
        paneSplitter.setAttribute("aria-disabled", appReady ? "false" : "true");
        paneSplitter.tabIndex = appReady ? 0 : -1;
    }

    runBtn.disabled = !appReady || state.running || state.sessionActive || state.switchingTemplate;
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
    queueTerminalFontSize();
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
    updateRunState();
    requestTerminalResizeSync();
}

function clearTerminalOutput() {
    state.activeRunId += 1;
    syncActiveRunId();

    const staleRunners = document.querySelectorAll('script[type="py"][data-model-coder-runner="true"]');
    staleRunners.forEach((node) => node.remove());

    resetTerminalContainer();
    state.sessionActive = false;
    state.running = false;
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

async function loadSelectedTemplate(templateKey) {
    const selected = templateKey;
    if (!(selected in TEMPLATE_SNIPPETS)) {
        return;
    }
    const templateChanged = state.selectedTemplate !== selected;

    const snippet = TEMPLATE_SNIPPETS[selected];

    state.switchingTemplate = true;
    updateRunState();

    try {
        if (templateChanged) {
            clearTerminalOutput();
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

// Maps CM6 defaultHighlightStyle colors (designed for white backgrounds) to
// bright alternatives that are readable on the dark #1e1e1e editor background.
// CM6 stores colors in rgb() format in the injected CSSStyleSheet rules.
const HIGHLIGHT_COLOR_MAP = new Map([
    ["rgb(119, 0, 136)", "#c586c0"],    // #708  keyword          → bright purple
    ["rgb(170, 17, 17)", "#ce9178"],    // #a11  string/deleted   → salmon
    ["rgb(17, 102, 68)",  "#b5cea8"],   // #164  number/literal   → light green
    ["rgb(153, 68, 0)",   "#6a9955"],   // #940  comment          → muted green
    ["rgb(0, 136, 85)",   "#4ec9b0"],   // #085  typeName         → teal
    ["rgb(0, 0, 255)",    "#9cdcfe"],   // #00f  variableName     → light blue
    ["rgb(34, 17, 153)",  "#569cd6"],   // #219  bool/atom/url    → blue
    ["rgb(238, 68, 0)",   "#d7ba7d"],   // #e40  regexp/escape    → gold
    ["rgb(51, 0, 170)",   "#9cdcfe"],   // #30a  local variable   → light blue
    ["rgb(17, 102, 119)", "#4ec9b0"],   // #167  className        → teal
    ["rgb(34, 85, 102)",  "#9cdcfe"],   // #256  macroName        → light blue
    ["rgb(0, 0, 204)",    "#9cdcfe"],   // #00c  propertyName def → light blue
    ["rgb(51, 187, 51)",  "#b5cea8"],   // #3b3  function name    → light green
    ["rgb(255, 0, 0)",    "#f44747"],   // #f00  invalid          → bright red
    ["rgb(0, 0, 0)",      "#d4d4d4"],   // black operator/punct  → base text
]);

function patchEditorHighlightColors(shadowRoot) {
    // Normalize to handle browser serialization differences:
    // "rgb(170, 17, 17)" vs "rgb(170,17,17)" vs "rgb(170,  17,  17)"
    const normalize = (s) => s.replace(/\s+/g, "").toLowerCase();
    const normalizedMap = new Map(
        [...HIGHLIGHT_COLOR_MAP.entries()].map(([k, v]) => [normalize(k), v])
    );

    // Collect all sheets — shadowRoot.styleSheets may not always include
    // adoptedStyleSheets in all browsers, so check both explicitly.
    const sheets = new Set(shadowRoot.styleSheets);
    for (const sheet of (shadowRoot.adoptedStyleSheets ?? [])) {
        sheets.add(sheet);
    }

    const overrides = [];
    for (const sheet of sheets) {
        try {
            for (const rule of sheet.cssRules) {
                if (!(rule instanceof CSSStyleRule)) {
                    continue;
                }
                const newColor = normalizedMap.get(normalize(rule.style.color));
                if (newColor) {
                    overrides.push(`${rule.selectorText} { color: ${newColor} !important; }`);
                }
            }
        } catch (_e) {
            // Sheet may be cross-origin or otherwise inaccessible.
        }
    }

    if (overrides.length === 0) {
        return false;
    }

    const styleId = "model-coder-syntax-colors";
    let styleEl = shadowRoot.getElementById(styleId);
    if (!styleEl) {
        styleEl = document.createElement("style");
        styleEl.id = styleId;
        shadowRoot.appendChild(styleEl);
    }
    styleEl.textContent = overrides.join("\n");
    return true;
}

function applyDarkEditorTheme() {
    const editor = getEditor();
    if (!editor) {
        return;
    }

    editor.setAttribute("data-theme", "dark");

    const container = document.getElementById("editor-container");
    const styleId = "model-coder-dark-theme";
    const roots = [];

    if (editor.shadowRoot) {
        roots.push(editor.shadowRoot);
    }

    if (container) {
        for (const node of container.querySelectorAll("*")) {
            if (node.shadowRoot?.querySelector(".cm-editor")) {
                roots.push(node.shadowRoot);
            }
        }
    }

    for (const root of roots) {
        if (!root.getElementById(styleId)) {
            const styleEl = document.createElement("style");
            styleEl.id = styleId;
            styleEl.textContent = `
                .cm-editor, .cm-scroller, .cm-content, .cm-gutters {
                    background-color: #1e1e1e !important;
                    color: #d4d4d4 !important;
                    font-size: 16px !important;
                }
                .cm-content { caret-color: #aeafad !important; }
                .cm-gutters {
                    background-color: #252526 !important;
                    color: #919191 !important;
                    border-right: 1px solid #3c3c3c !important;
                }
                .cm-activeLine, .cm-activeLineGutter { background-color: #2a2d2e !important; }
                .cm-cursor, .cm-dropCursor { border-left-color: #aeafad !important; }
                .cm-selectionBackground,
                .cm-selectionLayer .cm-selectionBackground { background-color: rgba(38, 79, 120, 0.7) !important; }
                .cm-focused .cm-selectionBackground { background-color: #264f78 !important; }
                .cm-editor ::selection, .cm-editor *::selection {
                    background-color: #264f78 !important;
                    color: #d4d4d4 !important;
                }
                .cm-panels { background-color: #252526 !important; color: #d4d4d4 !important; }
                .cm-searchMatch { background-color: #613214 !important; outline: 1px solid #f38518 !important; }
                .cm-tooltip {
                    background-color: #252526 !important;
                    border: 1px solid #3c3c3c !important;
                    color: #d4d4d4 !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
                }
                .cm-tooltip-autocomplete > ul > li {
                    color: #d4d4d4 !important;
                    padding: 2px 8px !important;
                }
                .cm-tooltip-autocomplete > ul > li[aria-selected="true"],
                .cm-tooltip-autocomplete > ul > li:hover {
                    background-color: #264f78 !important;
                    color: #ffffff !important;
                }
                .cm-completionLabel { color: inherit !important; }
                .cm-completionDetail { color: #919191 !important; font-style: italic; }
                .cm-completionMatchedText {
                    color: #18a3ff !important;
                    text-decoration: none !important;
                    font-weight: 700 !important;
                }
            `;
            root.appendChild(styleEl);
        }
        // Re-run on every call so newly injected CM6 highlight rules are caught.
        if (patchEditorHighlightColors(root)) {
            return true;
        }
    }
    return false;
}

function queueDarkEditorThemeApply() {
    let attempts = 0;
    const timer = setInterval(() => {
        attempts += 1;
        applyDarkEditorTheme();
        if (attempts >= 20) {
            clearInterval(timer);
        }
    }, 150);
}

function applyTerminalFontSize(size = 15) {
    const terminal = getActiveTerminalInstance();
    if (!terminal) {
        return false;
    }
    try {
        terminal.options.fontSize = size;
        if (typeof terminal.refresh === "function") {
            terminal.refresh(0, Math.max(0, (terminal.rows || 24) - 1));
        }
        requestTerminalResizeSync();
    } catch (_err) {
        // Ignore — terminal may not support dynamic font resizing.
    }
    return true;
}

function queueTerminalFontSize() {
    let attempts = 0;
    const timer = setInterval(() => {
        attempts += 1;
        if (applyTerminalFontSize() || attempts >= 30) {
            clearInterval(timer);
        }
    }, 200);
}

function markRuntimeReady() {
    if (state.runtimeInitialized) {
        return;
    }

    state.runtimeInitialized = true;
    state.pyReady = true;
    state.terminalReady = true;
    setupEditorAsEditOnly();
    suppressNativeEditorRunButton();
    enableEditorEscapeToTabOut();
    queueDarkEditorThemeApply();
    initializeEditorEmpty();
    updateRunState();
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
    const serializedUserCode = JSON.stringify(String(userCode || ""));
    const serializedRunId = Number.isFinite(runId) ? runId : -1;
    return `
import sys
import builtins
import asyncio

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
print("Running script...")
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

async function initializeApp() {
    updateRunState();

    const coiReady = await waitForCrossOriginIsolation();
    if (!coiReady) {
        if (runBtn) {
            runBtn.disabled = true;
        }
        return;
    }

    window.modelCoderMarkRunComplete = (runId) => {
        const parsedRunId = Number(runId);
        completeActiveRun(Number.isFinite(parsedRunId) ? parsedRunId : state.activeRunId);
    };

    syncActiveRunId();

    initializePaneSplitter();
    window.addEventListener("resize", requestTerminalResizeSync);

    window.addEventListener("py:ready", markRuntimeReady, { once: true });

    runBtn.addEventListener("click", runCurrentCode);

    if (examplesBtn && examplesDropdown) {
        examplesBtn.addEventListener("click", () => {
            const isOpen = !examplesDropdown.hidden;
            examplesDropdown.hidden = isOpen;
            examplesBtn.setAttribute("aria-expanded", String(!isOpen));
        });

        examplesDropdown.addEventListener("click", (event) => {
            const item = event.target.closest("[data-template]");
            if (!item) {
                return;
            }
            examplesDropdown.hidden = true;
            examplesBtn.setAttribute("aria-expanded", "false");
            void loadSelectedTemplate(item.dataset.template);
        });

        document.addEventListener("click", (event) => {
            if (!examplesDropdown.hidden && !examplesBtn.contains(event.target) && !examplesDropdown.contains(event.target)) {
                examplesDropdown.hidden = true;
                examplesBtn.setAttribute("aria-expanded", "false");
            }
        });
    }

    const pyReadyFallback = setInterval(() => {
        const editor = getEditor();
        if (editor) {
            markRuntimeReady();
            clearInterval(pyReadyFallback);
        }
    }, 300);

    setTimeout(() => clearInterval(pyReadyFallback), 15000);
}

initializeApp().catch((error) => {
    console.error(error);
});

