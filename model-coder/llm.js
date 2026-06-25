import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/index.js";

const WASM_PATHS = {
    default: "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/wasm/wllama.wasm"
};

const MODEL_REPO = "bartowski/Phi-3.5-mini-instruct-GGUF";
const MODEL_QUANT = "Q4_K_M";
const MODERATION_LIST_PATH = "./moderation/mod.txt";
const MODERATION_SAFE_RESPONSE = "I'm sorry. I can't help with that. Either your system instructions or user input included content that was flagged by the moderation system. If you think this was a mistake, please try rephrasing your input or instructions and try again.";
const WIKIPEDIA_MODEL_NAME = "Wikipedia API (Basic Chat)";

const STOPWORDS = new Set([
    // Articles, prepositions, conjunctions
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'with',
    'or', 'but', 'if', 'than', 'then', 'so', 'yet',
    'after', 'before', 'between', 'during', 'into', 'through', 'over',
    'under', 'until', 'up', 'down', 'out', 'off', 'above', 'below',
    // Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'i\'m',
    'you\'re', 'he\'s', 'she\'s', 'we\'re', 'they\'re',
    // Determiners and quantifiers
    'this', 'these', 'those', 'some', 'any', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'other', 'another', 'much', 'many',
    // Verbs (auxiliary, modal, and common generic)
    'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
    'may', 'might', 'must', 'shall', 'ought', 'will',
    'be', 'get', 'make', 'know', 'see', 'take', 'come', 'go', 'want',
    'use', 'find', 'need', 'try', 'ask', 'work', 'help', 'like', 'seem',
    'become', 'let', 'tell', 'show', 'give', 'provide', 'explain',
    'describe', 'define',
    // Question words
    'what', 'when', 'where', 'who', 'how', 'why', 'which', 'whom',
    'whose', 'whether', 'what\'s', 'whats', 'who\'s', 'whos', 'how\'s',
    'hows',
    // Common adverbs
    'also', 'just', 'now', 'here', 'there', 'then', 'very', 'too',
    'really', 'still', 'always', 'never', 'often', 'sometimes', 'maybe',
    'perhaps', 'about',
    // Other common words
    'yes', 'no', 'thing', 'something', 'anything', 'nothing',
    'everything', 'someone', 'anyone', 'everyone', 'understand', 'know',
    'think', 'believe', 'feel', 'appear',
]);

function makeId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function reverseWord(text) {
    return text.split("").reverse().join("");
}

function shiftWord(text, amount) {
    return text
        .split("")
        .map((char) => String.fromCharCode(char.charCodeAt(0) + amount))
        .join("");
}

function roleToChatML(role) {
    if (role === "developer" || role === "system") {
        return "system";
    }
    if (role === "assistant") {
        return "assistant";
    }
    return "user";
}

function isTextContentBlock(block) {
    if (!block || typeof block !== "object") {
        return false;
    }

    return ["input_text", "output_text", "text"].includes(String(block.type || ""));
}

function contentToText(content) {
    if (typeof content === "string") {
        return content;
    }

    if (Array.isArray(content)) {
        return content
            .map((block) => {
                if (typeof block === "string") {
                    return block;
                }

                if (isTextContentBlock(block)) {
                    return String(block.text ?? "");
                }

                return "";
            })
            .filter((text) => text.length > 0)
            .join("\n");
    }

    if (isTextContentBlock(content)) {
        return String(content.text ?? "");
    }

    return String(content ?? "");
}

function extractLeadingSentences(text, maxSentences = 2) {
    if (!text) {
        return "";
    }

    let sentenceCount = 0;
    let cutIndex = -1;

    for (let i = 0; i < text.length; i += 1) {
        const char = text[i];
        if (char !== "." && char !== "!" && char !== "?") {
            continue;
        }

        const prev = i > 0 ? text[i - 1] : "";
        const next = i < text.length - 1 ? text[i + 1] : "";

        // Ignore decimal separators such as 12.5.
        if (char === "." && /\d/.test(prev) && /\d/.test(next)) {
            continue;
        }

        sentenceCount += 1;
        cutIndex = i + 1;
        if (sentenceCount >= maxSentences) {
            break;
        }
    }

    if (cutIndex === -1) {
        return text.trim();
    }

    return text.slice(0, cutIndex).trim();
}

function extractKeywords(text, excludedWords = null) {
    const words = String(text || "")
        .toLowerCase()
        .replace(/[^\w\s]/g, "")
        .split(/\s+/);

    return words
        .filter((word) => !STOPWORDS.has(word) && word.length > 0 && !(excludedWords && excludedWords.has(word)))
        .join(" ");
}

function chunkTextForStreaming(text, minChunk = 12, maxChunk = 36) {
    const source = String(text || "");
    if (!source) {
        return [];
    }

    const chunks = [];
    let index = 0;
    while (index < source.length) {
        const remaining = source.length - index;
        const target = Math.min(
            remaining,
            Math.max(minChunk, Math.floor(Math.random() * (maxChunk - minChunk + 1)) + minChunk)
        );

        let nextIndex = index + target;

        // Prefer splitting on whitespace/punctuation boundaries for natural deltas.
        if (nextIndex < source.length) {
            const boundaryWindow = source.slice(index, Math.min(source.length, nextIndex + 10));
            const boundaryOffset = boundaryWindow.search(/[\s,.!?;:)]/);
            if (boundaryOffset > 0) {
                nextIndex = index + boundaryOffset + 1;
            }
        }

        chunks.push(source.slice(index, nextIndex));
        index = nextIndex;
    }

    return chunks.filter((chunk) => chunk.length > 0);
}

function validateMessageContent(content, label) {
    if (typeof content === "string") {
        return;
    }

    if (!Array.isArray(content)) {
        throw new Error(`${label} content must be a string or an array of content blocks.`);
    }

    for (const block of content) {
        if (typeof block === "string") {
            continue;
        }

        if (!block || typeof block !== "object") {
            throw new Error(`${label} content blocks must be strings or objects.`);
        }

        if (!("type" in block)) {
            throw new Error(`${label} content block objects must include a type.`);
        }

        if (isTextContentBlock(block) && typeof block.text !== "string") {
            throw new Error(`${label} text content blocks must include a text string.`);
        }
    }
}

function validateMessages(messages, label = "messages") {
    if (!Array.isArray(messages)) {
        throw new Error(`${label} must be an array.`);
    }

    const allowedRoles = new Set(["developer", "system", "user", "assistant"]);
    for (const message of messages) {
        if (!message || typeof message !== "object") {
            throw new Error(`${label} must contain objects with role and content.`);
        }
        if (!allowedRoles.has(message.role)) {
            throw new Error("Message role must be developer, user, assistant, or system.");
        }
        validateMessageContent(message.content, label);
    }
}

class ModelCoderLLM {
    constructor() {
        this.wllama = null;  // wllama engine for Phi 3.5-mini
        this.usingWllama = false;  // Track if wllama is active
        this.usingBasic = false;  // Basic Chat Wikipedia mode
        this.availableModes = { cpu: true, basic: true };
        this.isReady = false;
        this.isLoading = false;
        this.statusCallback = null;
        this.streamSessions = new Map();
        this.responsesById = new Map();
        this.sessionVersion = 0;
        this.activeGenerationTasks = new Set();
        this.activeRunId = 0;
        this.moderationTerms = null;
        this.moderationLoadPromise = null;
        this.modelLoadingCancelled = false;
        this.modelLoadingAbortController = null;
        this.initSessionId = 0;  // Track which init session we're in
        this.gpuFailed = false;     // True after a GPU inference failure; forces CPU-only on reload
        this.wllamaUsedGPU = false; // True when the loaded model is using GPU acceleration
    }

    checkHardwareRequirements() {
        const MIN_MEMORY_GB = 8;
        const MIN_CORES = 8;
        const deviceMemory = navigator.deviceMemory || 0;
        const cores = navigator.hardwareConcurrency || 0;

        console.log(`Hardware check: ${deviceMemory}GB RAM, ${cores} cores`);
        console.log(`Requirements: ${MIN_MEMORY_GB}GB RAM, ${MIN_CORES} cores`);

        if (deviceMemory < MIN_MEMORY_GB || cores < MIN_CORES) {
            console.log(`Hardware below minimum requirements - disabling Phi 3.5-mini`);
            return false;
        }
        return true;
    }

    async _ensureModerationTerms() {
        if (Array.isArray(this.moderationTerms)) {
            return this.moderationTerms;
        }

        if (this.moderationLoadPromise) {
            return this.moderationLoadPromise;
        }

        this.moderationLoadPromise = (async () => {
            const response = await fetch(MODERATION_LIST_PATH, { cache: "no-store" });
            if (!response.ok) {
                throw new Error("Failed to load moderation list.");
            }

            const text = await response.text();
            const lines = text
                .split(/\r?\n/)
                .map((line) => line.trim().toLowerCase())
                .filter((line) => line.length > 0);

            this.moderationTerms = lines
                .map((line) => shiftWord(reverseWord(line), 1))
                .filter((line) => line.length > 0);

            return this.moderationTerms;
        })();

        try {
            return await this.moderationLoadPromise;
        } finally {
            this.moderationLoadPromise = null;
        }
    }

    async _hasReversedModerationMatch(candidatePrompts) {
        const prompts = Array.isArray(candidatePrompts)
            ? candidatePrompts.map((v) => String(v ?? "").toLowerCase())
            : [];

        if (prompts.length === 0) {
            return false;
        }

        const terms = await this._ensureModerationTerms();
        if (!Array.isArray(terms) || terms.length === 0) {
            return false;
        }

        for (const prompt of prompts) {
            for (const term of terms) {
                if (term && prompt.includes(term)) {
                    return true;
                }
            }
        }

        return false;
    }

    _extractModeratedPromptsFromMessages(messages) {
        if (!Array.isArray(messages)) {
            return [];
        }

        return messages
            .filter((message) => message && ["user", "system", "developer"].includes(String(message.role || "")))
            .map((message) => contentToText(message.content));
    }

    _extractModeratedPromptsFromInput(input, instructions = "") {
        const prompts = [];

        if (instructions) {
            prompts.push(String(instructions));
        }

        if (Array.isArray(input)) {
            prompts.push(
                ...input
                    .filter((message) => message && ["user", "system", "developer"].includes(String(message.role || "user")))
                    .map((message) => contentToText(message.content))
            );
            return prompts;
        }

        prompts.push(contentToText(input));
        return prompts;
    }

    _createSafeResponseStream(streamType, requestedRunId = null) {
        const streamId = makeId("stream");
        const responseId = makeId("resp");
        const createdAtVersion = this.sessionVersion;

        const session = {
            queue: [],
            done: true,
            error: null,
            responseId,
            createdAtVersion,
            requestedRunId: Number.isFinite(Number(requestedRunId)) ? Number(requestedRunId) : null,
        };

        if (streamType === "chat") {
            session.queue.push({
                object: "chat.completion.chunk",
                choices: [
                    {
                        index: 0,
                        delta: {
                            content: MODERATION_SAFE_RESPONSE
                        }
                    }
                ]
            });
            session.queue.push({
                object: "chat.completion.chunk",
                choices: [
                    {
                        index: 0,
                        delta: {},
                        finish_reason: "stop"
                    }
                ]
            });
        } else {
            session.queue.push({
                type: "response.output_text.delta",
                delta: MODERATION_SAFE_RESPONSE
            });
            session.queue.push({
                type: "response.completed",
                response: {
                    id: responseId,
                    output_text: MODERATION_SAFE_RESPONSE
                }
            });
        }

        this.responsesById.set(responseId, MODERATION_SAFE_RESPONSE);
        this.streamSessions.set(streamId, session);

        return { stream_id: streamId, response_id: responseId };
    }

    _createSafeChatResponse() {
        const responseId = makeId("chatcmpl");
        this.responsesById.set(responseId, MODERATION_SAFE_RESPONSE);
        return {
            id: responseId,
            object: "chat.completion",
            choices: [
                {
                    index: 0,
                    finish_reason: "stop",
                    message: {
                        role: "assistant",
                        content: MODERATION_SAFE_RESPONSE
                    }
                }
            ]
        };
    }

    _createSafeResponsesResponse() {
        const responseId = makeId("resp");
        this.responsesById.set(responseId, MODERATION_SAFE_RESPONSE);
        return {
            id: responseId,
            object: "response",
            output_text: MODERATION_SAFE_RESPONSE,
            output: [
                {
                    type: "message",
                    role: "assistant",
                    content: [
                        {
                            type: "output_text",
                            text: MODERATION_SAFE_RESPONSE
                        }
                    ]
                }
            ]
        };
    }

    setActiveRunId(runId) {
        const numericRunId = Number(runId);
        this.activeRunId = Number.isFinite(numericRunId) ? numericRunId : 0;
    }

    _ensureActiveRun(runId, context = "request") {
        const numericRunId = Number(runId);
        if (!Number.isFinite(numericRunId)) {
            return;
        }

        if (numericRunId !== this.activeRunId) {
            throw new Error(`Stale ${context} ignored for run ${numericRunId}. Active run is ${this.activeRunId}.`);
        }
    }

    setStatusCallback(callback) {
        this.statusCallback = callback;
    }

    cancelModelLoading() {
        console.log('User requested to cancel model loading');
        this.modelLoadingCancelled = true;

        if (this.modelLoadingAbortController) {
            this.modelLoadingAbortController.abort();
        }

        // Clean up any loading state
        this.isLoading = false;

        this.usingBasic = true;
        this.usingWllama = false;
        this.isReady = true;

        this._status("ready", `${WIKIPEDIA_MODEL_NAME} ready (user cancelled loading)`);

        console.log('Switched to Basic mode after cancellation');
    }

    async resetSession() {
        console.log('[Model Reset] Soft reset - clearing conversation history only');
        this.sessionVersion += 1;
        this.streamSessions.clear();
        this.responsesById.clear();

        // Let in-flight generation loops observe the new sessionVersion and unwind.
        if (this.activeGenerationTasks.size > 0) {
            await Promise.race([
                Promise.allSettled(Array.from(this.activeGenerationTasks)),
                sleep(1500)
            ]);
        }

        // Note: KV cache clearing removed - not needed when doing hard resets
        // and not supported in all wllama versions
    }

    async hardResetSession(options = {}) {
        const { skipReinit = false } = options;
        console.log('[Model Reset] Hard reset - reloading model');

        // Clear any lingering status messages
        this._status("loading", "Resetting model...");

        await this.resetSession();

        // Preserve availability information before reset
        const preservedCpuAvailable = this.availableModes.cpu;

        // Cancel any ongoing model loading only if actually loading
        if (this.isLoading) {
            this.modelLoadingCancelled = true;
            if (this.modelLoadingAbortController) {
                this.modelLoadingAbortController.abort();
            }
            // Give a moment for the cancellation to be observed
            await sleep(100);
        }

        const currentWllama = this.wllama;

        this.wllama = null;
        this.isReady = false;
        this.isLoading = false;
        this.usingWllama = false;
        this.usingBasic = false;

        // Preserve mode availability knowledge
        this.availableModes.cpu = preservedCpuAvailable;

        // Clean up wllama
        if (currentWllama) {
            for (const methodName of ["dispose", "destroy", "unload", "unloadModel", "terminate", "exit"]) {
                const method = currentWllama?.[methodName];
                if (typeof method === "function") {
                    await Promise.resolve(method.call(currentWllama)).catch(() => { });
                }
            }
        }

        // Only reinitialize if not skipping (default behavior for backward compatibility)
        if (!skipReinit) {
            await this.initialize(2);
        }
    }

    getCurrentMode() {
        if (this.usingBasic) {
            return "basic";
        }
        return "cpu";
    }

    getAvailableModes() {
        return {
            cpu: Boolean(this.availableModes.cpu),
            basic: Boolean(this.availableModes.basic)
        };
    }

    _activateBasicMode(reason = "Local model unavailable") {
        this.usingBasic = true;
        this.usingWllama = false;
        this.availableModes.basic = true;
        this._status("ready", `${WIKIPEDIA_MODEL_NAME} ready (${reason})`);
    }

    _status(kind, message) {
        if (typeof this.statusCallback === "function") {
            this.statusCallback({ kind, message });
        }
    }

    async initialize(maxRetries = 3, options = {}) {
        const { forceBasic = false } = options;

        if (this.isReady) {
            return;
        }

        if (this.isLoading) {
            return;
        }

        this.modelLoadingCancelled = false;
        this.modelLoadingAbortController = new AbortController();
        this.isLoading = true;
        this.initSessionId++;
        console.log(`[Initialize] Session ID incremented to ${this.initSessionId}`);

        this.availableModes = {
            cpu: true,
            basic: true
        };

        // Check hardware requirements before attempting to load model
        if (!forceBasic && !this.checkHardwareRequirements()) {
            this.availableModes.cpu = false;
            this._activateBasicMode("AI model hardware requirements not met");
            this.isReady = true;
            this.isLoading = false;
            return;
        }

        if (forceBasic) {
            this._activateBasicMode("forced fallback mode");
            this.isReady = true;
            this.isLoading = false;
            return;
        }

        this._status("loading", "Initializing Phi 3.5-mini...");
        this.usingBasic = false;

        try {
            await this._loadWllama(maxRetries);

            if (this.modelLoadingCancelled) {
                console.log('Wllama loading was cancelled by user');
                return;
            }

            this.usingWllama = true;
            this.availableModes.cpu = true;
            this.isReady = true;
            this.isLoading = false;
        } catch (wllamaError) {
            if (this.modelLoadingCancelled || (wllamaError.message && wllamaError.message.includes('cancelled by user'))) {
                console.log('Wllama loading was cancelled by user');
                return;
            }

            console.error('Wllama initialization failed:', wllamaError);
            this.availableModes.cpu = false;
            this._activateBasicMode("AI model failed to load");
            this.isReady = true;
            this.isLoading = false;
        }
    }

    async _loadWllama(maxRetries = 3) {
        let lastError = null;

        // Capture current session ID to detect stale loads
        const currentSessionId = this.initSessionId;

        for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
            // Check if session changed
            if (this.initSessionId !== currentSessionId) {
                console.log('Wllama load aborted - session changed');
                throw new Error('Session changed during loading');
            }

            // Check for cancellation before each attempt
            if (this.modelLoadingCancelled) {
                throw new Error('Loading cancelled by user');
            }

            try {
                this._status("loading", `Loading local model (attempt ${attempt}/${maxRetries})...`);
                await this._loadWllamaModel();

                // Check if session changed after loading
                if (this.initSessionId !== currentSessionId) {
                    console.log('Wllama load completed but session has changed - discarding result');
                    throw new Error('Session changed during loading');
                }

                // Check if cancelled after loading
                if (this.modelLoadingCancelled) {
                    throw new Error('Loading cancelled by user');
                }

                this._status("ready", "Model ready: Phi 3.5-mini");
                return;
            } catch (error) {
                // If cancelled, rethrow immediately
                if (this.modelLoadingCancelled) {
                    throw error;
                }

                lastError = error;
                this._status("error", `Model load failed on attempt ${attempt}: ${error.message}`);
                if (attempt < maxRetries) {
                    await sleep(1200 * attempt);
                }
            }
        }

        throw lastError || new Error("Model initialization failed");
    }

    async _loadWllamaModel() {
        this.wllamaUsedGPU = false;

        // Detect GPU vendor and skip WebGPU for known-problematic GPUs
        let gpuEnabled = !this.gpuFailed && !!navigator.gpu;
        if (gpuEnabled) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const info = adapter.info ?? await adapter.requestAdapterInfo?.();
                    const vendor = (info?.vendor || '').toLowerCase();
                    if (vendor.includes('qualcomm') || vendor.includes('adreno')) {
                        // Open bug: ggml-org/llama.cpp#23558 — garbled output on Qualcomm WebGPU
                        console.warn('WebGPU disabled: Qualcomm/Adreno GPU detected. Using CPU.');
                        gpuEnabled = false;
                    } else if (vendor.includes('amd') || vendor.includes('advanced micro')) {
                        // Flashattention bug fixed in wllama 3.2.3+ (llama.cpp PR #23040); app uses 3.1.1
                        console.warn('WebGPU disabled: AMD GPU detected. Using CPU.');
                        gpuEnabled = false;
                    }
                } else {
                    gpuEnabled = false;
                }
            } catch (e) {
                console.warn('Could not query WebGPU adapter info:', e);
                gpuEnabled = false;
            }
        }

        const useMultiThread = window.crossOriginIsolated === true;
        const availableThreads = navigator.hardwareConcurrency || 4;
        const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;

        const progressCallback = ({ loaded, total }) => {
            if (!total) {
                this._status("loading", "Loading Phi 3.5-mini...");
                return;
            }
            const pct = Math.round((loaded / total) * 100);
            this._status("loading", `Downloading Phi 3.5-mini: ${pct}%`);
        };

        const modelRef = { repo: MODEL_REPO, quant: MODEL_QUANT };

        const attemptLoad = async (n_gpu_layers, n_threads) => {
            if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
            this.wllama = new Wllama(WASM_PATHS);
            await this.wllama.loadModelFromHF(modelRef, { n_ctx: 712, n_gpu_layers, n_threads, progressCallback });
        };

        if (gpuEnabled) {
            try {
                console.log('Attempting GPU load (32 layers)...');
                this._status("loading", "Loading with GPU acceleration...");
                await attemptLoad(32, preferredThreads);
                this.wllamaUsedGPU = true;
                console.log('Model loaded with GPU acceleration.');
                return;
            } catch (gpuErr) {
                console.warn('GPU load failed, falling back to CPU:', gpuErr);
                this.wllamaUsedGPU = false;
            }
        }

        if (preferredThreads > 1) {
            try {
                console.log('Attempting CPU load (multi-thread)...');
                this._status("loading", "Loading with CPU (multi-thread)...");
                await attemptLoad(0, preferredThreads);
                return;
            } catch (multiErr) {
                console.warn('CPU multi-thread load failed, trying single-thread:', multiErr);
            }
        }

        console.log('Attempting CPU load (single-thread)...');
        this._status("loading", "Loading with CPU (single-thread)...");
        await attemptLoad(0, 1);
    }

    /**
     * Tears down wllama and reloads the model CPU-only.
     * Called when GPU inference produces empty output at runtime.
     */
    async _reloadOnCpu() {
        console.warn('GPU produced empty response — reloading model on CPU.');
        this.gpuFailed = true;
        this.wllamaUsedGPU = false;
        if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }

        const useMultiThread = window.crossOriginIsolated === true;
        const preferredThreads = useMultiThread ? Math.max(1, (navigator.hardwareConcurrency || 4) - 2) : 1;
        const modelRef = { repo: MODEL_REPO, quant: MODEL_QUANT };

        const tryLoad = async (n_threads) => {
            if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
            this.wllama = new Wllama(WASM_PATHS);
            await this.wllama.loadModelFromHF(modelRef, { n_ctx: 712, n_gpu_layers: 0, n_threads, progressCallback: () => { } });
        };

        if (preferredThreads > 1) {
            try { await tryLoad(preferredThreads); } catch (_) { await tryLoad(1); }
        } else {
            await tryLoad(1);
        }
    }

    _ensureClient(model) {
        if (!this.isReady || (!this.usingBasic && !this.wllama)) {
            throw new Error("Model is not ready yet.");
        }
        if (model !== "phi") {
            throw new Error("The model parameter must be 'phi'.");
        }
    }

    _toChatML(messages) {
        let prompt = "";
        for (const message of messages) {
            const role = roleToChatML(message.role);
            const content = contentToText(message.content);
            prompt += `<|im_start|>${role}\n${content}\n<|im_end|>\n\n`;
        }
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    _buildResponsesMessages(input, instructions, previousResponseId) {
        const messages = [];
        if (instructions) {
            messages.push({ role: "developer", content: String(instructions) });
        }

        if (previousResponseId && this.responsesById.has(previousResponseId)) {
            messages.push({ role: "assistant", content: this.responsesById.get(previousResponseId) });
        }

        if (Array.isArray(input)) {
            for (const message of input) {
                messages.push({
                    role: String(message.role || "user"),
                    content: contentToText(message.content)
                });
            }
        } else {
            messages.push({ role: "user", content: contentToText(input) });
        }

        return messages;
    }

    async _complete(messagesOrPrompt, onDelta, expectedSessionVersion = this.sessionVersion) {
        if (this.usingBasic) {
            const messages = Array.isArray(messagesOrPrompt)
                ? messagesOrPrompt
                : this._parseChatMLToMessages(messagesOrPrompt);

            const userMessages = messages.filter((message) => String(message?.role || "") === "user");
            const latestUserText = contentToText(userMessages[userMessages.length - 1]?.content || "");
            const summary = await this._generateWithWikipedia(latestUserText);
            if (summary && typeof onDelta === "function") {
                onDelta(summary);
            }
            return String(summary || "").trim();
        }

        const messages = Array.isArray(messagesOrPrompt) ? messagesOrPrompt : this._parseChatMLToMessages(messagesOrPrompt);
        return await this._completeWithWllama(messages, onDelta, expectedSessionVersion);
    }

    async _completeWithWllama(messages, onDelta, expectedSessionVersion = this.sessionVersion) {
        const useStreaming = typeof onDelta === "function";
        console.log(`[wllama] Phi 3.5-mini (${useStreaming ? "stream" : "sync"}):`, messages);

        if (useStreaming) {
            let fullText = "";
            const completion = await this.wllama.createChatCompletion({
                messages,
                max_tokens: 512,
                temperature: 0.2,
                top_k: 30,
                top_p: 0.85,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                cache_prompt: false,
                stream: true
            });
            for await (const chunk of completion) {
                if (expectedSessionVersion !== this.sessionVersion) break;
                const token = chunk.choices?.[0]?.delta?.content ?? '';
                if (token) {
                    fullText += token;
                    onDelta(token);
                }
            }
            const trimmed = fullText.trim();
            if (!trimmed && this.wllamaUsedGPU && !this.gpuFailed) {
                await this._reloadOnCpu();
                return await this._completeWithWllama(messages, onDelta, expectedSessionVersion);
            }
            return trimmed;
        }

        // Non-streaming: single completion call, returns when fully generated
        const result = await this.wllama.createChatCompletion({
            messages,
            max_tokens: 512,
            temperature: 0.2,
            top_k: 30,
            top_p: 0.85,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            cache_prompt: false,
            stream: false
        });
        const text = String(result?.choices?.[0]?.message?.content ?? '').trim();
        if (!text && this.wllamaUsedGPU && !this.gpuFailed) {
            await this._reloadOnCpu();
            return await this._completeWithWllama(messages, onDelta, expectedSessionVersion);
        }
        return text;
    }

    _parseChatMLToMessages(chatMLPrompt) {
        // Simple parser to convert ChatML back to messages array
        const messages = [];
        const pattern = /<\|im_start\|>(system|user|assistant)\n([\s\S]*?)<\|im_end\|>/g;
        let match;

        while ((match = pattern.exec(chatMLPrompt)) !== null) {
            const role = match[1];
            const content = match[2].trim();
            messages.push({ role, content });
        }

        return messages;
    }

    _extractPreviousAssistantFromMessages(messages) {
        if (!Array.isArray(messages)) {
            return "";
        }

        for (let i = messages.length - 1; i >= 0; i -= 1) {
            const message = messages[i];
            if (!message || String(message.role || "") !== "assistant") {
                continue;
            }

            const text = contentToText(message.content).trim();
            if (text) {
                return text;
            }
        }

        return "";
    }

    _appendPreviousResponseNote(outputText, previousText) {
        const current = String(outputText || "").trim();
        const priorRaw = String(previousText || "").trim();
        const prior = this._stripPreviousResponseNote(priorRaw);
        if (!current || !prior) {
            return current;
        }

        return `${current}\n\n(Previous response: ${prior})`;
    }

    _stripPreviousResponseNote(text) {
        const value = String(text || "");
        if (!value) {
            return "";
        }

        const marker = "\n(Previous response:";
        const markerIndex = value.indexOf(marker);
        if (markerIndex === -1) {
            return value.trim();
        }

        return value.slice(0, markerIndex).trim();
    }

    async _generateWithWikipedia(query) {
        try {
            const firstLine = String(query || "").split("\n")[0];
            const keywords = extractKeywords(firstLine);
            if (!keywords) {
                return "Please enter a more specific query.";
            }

            const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(keywords)}&format=json&origin=*&srlimit=1`;
            const searchResponse = await fetch(searchUrl);
            if (!searchResponse.ok) {
                throw new Error("Wikipedia search request failed");
            }

            const searchData = await searchResponse.json();
            const results = searchData?.query?.search;
            if (!Array.isArray(results) || results.length === 0) {
                return "I'm sorry. I don't know about that topic.";
            }

            const title = results[0].title;
            const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
            const summaryResponse = await fetch(summaryUrl);
            if (!summaryResponse.ok) {
                throw new Error("Wikipedia summary request failed");
            }

            const summaryData = await summaryResponse.json();
            const extract = String(summaryData?.extract || "").trim();
            if (!extract || extract.length < 20) {
                return "I'm sorry. I don't know about that topic.";
            }

            const firstParagraph = extract.split("\n").find((paragraph) => paragraph.trim().length > 0) || extract;
            const concise = extractLeadingSentences(firstParagraph, 2) || firstParagraph.slice(0, 300).trim();
            return concise.length >= 20 ? concise : "I'm sorry. I don't know about that topic.";
        } catch (error) {
            console.error("Wikipedia lookup failed:", error);
            return "Sorry, I had trouble searching Wikipedia right now.";
        }
    }

    async _createStaticResponseStream(streamType, outputText, requestedRunId = null) {
        const streamId = makeId("stream");
        const responseId = makeId("resp");
        const createdAtVersion = this.sessionVersion;

        const session = {
            queue: [],
            done: false,
            error: null,
            responseId,
            createdAtVersion,
            requestedRunId: Number.isFinite(Number(requestedRunId)) ? Number(requestedRunId) : null,
        };

        const text = String(outputText || "");
        this.streamSessions.set(streamId, session);

        const chunks = chunkTextForStreaming(text);
        let streamTask;
        streamTask = (async () => {
            try {
                if (chunks.length === 0) {
                    if (streamType === "chat") {
                        session.queue.push({
                            object: "chat.completion.chunk",
                            choices: [
                                {
                                    index: 0,
                                    delta: {},
                                    finish_reason: "stop"
                                }
                            ]
                        });
                    } else {
                        session.queue.push({
                            type: "response.completed",
                            response: {
                                id: responseId,
                                output_text: ""
                            }
                        });
                    }
                    this.responsesById.set(responseId, "");
                    session.done = true;
                    return;
                }

                for (let i = 0; i < chunks.length; i += 1) {
                    if (createdAtVersion !== this.sessionVersion) {
                        session.done = true;
                        return;
                    }

                    const delta = chunks[i];
                    if (streamType === "chat") {
                        session.queue.push({
                            object: "chat.completion.chunk",
                            choices: [
                                {
                                    index: 0,
                                    delta: {
                                        content: delta
                                    }
                                }
                            ]
                        });
                    } else {
                        session.queue.push({
                            type: "response.output_text.delta",
                            delta
                        });
                    }

                    // Small jitter to mimic natural token streaming cadence.
                    const pauseMs = 20 + Math.floor(Math.random() * 50);
                    await sleep(pauseMs);
                }

                if (createdAtVersion !== this.sessionVersion) {
                    session.done = true;
                    return;
                }

                if (streamType === "chat") {
                    session.queue.push({
                        object: "chat.completion.chunk",
                        choices: [
                            {
                                index: 0,
                                delta: {},
                                finish_reason: "stop"
                            }
                        ]
                    });
                } else {
                    session.queue.push({
                        type: "response.completed",
                        response: {
                            id: responseId,
                            output_text: text
                        }
                    });
                }

                this.responsesById.set(responseId, text);
                session.done = true;
            } catch (error) {
                session.error = error;
                session.done = true;
            }
        })().finally(() => {
            this.activeGenerationTasks.delete(streamTask);
        });

        this.activeGenerationTasks.add(streamTask);

        return { stream_id: streamId, response_id: responseId };
    }

    async _createStreamSession(messagesOrPrompt, streamType = "responses", requestedRunId = null) {
        const streamId = makeId("stream");
        const responseId = makeId("resp");
        const createdAtVersion = this.sessionVersion;

        const session = {
            queue: [],
            done: false,
            error: null,
            responseId,
            createdAtVersion,
            requestedRunId: Number.isFinite(Number(requestedRunId)) ? Number(requestedRunId) : null,
        };

        this.streamSessions.set(streamId, session);

        let generationTask;
        generationTask = this._complete(messagesOrPrompt, (delta) => {
            if (createdAtVersion !== this.sessionVersion) {
                return;
            }

            if (streamType === "chat") {
                session.queue.push({
                    object: "chat.completion.chunk",
                    choices: [
                        {
                            index: 0,
                            delta: {
                                content: delta
                            }
                        }
                    ]
                });
                return;
            }

            session.queue.push({
                type: "response.output_text.delta",
                delta
            });
        }, createdAtVersion).then((finalText) => {
            if (createdAtVersion !== this.sessionVersion) {
                session.done = true;
                return;
            }

            this.responsesById.set(responseId, finalText);
            if (streamType === "chat") {
                session.queue.push({
                    object: "chat.completion.chunk",
                    choices: [
                        {
                            index: 0,
                            delta: {},
                            finish_reason: "stop"
                        }
                    ]
                });
            } else {
                session.queue.push({
                    type: "response.completed",
                    response: {
                        id: responseId,
                        output_text: finalText
                    }
                });
            }
            session.done = true;
        }).catch((error) => {
            session.error = error;
            session.done = true;
        }).finally(() => {
            this.activeGenerationTasks.delete(generationTask);
        });

        this.activeGenerationTasks.add(generationTask);

        return { stream_id: streamId, response_id: responseId };
    }

    async nextStreamChunk(streamId, runId = null) {
        this._ensureActiveRun(runId, "stream chunk");

        const session = this.streamSessions.get(streamId);
        if (!session) {
            return { done: true, chunk: null };
        }

        if (session.requestedRunId !== null && session.requestedRunId !== this.activeRunId) {
            this.streamSessions.delete(streamId);
            return { done: true, chunk: null };
        }

        if (session.createdAtVersion !== this.sessionVersion) {
            this.streamSessions.delete(streamId);
            return { done: true, chunk: null };
        }

        for (let i = 0; i < 300; i += 1) {
            if (session.queue.length > 0) {
                return { done: false, chunk: session.queue.shift() };
            }

            if (session.done) {
                if (session.error) {
                    const message = session.error.message || "Unknown streaming error";
                    this.streamSessions.delete(streamId);
                    return { done: true, error: message };
                }

                this.streamSessions.delete(streamId);
                return { done: true, chunk: null };
            }

            await sleep(50);
        }

        return { done: false, chunk: null };
    }

    async _requestInternal(payload) {
        if (!payload || typeof payload !== "object") {
            throw new Error("Invalid request payload.");
        }

        this._ensureActiveRun(payload.run_id, "model request");

        if (payload.type === "chat.completions.create") {
            this._ensureClient(payload.model);
            let messages = Array.isArray(payload.messages) ? payload.messages : [];
            validateMessages(messages, "messages");

            const moderatedChatPrompts = this._extractModeratedPromptsFromMessages(messages);
            if (await this._hasReversedModerationMatch(moderatedChatPrompts)) {
                if (payload.stream) {
                    const streamMeta = this._createSafeResponseStream("chat", payload.run_id);
                    return {
                        stream: true,
                        stream_id: streamMeta.stream_id,
                        id: streamMeta.response_id
                    };
                }

                return this._createSafeChatResponse();
            }

            if (this.usingBasic) {
                const userMessages = messages.filter((message) => String(message?.role || "") === "user");
                const latestUserText = contentToText(userMessages[userMessages.length - 1]?.content || "");
                const previousAssistant = this._extractPreviousAssistantFromMessages(messages);
                const wikipediaText = await this._generateWithWikipedia(latestUserText);
                const outputText = this._appendPreviousResponseNote(wikipediaText, previousAssistant);

                if (payload.stream) {
                    const streamMeta = await this._createStaticResponseStream("chat", outputText, payload.run_id);
                    return {
                        stream: true,
                        stream_id: streamMeta.stream_id,
                        id: streamMeta.response_id
                    };
                }

                const responseId = makeId("chatcmpl");
                this.responsesById.set(responseId, outputText);
                return {
                    id: responseId,
                    object: "chat.completion",
                    choices: [
                        {
                            index: 0,
                            finish_reason: "stop",
                            message: {
                                role: "assistant",
                                content: outputText
                            }
                        }
                    ]
                };
            }

            if (payload.stream) {
                const streamMeta = await this._createStreamSession(messages, "chat", payload.run_id);
                return {
                    stream: true,
                    stream_id: streamMeta.stream_id,
                    id: streamMeta.response_id
                };
            }

            const outputText = await this._complete(messages);
            const responseId = makeId("chatcmpl");
            this.responsesById.set(responseId, outputText);

            return {
                id: responseId,
                object: "chat.completion",
                choices: [
                    {
                        index: 0,
                        finish_reason: "stop",
                        message: {
                            role: "assistant",
                            content: outputText
                        }
                    }
                ]
            };
        }

        if (payload.type === "responses.create") {
            this._ensureClient(payload.model);

            const normalizedInstructions = payload.instructions ?? payload.insructions;

            const moderatedResponsePrompts = this._extractModeratedPromptsFromInput(payload.input, normalizedInstructions);
            if (await this._hasReversedModerationMatch(moderatedResponsePrompts)) {
                if (payload.stream) {
                    const streamMeta = this._createSafeResponseStream("responses", payload.run_id);
                    return {
                        stream: true,
                        stream_id: streamMeta.stream_id,
                        id: streamMeta.response_id
                    };
                }

                return this._createSafeResponsesResponse();
            }

            let messages = this._buildResponsesMessages(
                payload.input,
                normalizedInstructions,
                payload.previous_response_id
            );
            validateMessages(messages, "input");

            if (this.usingBasic) {
                const userMessages = messages.filter((message) => String(message?.role || "") === "user");
                const latestUserText = contentToText(userMessages[userMessages.length - 1]?.content || "");

                const previousById = payload.previous_response_id && this.responsesById.has(payload.previous_response_id)
                    ? this.responsesById.get(payload.previous_response_id)
                    : "";
                const previousAssistant = previousById || this._extractPreviousAssistantFromMessages(messages);

                const wikipediaText = await this._generateWithWikipedia(latestUserText);
                const outputText = this._appendPreviousResponseNote(wikipediaText, previousAssistant);

                if (payload.stream) {
                    const streamMeta = await this._createStaticResponseStream("responses", outputText, payload.run_id);
                    return {
                        stream: true,
                        stream_id: streamMeta.stream_id,
                        id: streamMeta.response_id
                    };
                }

                const responseId = makeId("resp");
                this.responsesById.set(responseId, outputText);
                return {
                    id: responseId,
                    object: "response",
                    output_text: outputText,
                    output: [
                        {
                            type: "message",
                            role: "assistant",
                            content: [
                                {
                                    type: "output_text",
                                    text: outputText
                                }
                            ]
                        }
                    ]
                };
            }

            if (payload.stream) {
                const streamMeta = await this._createStreamSession(messages, "responses", payload.run_id);
                return {
                    stream: true,
                    stream_id: streamMeta.stream_id,
                    id: streamMeta.response_id
                };
            }

            const outputText = await this._complete(messages);
            const responseId = makeId("resp");
            this.responsesById.set(responseId, outputText);

            return {
                id: responseId,
                object: "response",
                output_text: outputText,
                output: [
                    {
                        type: "message",
                        role: "assistant",
                        content: [
                            {
                                type: "output_text",
                                text: outputText
                            }
                        ]
                    }
                ]
            };
        }

        throw new Error(`Unsupported request type: ${payload.type}`);
    }

    async request(payload) {
        return this._requestInternal(payload);
    }
}

const llmRuntime = new ModelCoderLLM();

const modelCoderSetStatusListener = (callback) => {
    llmRuntime.setStatusCallback(callback);
};

const modelCoderInit = async (maxRetries = 3, options = {}) => {
    await llmRuntime.initialize(maxRetries, options);
};

const modelCoderInitWithMode = async (mode = 'auto', maxRetries = 3) => {
    // mode can be 'cpu', 'basic', or 'auto'
    const options = {};
    if (mode === 'basic') {
        options.forceBasic = true;
    }
    await llmRuntime.initialize(maxRetries, options);
};

const modelCoderSetActiveRunId = (runId) => {
    llmRuntime.setActiveRunId(runId);
};

const modelCoderRequest = async (requestJson) => {
    const payload = JSON.parse(requestJson);
    const response = await llmRuntime.request(payload);
    return JSON.stringify(response);
};

const modelCoderResetSession = async () => {
    await llmRuntime.resetSession();
};

const modelCoderHardResetSession = async (options = {}) => {
    await llmRuntime.hardResetSession(options);
};

const modelCoderNextStreamChunk = async (streamId, runId = null) => {
    const next = await llmRuntime.nextStreamChunk(streamId, runId);
    return JSON.stringify(next);
};

const modelCoderIsUsingCPUMode = () => {
    return llmRuntime.usingWllama;
};

const modelCoderGetCurrentMode = () => {
    return llmRuntime.getCurrentMode();
};

const modelCoderGetAvailableModes = () => {
    return llmRuntime.getAvailableModes();
};

const modelCoderCancelLoading = () => {
    llmRuntime.cancelModelLoading();
};

const modelCoderBridge = {
    modelCoderSetStatusListener,
    modelCoderInit,
    modelCoderInitWithMode,
    modelCoderSetActiveRunId,
    modelCoderRequest,
    modelCoderResetSession,
    modelCoderHardResetSession,
    modelCoderNextStreamChunk,
    modelCoderIsUsingCPUMode,
    modelCoderGetCurrentMode,
    modelCoderGetAvailableModes,
    modelCoderCancelLoading,
};

function attachBridge(target) {
    if (!target) {
        return;
    }
    target.modelCoderSetStatusListener = modelCoderSetStatusListener;
    target.modelCoderInit = modelCoderInit;
    target.modelCoderInitWithMode = modelCoderInitWithMode;
    target.modelCoderSetActiveRunId = modelCoderSetActiveRunId;
    target.modelCoderRequest = modelCoderRequest;
    target.modelCoderResetSession = modelCoderResetSession;
    target.modelCoderHardResetSession = modelCoderHardResetSession;
    target.modelCoderNextStreamChunk = modelCoderNextStreamChunk;
    target.modelCoderIsUsingCPUMode = modelCoderIsUsingCPUMode;
    target.modelCoderGetCurrentMode = modelCoderGetCurrentMode;
    target.modelCoderGetAvailableModes = modelCoderGetAvailableModes;
    target.modelCoderCancelLoading = modelCoderCancelLoading;
    target.modelCoderBridge = modelCoderBridge;
}

attachBridge(globalThis);
attachBridge(typeof window !== "undefined" ? window : null);
attachBridge(typeof self !== "undefined" ? self : null);

export default llmRuntime;
