import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js";

const WASM_PATHS = {
    "single-thread/wllama.wasm": "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm",
    "multi-thread/wllama.wasm": "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm"
};

const PHI2_REPO = "Felladrin/gguf-sharded-phi-2-orange-v2";
const PHI2_FILE = "phi-2-orange-v2.Q5_K_M.shard-00001-of-00025.gguf";
const PHI3_MODEL_ID = "Phi-3-mini-4k-instruct-q4f16_1-MLC";
const MODERATION_LIST_PATH = "./moderation/mod.txt";
const MODERATION_SAFE_RESPONSE = "I'm sorry. I can't help with that. Either your system instructions or user input included content that was flagged by the moderation system. If you think this was a mistake, please try rephrasing your input or instructions and try again.";
const WIKIPEDIA_MODEL_NAME = "Wikipedia API (Basic Chat)";

const STOPWORDS = new Set([
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "of", "in", "on", "at", "to", "from", "with", "by", "for",
    "about", "as", "that", "this", "these", "those", "it", "they",
    "he", "she", "we", "you", "i", "me", "my", "him", "her", "us",
    "them", "which", "who", "whom", "whose", "what", "where", "when",
    "why", "how", "can", "could", "will", "would", "should",
    "may", "might", "must", "find", "search", "show", "tell", "look",
    "ebay", "sale", "buy", "price", "cost", "need", "one"
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
        this.engine = null;  // WebLLM engine for GPU mode
        this.wllama = null;  // wllama engine for CPU mode
        this.usingWllama = false;  // Track which engine is active
        this.usingBasic = false;  // Basic Chat Wikipedia mode
        this.webllmAvailable = false;  // Track if WebLLM model successfully loaded
        this.availableModes = { gpu: false, cpu: true, basic: true };
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

        // Clear KV cache for wllama if it's being used
        if (this.wllama && this.usingWllama) {
            await this.wllama.kvClear().catch(() => { });
        }

        // Note: WebLLM engine doesn't require explicit KV cache clearing
    }

    async hardResetSession(options = {}) {
        const { skipReinit = false } = options;
        console.log('[Model Reset] Hard reset - reloading model');
        await this.resetSession();

        const currentWllama = this.wllama;
        const currentEngine = this.engine;

        this.wllama = null;
        this.engine = null;
        this.isReady = false;
        this.isLoading = false;
        this.usingWllama = false;
        this.usingBasic = false;
        this.webllmAvailable = false;

        // Clean up wllama
        if (currentWllama) {
            for (const methodName of ["dispose", "destroy", "unload", "unloadModel", "terminate", "exit"]) {
                const method = currentWllama?.[methodName];
                if (typeof method === "function") {
                    await Promise.resolve(method.call(currentWllama)).catch(() => { });
                }
            }
        }

        // Clean up WebLLM engine
        if (currentEngine) {
            // WebLLM engine cleanup if needed
            // Currently no explicit cleanup required for WebLLM
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
        return this.usingWllama ? "cpu" : "gpu";
    }

    getAvailableModes() {
        return {
            gpu: Boolean(this.availableModes.gpu),
            cpu: Boolean(this.availableModes.cpu),
            basic: Boolean(this.availableModes.basic)
        };
    }

    _activateBasicMode(reason = "Local model unavailable") {
        this.usingBasic = true;
        this.usingWllama = false;
        this.webllmAvailable = false;
        this.availableModes.basic = true;
        this._status("ready", `${WIKIPEDIA_MODEL_NAME} ready (${reason})`);
    }

    _status(kind, message) {
        if (typeof this.statusCallback === "function") {
            this.statusCallback({ kind, message });
        }
    }

    checkWebGPUSupport() {
        // Check if WebGPU is available in the browser
        if (!navigator.gpu) {
            console.log('WebGPU not supported in this browser');
            return false;
        }
        return true;
    }

    async initialize(maxRetries = 3, options = {}) {
        const { forceCPU = false, forceGPU = false, forceBasic = false } = options;

        if (this.isReady) {
            return;
        }

        if (this.isLoading) {
            return;
        }

        this.isLoading = true;

        this.availableModes = {
            gpu: false,  // Only set true once WebLLM actually loads successfully, not just because navigator.gpu exists
            cpu: true,
            basic: true
        };

        if (forceBasic) {
            this._activateBasicMode("forced fallback mode");
            this.isReady = true;
            this.isLoading = false;
            return;
        }

        // If forcing CPU mode, skip GPU check and go straight to wllama
        if (forceCPU) {
            console.log('Forcing CPU mode (wllama)');
            this.webllmAvailable = false;
            this.usingBasic = false;
            try {
                await this._loadWllama(maxRetries);
                this.usingWllama = true;
                this.availableModes.cpu = true;
                this.isReady = true;
                this.isLoading = false;
                return;
            } catch (wllamaError) {
                console.error('Wllama initialization failed:', wllamaError);
                this.availableModes.cpu = false;
                this.isLoading = false;
                throw wllamaError;
            }
        }

        // Check for WebGPU support before attempting to load WebLLM
        const hasWebGPU = this.checkWebGPUSupport();

        if (!hasWebGPU) {
            if (forceGPU) {
                this.isLoading = false;
                throw new Error('GPU mode requested but WebGPU is not available');
            }
            console.log('WebGPU not available, using wllama (CPU mode)');
            this.webllmAvailable = false;
            this.availableModes.gpu = false;
            this.usingBasic = false;
            try {
                await this._loadWllama(maxRetries);
                this.usingWllama = true;
                this.availableModes.cpu = true;
                this.isReady = true;
                this.isLoading = false;
                return;
            } catch (wllamaError) {
                console.error('Wllama initialization failed:', wllamaError);
                this.availableModes.cpu = false;
                this._activateBasicMode("GPU unavailable and CPU init failed");
                this.isReady = true;
                this.isLoading = false;
                return;
            }
        }

        // Try WebLLM first (faster with GPU) unless forcing CPU
        try {
            console.log('Attempting to initialize WebLLM with WebGPU...');
            await this._loadWebLLM();
            console.log('WebLLM initialized successfully');
            this.webllmAvailable = true;
            this.usingWllama = false;
            this.usingBasic = false;
            this.availableModes.gpu = true;
            this.isReady = true;
            this.isLoading = false;
            return;
        } catch (error) {
            console.error('WebLLM initialization failed, loading wllama fallback:', error);
            this.webllmAvailable = false;
            this.availableModes.gpu = false;

            if (forceGPU) {
                this.isLoading = false;
                throw new Error('GPU mode requested but WebLLM failed to initialize: ' + error.message);
            }

            try {
                await this._loadWllama(maxRetries);
                console.log('Wllama initialized successfully as fallback');
                this.usingWllama = true;
                this.usingBasic = false;
                this.availableModes.cpu = true;
                this.isReady = true;
                this.isLoading = false;
                return;
            } catch (wllamaError) {
                console.error('Both WebLLM and wllama initialization failed:', wllamaError);
                this.availableModes.cpu = false;
                this._activateBasicMode("GPU and CPU init failed");
                this.isReady = true;
                this.isLoading = false;
                return;
            }
        }
    }

    async _loadWebLLM() {
        console.log('_loadWebLLM called - starting model initialization');
        this._status("loading", "Discovering available models...");

        // Check if WebLLM is available
        if (!webllm || !webllm.CreateMLCEngine || !webllm.prebuiltAppConfig) {
            console.error('WebLLM check failed');
            throw new Error('WebLLM not properly loaded');
        }

        // Get available models from WebLLM
        const models = webllm.prebuiltAppConfig.model_list;
        console.log('All available models:', models.map(m => m.model_id));

        // Filter for the specific Phi-3 model only
        let availableModels = models.filter(model =>
            model.model_id === PHI3_MODEL_ID
        );

        if (availableModels.length === 0) {
            throw new Error('Phi-3-mini-4k-instruct model not found');
        }

        console.log('Available models for loading:', availableModels.map(m => m.model_id));
        this._status("loading", "Loading WebLLM model (GPU mode)...");

        // Try to load the model
        try {
            console.log(`Trying to load model: ${PHI3_MODEL_ID}`);

            this.engine = await webllm.CreateMLCEngine(
                PHI3_MODEL_ID,
                {
                    initProgressCallback: (progress) => {
                        console.log('Progress:', progress);
                        const percentage = Math.max(15, Math.round(progress.progress * 85) + 15);
                        const progressText = `Loading ${PHI3_MODEL_ID}: ${Math.round(progress.progress * 100)}%`;
                        this._status("loading", progressText);
                    }
                }
            );

            console.log(`Successfully loaded model: ${PHI3_MODEL_ID}`);
            this._status("ready", "Model ready: Phi-3 (GPU mode)");
        } catch (modelError) {
            console.error(`Failed to load ${PHI3_MODEL_ID}:`, modelError);
            throw modelError;
        }
    }

    async _loadWllama(maxRetries = 3) {
        let lastError = null;

        for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
            try {
                this._status("loading", `Loading local model (attempt ${attempt}/${maxRetries})...`);
                await this._loadWllamaModel();
                this._status("ready", "Model ready: Phi-2 (CPU mode)");
                return;
            } catch (error) {
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
        this.wllama = new Wllama(WASM_PATHS);

        const useMultiThread = window.crossOriginIsolated === true;
        const availableThreads = navigator.hardwareConcurrency || 4;
        const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;

        const baseConfig = {
            n_ctx: 384,
            n_threads: preferredThreads,
            progressCallback: ({ loaded, total }) => {
                if (!total) {
                    this._status("loading", "Loading local model...");
                    return;
                }
                const pct = Math.round((loaded / total) * 100);
                this._status("loading", `Downloading model: ${pct}%`);
            }
        };

        try {
            await this.wllama.loadModelFromHF(PHI2_REPO, PHI2_FILE, baseConfig);
        } catch (multiErr) {
            if (preferredThreads > 1) {
                await this.wllama.loadModelFromHF(PHI2_REPO, PHI2_FILE, {
                    ...baseConfig,
                    n_threads: 1
                });
            } else {
                throw multiErr;
            }
        }

        await this._warmWllamaCache();
    }

    async _warmWllamaCache() {
        if (!this.wllama) {
            return;
        }

        const systemInstruction = '<|im_start|>system\nYou are a helpful coding assistant.\n<|im_end|>';
        try {
            await this.wllama.createCompletion(systemInstruction, {
                nPredict: 1,
                sampling: {
                    temp: 0.0
                }
            });
        } catch (error) {
            console.log('[wllama] Cache warmup failed (non-critical):', error?.message || error);
        }
    }

    _ensureClient(model) {
        if (!this.isReady || (!this.usingBasic && !this.wllama && !this.engine)) {
            throw new Error("Model is not ready yet.");
        }
        if (model !== "local-llm") {
            throw new Error("The model parameter must be 'local-llm'.");
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

    _translateToPhi3Prompt(messages) {
        // For Phi-3, we just pass messages through without aggressive translation
        // Phi-3 is capable enough to handle the requests as-is
        // We only need to ensure consistent role mapping
        console.log('[Phi-3] Original messages:', messages);

        const translatedMessages = [];

        for (const message of messages) {
            const role = message.role;
            const content = contentToText(message.content);

            // Map developer/system to system, keep everything else as-is
            if (role === "developer") {
                translatedMessages.push({ role: "system", content });
            } else {
                translatedMessages.push({ role, content });
            }
        }

        console.log('[Phi-3] Translated messages:', translatedMessages);
        return translatedMessages;
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

        // Route to appropriate engine
        if (this.usingWllama) {
            // wllama expects ChatML prompt
            const prompt = typeof messagesOrPrompt === 'string' ? messagesOrPrompt : this._toChatML(messagesOrPrompt);
            return await this._completeWithWllama(prompt, onDelta, expectedSessionVersion);
        } else {
            // WebLLM expects messages array
            const messages = Array.isArray(messagesOrPrompt) ? messagesOrPrompt : this._parseChatMLToMessages(messagesOrPrompt);
            return await this._completeWithWebLLM(messages, onDelta, expectedSessionVersion);
        }
    }

    async _completeWithWllama(prompt, onDelta, expectedSessionVersion = this.sessionVersion) {
        console.log('[wllama] Sending to Phi-2 (ChatML):', prompt.substring(0, 200) + '...');

        let previousText = "";
        let fullText = "";

        const stream = await this.wllama.createCompletion(prompt, {
            nPredict: 200,
            seed: -1,
            sampling: {
                temp: 0.1,
                top_k: 20,
                top_p: 0.85,
                penalty_repeat: 1.1,
                mirostat: 0
            },
            stopTokens: ["<|im_end|>", "<|im_start|>"],
            stream: true
        });

        for await (const chunk of stream) {
            if (expectedSessionVersion !== this.sessionVersion) {
                break;
            }

            if (!chunk.currentText) {
                continue;
            }

            fullText = chunk.currentText;
            const delta = fullText.slice(previousText.length);
            if (delta && typeof onDelta === "function") {
                onDelta(delta);
            }
            previousText = fullText;
        }

        await this.wllama.kvClear().catch(() => { });
        return fullText.trim();
    }

    async _completeWithWebLLM(messages, onDelta, expectedSessionVersion = this.sessionVersion) {
        // WebLLM expects messages array directly (not ChatML)
        console.log('[WebLLM] Sending to Phi-3-mini:', messages);
        let fullText = "";

        const completion = await this.engine.chat.completions.create({
            messages: messages,
            temperature: 0.7,
            max_tokens: 320,
            stream: true
        });

        for await (const chunk of completion) {
            if (expectedSessionVersion !== this.sessionVersion) {
                break;
            }

            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
                fullText += content;
                if (typeof onDelta === "function") {
                    onDelta(content);
                }
            }
        }

        return fullText.trim();
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

            // Translate messages for Phi-3 when using WebLLM
            if (!this.usingWllama) {
                messages = this._translateToPhi3Prompt(messages);
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

            // Translate messages for Phi-3 when using WebLLM
            if (!this.usingWllama) {
                messages = this._translateToPhi3Prompt(messages);
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
    // mode can be 'auto', 'gpu', 'cpu', or 'basic'
    const options = {};
    if (mode === 'gpu') {
        options.forceGPU = true;
    } else if (mode === 'cpu') {
        options.forceCPU = true;
    } else if (mode === 'basic') {
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
    target.modelCoderBridge = modelCoderBridge;
}

attachBridge(globalThis);
attachBridge(typeof window !== "undefined" ? window : null);
attachBridge(typeof self !== "undefined" ? self : null);

export default llmRuntime;
