import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js";

const WASM_PATHS = {
    "single-thread/wllama.wasm": "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm",
    // Force single-thread wasm for stability on static/browser-hosted scenarios.
    "multi-thread/wllama.wasm": "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm"
};

const MODEL_REPO = "ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF";
const MODEL_FILE = "smollm2-360m-instruct-q8_0.gguf";
const MODERATION_LIST_PATH = "./moderation/mod.txt";
const MODERATION_SAFE_RESPONSE = "I'm sorry. I can't help with that.";

function makeId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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
        if (typeof message.content !== "string") {
            throw new Error("Message content must be a string.");
        }
    }
}

class ModelCoderLLM {
    constructor() {
        this.wllama = null;
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

            // List contains reversed words, so reverse each term for prompt matching.
            this.moderationTerms = lines
                .map((line) => line.split("").reverse().join(""))
                .filter((line) => line.length > 0);

            return this.moderationTerms;
        })();

        try {
            return await this.moderationLoadPromise;
        } finally {
            this.moderationLoadPromise = null;
        }
    }

    async _hasReversedModerationMatch(userPrompts) {
        const prompts = Array.isArray(userPrompts)
            ? userPrompts.map((v) => String(v ?? "").toLowerCase())
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

    _extractUserPromptsFromMessages(messages) {
        if (!Array.isArray(messages)) {
            return [];
        }

        return messages
            .filter((message) => message && message.role === "user")
            .map((message) => String(message.content ?? ""));
    }

    _extractUserPromptsFromInput(input) {
        if (Array.isArray(input)) {
            return input
                .filter((message) => message && String(message.role || "user") === "user")
                .map((message) => String(message.content ?? ""));
        }

        return [String(input ?? "")];
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

        if (this.wllama) {
            await this.wllama.kvClear().catch(() => {});
        }
    }

    async hardResetSession() {
        await this.resetSession();

        const current = this.wllama;
        this.wllama = null;
        this.isReady = false;
        this.isLoading = false;

        if (current) {
            for (const methodName of ["dispose", "destroy", "unload", "unloadModel", "terminate", "exit"]) {
                const method = current?.[methodName];
                if (typeof method === "function") {
                    await Promise.resolve(method.call(current)).catch(() => {});
                }
            }
        }

        await this.initialize(2);
    }

    _status(kind, message) {
        if (typeof this.statusCallback === "function") {
            this.statusCallback({ kind, message });
        }
    }

    async initialize(maxRetries = 3) {
        if (this.isReady) {
            return;
        }

        if (this.isLoading) {
            return;
        }

        this.isLoading = true;
        let lastError = null;

        for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
            try {
                this._status("loading", `Loading local model (attempt ${attempt}/${maxRetries})...`);
                await this._loadModel();
                this.isReady = true;
                this._status("ready", "Model ready: SmolLM2 smollm2");
                this.isLoading = false;
                return;
            } catch (error) {
                lastError = error;
                this._status("error", `Model load failed on attempt ${attempt}: ${error.message}`);
                if (attempt < maxRetries) {
                    await sleep(1200 * attempt);
                }
            }
        }

        this.isLoading = false;
        throw lastError || new Error("Model initialization failed");
    }

    async _loadModel() {
        this.wllama = new Wllama(WASM_PATHS);
        await this.wllama.loadModelFromHF(MODEL_REPO, MODEL_FILE, {
            n_ctx: 2048,
            n_threads: 1,
            progressCallback: ({ loaded, total }) => {
                if (!total) {
                    this._status("loading", "Loading local model...");
                    return;
                }
                const pct = Math.round((loaded / total) * 100);
                this._status("loading", `Downloading model: ${pct}%`);
            }
        });
    }

    _ensureClient(model) {
        if (!this.isReady || !this.wllama) {
            throw new Error("Model is not ready yet.");
        }
        if (model !== "smollm2") {
            throw new Error("The model parameter must be 'smollm2'.");
        }
    }

    _toChatML(messages) {
        let prompt = "";
        for (const message of messages) {
            const role = roleToChatML(message.role);
            const content = String(message.content ?? "");
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
                    content: String(message.content || "")
                });
            }
        } else {
            messages.push({ role: "user", content: String(input || "") });
        }

        return messages;
    }

    async _complete(prompt, onDelta, expectedSessionVersion = this.sessionVersion) {
        await this.wllama.kvClear().catch(() => {});

        let previousText = "";
        let fullText = "";

        const stream = await this.wllama.createCompletion(prompt, {
            nPredict: 320,
            seed: -1,
            sampling: {
                temp: 0.6,
                top_k: 40,
                top_p: 0.92,
                penalty_repeat: 1.05,
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

        await this.wllama.kvClear().catch(() => {});
        return fullText.trim();
    }

    async _createStreamSession(prompt, streamType = "responses", requestedRunId = null) {
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
        generationTask = this._complete(prompt, (delta) => {
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
            const messages = Array.isArray(payload.messages) ? payload.messages : [];
            validateMessages(messages, "messages");

            const chatUserPrompts = this._extractUserPromptsFromMessages(messages);
            if (await this._hasReversedModerationMatch(chatUserPrompts)) {
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

            const prompt = this._toChatML(messages);

            if (payload.stream) {
                const streamMeta = await this._createStreamSession(prompt, "chat", payload.run_id);
                return {
                    stream: true,
                    stream_id: streamMeta.stream_id,
                    id: streamMeta.response_id
                };
            }

            const outputText = await this._complete(prompt);
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

            const responsesUserPrompts = this._extractUserPromptsFromInput(payload.input);
            if (await this._hasReversedModerationMatch(responsesUserPrompts)) {
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

            const normalizedInstructions = payload.instructions ?? payload.insructions;
            const messages = this._buildResponsesMessages(
                payload.input,
                normalizedInstructions,
                payload.previous_response_id
            );
            validateMessages(messages, "input");
            const prompt = this._toChatML(messages);

            if (payload.stream) {
                const streamMeta = await this._createStreamSession(prompt, "responses", payload.run_id);
                return {
                    stream: true,
                    stream_id: streamMeta.stream_id,
                    id: streamMeta.response_id
                };
            }

            const outputText = await this._complete(prompt);
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

const modelCoderInit = async (maxRetries = 3) => {
    await llmRuntime.initialize(maxRetries);
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

const modelCoderHardResetSession = async () => {
    await llmRuntime.hardResetSession();
};

const modelCoderNextStreamChunk = async (streamId, runId = null) => {
    const next = await llmRuntime.nextStreamChunk(streamId, runId);
    return JSON.stringify(next);
};

const modelCoderBridge = {
    modelCoderSetStatusListener,
    modelCoderInit,
    modelCoderSetActiveRunId,
    modelCoderRequest,
    modelCoderResetSession,
    modelCoderHardResetSession,
    modelCoderNextStreamChunk,
};

function attachBridge(target) {
    if (!target) {
        return;
    }
    target.modelCoderSetStatusListener = modelCoderSetStatusListener;
    target.modelCoderInit = modelCoderInit;
    target.modelCoderSetActiveRunId = modelCoderSetActiveRunId;
    target.modelCoderRequest = modelCoderRequest;
    target.modelCoderResetSession = modelCoderResetSession;
    target.modelCoderHardResetSession = modelCoderHardResetSession;
    target.modelCoderNextStreamChunk = modelCoderNextStreamChunk;
    target.modelCoderBridge = modelCoderBridge;
}

attachBridge(globalThis);
attachBridge(typeof window !== "undefined" ? window : null);
attachBridge(typeof self !== "undefined" ? self : null);

export default llmRuntime;
