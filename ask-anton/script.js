import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.83/+esm";
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/index.js';

// Delay (ms) before clearing the search-status hint shown beneath the input
// after a response finishes streaming. Tuned to stay visible long enough to read.
const SEARCH_STATUS_CLEAR_DELAY = 2000;

// Inline styles for the small italic note that accompanies the typing indicator
// in CPU and Basic modes (kept inline so a single innerHTML write covers both).
const TYPING_NOTE_STYLE = 'font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;';

// Stop words for n-gram search (performSearch). Hoisted so the Set isn't
// reallocated on every keystroke / question. Tuned for short question forms.
const SEARCH_STOP_WORDS = new Set([
    'what', 'is', 'are', 'the', 'a', 'an', 'how', 'does', 'do', 'can', 'about',
    'tell', 'me', 'explain', 'describe', 'show', 'give', 'anton',
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'why', 'which', 'whom', 'whose',
    'all', 'any', 'this', 'that', 'these', 'those'
]);

// Stop words for the Bing keyword extractor (extractBingSearchKeywords).
// Broader than SEARCH_STOP_WORDS because we strip common verbs / pronouns
// to keep the resulting query short and on-topic.
const BING_STOP_WORDS = new Set([
    // Articles, prepositions, conjunctions
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'with',
    'or', 'but', 'if', 'than', 'then', 'so', 'yet',
    'after', 'before', 'between', 'during', 'into', 'through', 'over',
    'under', 'until', 'up', 'down', 'out', 'off', 'above', 'below',
    // Pronouns
    'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'our', 'their', "i'm",
    "you're", "he's", "she's", "we're", "they're",
    // Determiners and quantifiers
    'this', 'these', 'those', 'some', 'any', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'other', 'another', 'much', 'many',
    // Verbs (auxiliary, modal, and common generic)
    'am', 'was', 'were', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
    'may', 'might', 'must', 'shall', 'ought', 'will',
    'get', 'make', 'know', 'see', 'take', 'come', 'go', 'want',
    'use', 'find', 'need', 'try', 'ask', 'work', 'help', 'like', 'seem',
    'become', 'let', 'tell', 'show', 'give', 'provide', 'explain',
    'describe', 'define',
    // Question words
    'what', 'when', 'where', 'who', 'how', 'why', 'which', 'whom',
    'whose', 'whether', "what's", 'whats', "who's", 'whos', "how's",
    'hows',
    // Common adverbs
    'also', 'just', 'now', 'here', 'there', 'very', 'too',
    'really', 'still', 'always', 'never', 'often', 'sometimes', 'maybe',
    'perhaps', 'about',
    // Other common words
    'yes', 'no', 'thing', 'something', 'anything', 'nothing',
    'everything', 'someone', 'anyone', 'everyone', 'understand',
    'think', 'believe', 'feel', 'appear', 'say',
    'anton', 'please', 'using', 'search', 'docs',
    'documentation', 'learn', 'details', 'overview'
]);

class AskAnton {
    constructor() {
        // Debug flags for testing failover (can be set via URL params or console)
        this.debugConfig = this.parseDebugConfig();

        this.engine = null;      // WebLLM engine
        this.wllama = null;      // Wllama engine (fallback)
        this.conversationHistory = [];
        this.isGenerating = false;
        this.indexData = null;
        this.stopRequested = false;
        this.currentStream = null;
        this.currentAbortController = null;
        this.isStoppingGeneration = false;
        this.currentMode = 'basic';
        this.availableModes = {
            gpu: false,
            cpu: true,
            basic: true
        };
        this.isLoadingModel = false;
        this.modelLoadingCancelled = false;
        this.modelLoadingAbortController = null;
        this.currentModal = null;
        this.lastFocusedElement = null;
        this.modalFocusTrapHandler = null;
        this.videoPopupWidth = 800;
        this.videoPopupHeight = 600;
        this.usedVoiceInput = false;
        this.gpuModeFailureMessage = "I'm sorry, something went wrong in GPU mode. If this keeps happening, please try switching to CPU mode or Basic mode.";
        this.cpuModeFailureMessage = "I'm sorry, something went wrong in CPU mode. If this keeps happening, please try switching to Basic mode.";
        this.lastWebLLMCompletionErrored = false;
        this.lastWllamaCompletionErrored = false;

        // Vosk speech recognition (lazy-loaded fallback)
        this.voskModel = null;
        this.voskRecognizer = null;
        this.voskLoaded = false;
        this.voskLoadingFailed = false;
        this.isRecording = false;
        this.mediaStream = null;
        this.audioContext = null;
        this.processorNode = null;
        this.sourceNode = null;
        // Calculate speech model path relative to the base path
        const basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
        const rootPath = basePath.substring(0, basePath.lastIndexOf('/'));
        this.speechModelUrl = `${rootPath}/speech-model/speech-model.tar.gz`;
        this.silenceTimer = null;
        this.noSpeechTimer = null;
        this.lastSpeechTime = null;
        this.hasSpeech = false;
        this.silenceTimeout = 2000; // Auto-stop after 2 seconds of silence
        this.noSpeechTimeout = 5000; // Cancel after 5 seconds of no speech
        this.usingWebSpeech = true; // Try Web Speech API first

        this.elements = {
            progressSection: document.getElementById('progress-section'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            cancelLoadingLink: document.getElementById('cancel-loading-link'),
            chatContainer: document.getElementById('chat-container'),
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            micBtn: document.getElementById('mic-btn'),
            restartBtn: document.getElementById('restart-btn'),
            searchStatus: document.getElementById('search-status'),
            modeSelect: document.getElementById('mode-select'),
            aboutBtn: document.getElementById('about-btn'),
            aboutModal: document.getElementById('about-modal'),
            aboutModalClose: document.getElementById('about-modal-close'),
            aboutModalOk: document.getElementById('about-modal-ok'),
            aiModeModal: document.getElementById('ai-mode-modal'),
            modalClose: document.getElementById('modal-close'),
            modalOk: document.getElementById('modal-ok')
        };

        // Prompt constants for consistent behavior across both models
        this.SYSTEM_PROMPT = `You are an AI tutor that provides learner-friendly answers to questions about AI. You politely decline to discuss topics not related to AI or computing.`;

        this.PROMPT_WITH_CONTEXT = `Respond based on the following information:`;
        this.PROMPT_WITHOUT_CONTEXT = `Continue the conversation, keeping responses concise and focused on AI topics. If you don't know the answer, say you don't know.`;

        // Prohibited words for content moderation (whole words only)
        this.prohibitedWords = [];

        this.initialize();
    }

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    /**
     * Read URL query string for debug overrides used to force engine-init
     * failure paths (?debug=true&forceWebGPUFail=true&forceWllamaFail=true
     * &forceBasicMode=true). Returns a config object consumed by
     * {@link initializeEngine}, {@link initializeWebLLM}, {@link initializeWllama}.
     * @returns {{enabled:boolean, forceWebGPUFail:boolean, forceWllamaFail:boolean, forceBasicMode:boolean}}
     */
    parseDebugConfig() {
        // Parse URL parameters for debug flags
        // Usage: ?debug=true&forceWebGPUFail=true&forceWllamaFail=true
        const params = new URLSearchParams(window.location.search);
        const config = {
            enabled: params.has('debug'),
            forceWebGPUFail: params.has('forceWebGPUFail') || params.get('forceWebGPUFail') === 'true',
            forceWllamaFail: params.has('forceWllamaFail') || params.get('forceWllamaFail') === 'true',
            forceBasicMode: params.has('forceBasicMode') || params.get('forceBasicMode') === 'true'
        };

        if (config.enabled) {
            console.log('🧪 Debug mode enabled:', config);
            console.log('💡 To force failures, add URL params: ?debug=true&forceWebGPUFail=true&forceWllamaFail=true');
            console.log('💡 Or use console: window.askAnton.debugConfig.forceWebGPUFail = true');
        }

        return config;
    }

    /**
     * Top-level boot sequence: load moderation list, load knowledge-base
     * index, pick & initialize an inference engine (with mode fallback),
     * then wire DOM event listeners. Surface errors via {@link showError}.
     */
    async initialize() {
        try {
            // Setup event listeners FIRST so cancel link works during loading
            this.setupEventListeners();

            // Hide cancel link initially
            if (this.elements.cancelLoadingLink) {
                this.elements.cancelLoadingLink.style.display = 'none';
            }

            // Load prohibited words used by content moderation
            await this.loadProhibitedWords();

            // Load the index (no longer loading Vosk upfront)
            await this.loadIndex();

            // Try to initialize WebLLM first, fall back to wllama if needed
            await this.initializeEngine();

        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize. Please refresh the page.');
        }
    }

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================

    /** Reverse a string character-by-character. */
    reverseWord(text) {
        return text.split('').reverse().join('');
    }

    /** Caesar-shift each character's code point by `amount`. */
    shiftWord(text, amount) {
        return text
            .split('')
            .map(char => String.fromCharCode(char.charCodeAt(0) + amount))
            .join('');
    }

    /**
     * Load `moderation/mod.txt` and decode each line. Entries are stored
     * lightly obfuscated (reversed + shifted by 1) so the raw word list is
     * not visible in the repo or network tab; decoding here yields the
     * lowercase words used by {@link containsProhibitedWords}.
     * Also precompiles a whole-word regex per term into `prohibitedPatterns`
     * so moderation checks don't rebuild regexes on every message.
     */
    async loadProhibitedWords() {
        try {
            const response = await fetch('moderation/mod.txt', { cache: 'no-store' });
            if (!response.ok) throw new Error('Failed to load prohibited words');

            const encodedWordsText = await response.text();
            this.prohibitedWords = encodedWordsText
                .split(/\r?\n/)
                .map(word => word.trim())
                .filter(word => word.length > 0)
                .map(word => this.shiftWord(this.reverseWord(word.toLowerCase()), 1));

            // Escape regex metacharacters in each decoded word, then compile
            // a case-insensitive whole-word matcher once.
            this.prohibitedPatterns = this.prohibitedWords.map(word => ({
                word,
                regex: new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i')
            }));

            console.log('Loaded prohibited words:', this.prohibitedWords.length);
        } catch (error) {
            console.error('Error loading prohibited words:', error);
            throw error;
        }
    }

    /**
     * Fetch `index.json` (the knowledge base) and build `this.keywordMap`,
     * a flat lookup from normalized keyword -> Array<{document, category, link}>
     * used by {@link performSearch} for n-gram matching. Arrays (rather than
     * a single value) preserve every document that declares a shared keyword.
     */
    async loadIndex() {
        try {
            this.updateProgress(5, 'Loading knowledge base...');
            const response = await fetch('index.json', { cache: 'no-store' });
            if (!response.ok) throw new Error('Failed to load index');
            this.indexData = await response.json();
            console.log('Loaded index with', this.indexData.length, 'categories');

            // Build a flat lookup map: keyword -> [{document, category, link}, ...]
            // (multiple entries per keyword so collisions don't drop documents.)
            this.keywordMap = new Map();
            this.indexData.forEach(category => {
                category.documents.forEach(doc => {
                    doc.keywords.forEach(keyword => {
                        const normalizedKeyword = keyword.toLowerCase().trim();
                        if (!normalizedKeyword) return;
                        let entries = this.keywordMap.get(normalizedKeyword);
                        if (!entries) {
                            entries = [];
                            this.keywordMap.set(normalizedKeyword, entries);
                        }
                        entries.push({
                            document: doc,
                            category: category.category,
                            link: category.link
                        });
                    });
                });
            });
            console.log('Built keyword map with', this.keywordMap.size, 'keywords');
        } catch (error) {
            console.error('Error loading index:', error);
            throw error;
        }
    }

    /**
     * Lazy-load the Vosk WASM speech model used as a fallback when the
     * browser's Web Speech API is unavailable. Idempotent: returns the
     * cached load result after the first call.
     * @returns {Promise<boolean>} true if the model is ready, false if loading failed.
     */
    async loadVoskModel() {
        if (this.voskLoaded || this.voskLoadingFailed) {
            return this.voskLoaded;
        }

        try {
            console.log('Loading Vosk speech model from', this.speechModelUrl);

            if (!window.Vosk || typeof Vosk.createModel !== 'function') {
                console.warn('Vosk library not loaded');
                this.voskLoadingFailed = true;
                return false;
            }

            const loadingMsg = this.addSystemMessage('Loading offline speech model... This may take a moment.');
            this.disableInput();
            this.elements.micBtn.disabled = true;

            this.voskModel = await Vosk.createModel(this.speechModelUrl);
            this.voskRecognizer = new this.voskModel.KaldiRecognizer(16000);

            // Set up recognizer event handlers
            this.voskRecognizer.on("result", (message) => {
                const result = message.result;
                if (result && result.text) {
                    // Clear no-speech timer since we got speech
                    if (this.noSpeechTimer) {
                        clearTimeout(this.noSpeechTimer);
                        this.noSpeechTimer = null;
                    }

                    // Append the recognized text to the input
                    const currentText = this.elements.userInput.value;
                    this.elements.userInput.value = currentText + (currentText ? " " : "") + result.text;
                    this.autoResizeTextarea();
                    this.hasSpeech = true;
                    this.lastSpeechTime = Date.now();
                    this.resetSilenceTimer();
                }
            });

            this.voskRecognizer.on("partialresult", (message) => {
                // Reset silence timer on partial results too
                const result = message.result;
                if (result && result.partial && result.partial.trim()) {
                    // Clear no-speech timer on partial results
                    if (this.noSpeechTimer) {
                        clearTimeout(this.noSpeechTimer);
                        this.noSpeechTimer = null;
                    }

                    this.lastSpeechTime = Date.now();
                    this.resetSilenceTimer();
                }
            });

            this.voskLoaded = true;
            console.log('Vosk speech model loaded successfully');

            // Update the loading message
            const msgP = loadingMsg.querySelector('p');
            if (msgP) {
                msgP.textContent = 'Offline speech model ready! Please try your voice input again.';
            }

            // Only re-enable inputs if we're not currently loading a model
            if (!this.isLoadingModel) {
                this.enableInput();
                this.elements.micBtn.disabled = false;
            }
            return true;
        } catch (error) {
            console.error('Error loading Vosk model:', error);
            this.voskLoadingFailed = true;
            this.addSystemMessage('Failed to load offline speech model. Voice input is unavailable.');
            // Only re-enable inputs if we're not currently loading a model
            if (!this.isLoadingModel) {
                this.enableInput();
                this.elements.micBtn.disabled = false;
            }
            return false;
        }
    }

    // ============================================================================
    // LLM ENGINE INITIALIZATION (WebLLM & Wllama)
    // ============================================================================

    /** @returns {boolean} true when WebGPU is exposed by the browser. */
    checkWebGPUSupport() {
        // Check if WebGPU is available in the browser
        if (!navigator.gpu) {
            console.log('WebGPU not supported in this browser');
            return false;
        }
        return true;
    }

    /**
     * Run `task` with `navigator.gpu` hidden so any code path inside it
     * cannot detect or use WebGPU. Restores the original descriptor on
     * exit even if `task` throws. Used to force the CPU code path in
     * libraries that auto-detect WebGPU.
     */
    async withWebGpuTemporarilyDisabled(task) {
        const nav = navigator;
        const hadOwnGpu = Object.prototype.hasOwnProperty.call(nav, 'gpu');
        const ownGpuDescriptor = hadOwnGpu ? Object.getOwnPropertyDescriptor(nav, 'gpu') : null;
        let gpuMasked = false;

        const unavailableGpu = {
            requestAdapter: async () => null
        };

        try {
            Object.defineProperty(nav, 'gpu', {
                configurable: true,
                enumerable: false,
                get: () => unavailableGpu
            });
            gpuMasked = true;
            console.log('Temporarily stubbed navigator.gpu as unavailable for wllama initialization');
        } catch (error) {
            console.warn('Unable to mask navigator.gpu during wllama initialization:', error);
        }

        try {
            return await task();
        } finally {
            if (!gpuMasked) {
                return;
            }

            try {
                if (hadOwnGpu && ownGpuDescriptor) {
                    Object.defineProperty(nav, 'gpu', ownGpuDescriptor);
                } else {
                    delete nav.gpu;
                }
                console.log('Restored navigator.gpu after wllama initialization');
            } catch (error) {
                console.warn('Unable to restore navigator.gpu after wllama initialization:', error);
            }
        }
    }

    /**
     * Run `task` while transparently rewriting `new Worker(url)` so worker
     * scripts also see WebGPU as unavailable. Pairs with
     * {@link withWebGpuTemporarilyDisabled} for libraries (e.g. wllama)
     * that spawn workers which would otherwise re-enable the GPU backend.
     */
    async withWebGpuDisabledForWorkers(task) {
        const NativeWorker = window.Worker;
        let workerPatched = false;

        try {
            if (typeof NativeWorker === 'function') {
                window.Worker = class WorkerWithoutWebGPU extends NativeWorker {
                    constructor(scriptURL, options) {
                        let wrappedURL = scriptURL;
                        let createdWrappedBlobUrl = false;

                        try {
                            const workerType = options?.type === 'module' ? 'module' : 'classic';
                            const source = workerType === 'module'
                                ? `Object.defineProperty(self.navigator, 'gpu', { configurable: true, get: () => ({ requestAdapter: async () => null }) });\nimport ${JSON.stringify(String(scriptURL))};`
                                : `Object.defineProperty(self.navigator, 'gpu', { configurable: true, get: () => ({ requestAdapter: async () => null }) });\nimportScripts(${JSON.stringify(String(scriptURL))});`;

                            wrappedURL = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
                            createdWrappedBlobUrl = true;
                        } catch (error) {
                            wrappedURL = scriptURL;
                            createdWrappedBlobUrl = false;
                        }

                        super(wrappedURL, options);

                        if (createdWrappedBlobUrl) {
                            setTimeout(() => URL.revokeObjectURL(wrappedURL), 0);
                        }
                    }
                };

                workerPatched = true;
                console.log('Temporarily patched Worker to disable WebGPU inside wllama workers');
            }

            return await this.withWebGpuTemporarilyDisabled(task);
        } finally {
            if (workerPatched) {
                window.Worker = NativeWorker;
                console.log('Restored Worker after wllama initialization');
            }
        }
    }

    /**
     * Best-effort mobile detection used to skip GPU-model load on devices
     * that typically lack the memory/bandwidth for it.
     */
    isLikelyMobileDevice() {
        // Prefer userAgentData when available, then fall back to UA + touch heuristics.
        if (navigator.userAgentData && typeof navigator.userAgentData.mobile === 'boolean') {
            return navigator.userAgentData.mobile;
        }

        const ua = navigator.userAgent || '';
        const mobileUaPattern = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|EdgA/i;
        const isMobileUa = mobileUaPattern.test(ua);
        const isTouchMac = /Macintosh/i.test(ua) && (navigator.maxTouchPoints || 0) > 1;
        const hasTouch = (navigator.maxTouchPoints || 0) > 0 || ('ontouchstart' in window);
        const hasCoarsePointer = window.matchMedia?.('(pointer: coarse)').matches ?? false;
        const shortestScreenSide = Math.min(window.screen?.width || 0, window.screen?.height || 0);
        const isSmallScreen = shortestScreenSide > 0 && shortestScreenSide <= 900;
        const isTouchSmallScreen = hasTouch && hasCoarsePointer && isSmallScreen;

        return isMobileUa || isTouchMac || isTouchSmallScreen;
    }

    /**
     * Decide which inference engine to bring up at startup. Order of
     * preference: forced basic (debug) -> WebLLM/WebGPU (desktop only) ->
     * wllama/CPU -> basic (no model). Sets `this.availableModes` along
     * the way so the UI can offer manual mode switches later.
     */
    async initializeEngine() {
        // 🧪 DEBUG: Force Basic mode for testing
        if (this.debugConfig.enabled && this.debugConfig.forceBasicMode) {
            console.log('🧪 DEBUG: Forcing Basic mode');
            this.initializeBasicMode(
                'Ready to chat! (Basic mode)',
                '🧪 DEBUG: Running in forced Basic mode for testing.'
            );
            return;
        }

        const isMobileDevice = this.isLikelyMobileDevice();
        if (isMobileDevice) {
            this.availableModes.gpu = this.checkWebGPUSupport();
            console.log('Mobile device detected on startup. Defaulting to Basic mode.');
            document.body.classList.add('mobile-layout');
            this.initializeBasicMode(
                'Ready to chat! (Basic mode)',
                'Basic mode was selected automatically for mobile startup. You can switch modes anytime from the mode selector.'
            );
            return;
        }

        const hasWebGPU = this.checkWebGPUSupport();
        this.availableModes.gpu = hasWebGPU;

        // Create abort controller for model loading
        this.modelLoadingAbortController = new AbortController();

        if (hasWebGPU) {
            try {
                await this.initializeWebLLM();
                // Check if cancelled during initialization
                if (this.modelLoadingCancelled) {
                    return;
                }
                return;
            } catch (error) {
                // Check if cancelled during initialization
                if (this.modelLoadingCancelled) {
                    return;
                }
                console.log('WebLLM initialization failed, falling back to CPU mode');
                this.availableModes.gpu = false;
            }
        }

        try {
            await this.initializeWllama(null, {
                activateMode: true,
                showChatInterface: true,
                showFatalError: false
            });
            // Check if cancelled during initialization
            if (this.modelLoadingCancelled) {
                return;
            }
            return;
        } catch (error) {
            // Check if cancelled during initialization
            if (this.modelLoadingCancelled) {
                return;
            }
            console.log('CPU model initialization failed, falling back to Basic mode');
            this.availableModes.cpu = false;
        }

        this.initializeBasicMode(
            'Ready to chat! (Basic mode)',
            'Using Basic mode because the GPU and CPU models could not be loaded.'
        );
    }

    /**
     * Bring up the WebLLM (WebGPU) engine with the Phi-3.5 model and wire
     * its progress callback to the loading UI. Resolves once the model is
     * ready to chat; rejects if WebGPU init or model download fails.
     * @param {(p:number)=>void|null} [progressCallback] Optional callback for progress updates (used when loading from mode switch)
     * @param {{isLazyLoad?:boolean, activateMode?:boolean, showChatInterface?:boolean}} [options]
     */
    async initializeWebLLM(progressCallback = null, options = {}) {
        const { isLazyLoad = false, activateMode = true, showChatInterface = true } = options;

        try {
            if (!isLazyLoad) {
                this.updateProgress(15, 'Loading AI model (WebGPU)...');
            }

            // 🧪 DEBUG: Force WebGPU initialization failure for testing error handling
            if (this.debugConfig.enabled && this.debugConfig.forceWebGPUFail) {
                console.log('🧪 DEBUG: Forcing WebGPU initialization to fail (testing error handling)');
                await new Promise(resolve => setTimeout(resolve, 500)); // Simulate some initialization time
                throw new Error('DEBUG: Forced WebGPU initialization failure');
            }

            // Use Phi-3.5-mini with correct model library URL from WebLLM config
            const targetModelId = 'Phi-3.5-mini-instruct-q4f16_1-MLC';

            const appConfig = {
                model_list: [
                    {
                        model: 'https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC',
                        model_id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
                        model_lib: 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_83/base/Phi-3.5-mini-instruct-q4f16_1_cs1k-webgpu.wasm',
                        vram_required_MB: 3672.07,
                        low_resource_required: false,
                        overrides: {
                            context_window_size: 2048
                        }
                    }
                ]
            };

            this.engine = await webllm.CreateMLCEngine(
                targetModelId,
                {
                    appConfig: appConfig,
                    initProgressCallback: (progress) => {
                        // Check if user cancelled loading
                        if (this.modelLoadingCancelled) {
                            return;
                        }

                        const percentage = Math.max(15, Math.round(progress.progress * 85) + 15);

                        if (!isLazyLoad) {
                            this.updateProgress(
                                percentage,
                                `Loading model: ${Math.round(progress.progress * 100)}%`
                            );
                        } else {
                            console.log(`Loading WebLLM: ${Math.round(progress.progress * 100)}%`);
                            // Call the progress callback for lazy loading
                            if (progressCallback) {
                                progressCallback(progress.progress);
                            }
                        }
                    }
                }
            );

            // Check if cancelled before finalizing
            if (this.modelLoadingCancelled) {
                this.engine = null;
                throw new Error('Model loading cancelled by user');
            }

            if (!isLazyLoad) {
                this.updateProgress(100, 'Ready to chat!');
            }
            console.log('WebLLM engine initialized successfully');
            this.availableModes.gpu = true;

            if (activateMode) {
                this.setCurrentMode('gpu');
            }

            // Double-check cancellation before showing interface
            if (showChatInterface && !this.modelLoadingCancelled) {
                setTimeout(() => {
                    this.showChatInterface();
                }, 500);
            }

        } catch (error) {
            console.error('Failed to initialize WebLLM:', error);
            this.availableModes.gpu = false;
            throw error; // Re-throw to trigger fallback
        }
    }

    /**
     * Bring up the wllama (CPU/WASM) engine with the Phi-2 model. Reuses
     * an existing instance if one is already loaded.
     * @param {(p:number)=>void|null} [progressCallback] Forwarded download progress (0..1).
     * @param {{isLazyLoad?:boolean, activateMode?:boolean, showChatInterface?:boolean, showFatalError?:boolean}} [options]
     *   - isLazyLoad: hide progress UI when triggered by mode switch.
     *   - activateMode: set `currentMode='cpu'` on success.
     *   - showChatInterface: reveal the chat UI on success.
     *   - showFatalError: surface error via {@link showError} on failure.
     */
    async initializeWllama(progressCallback = null, options = {}) {
        const {
            activateMode = !this.availableModes.gpu,
            showChatInterface = !this.availableModes.gpu,
            showFatalError = false
        } = options;

        try {
            // Check if already initialized
            if (this.wllama) {
                console.log('Wllama already initialized');
                this.availableModes.cpu = true;
                if (activateMode) {
                    this.setCurrentMode('cpu');
                }
                return;
            }

            const isLazyLoad = progressCallback !== null || !showChatInterface;

            // 🧪 DEBUG: Force Wllama initialization failure for testing error handling
            if (this.debugConfig.enabled && this.debugConfig.forceWllamaFail) {
                console.log('🧪 DEBUG: Forcing Wllama initialization to fail (testing error handling)');
                if (!isLazyLoad) {
                    this.updateProgress(15, 'Loading AI model (CPU mode)...');
                }
                await new Promise(resolve => setTimeout(resolve, 500)); // Simulate some initialization time
                throw new Error('DEBUG: Forced Wllama initialization failure');
            }

            if (!isLazyLoad) {
                this.updateProgress(15, 'Loading AI model (CPU mode)...');
            }

            // Configure WASM paths for CDN
            const CONFIG_PATHS = {
                default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/wasm/wllama.wasm',
            };

            // Try multithreaded first if cross-origin isolated, fall back to single-threaded
            const useMultiThread = window.crossOriginIsolated === true;
            const availableThreads = navigator.hardwareConcurrency || 4; // Fallback to 4 if not available
            const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;
            console.log(`Cross-origin isolated: ${window.crossOriginIsolated}, available threads: ${availableThreads}, attempting ${preferredThreads} thread(s)`);

            const modelConfig = {
                n_ctx: 512,      // Minimum context size (512 is enforced by llama.cpp)
                n_gpu_layers: 0, // Force CPU-only: never use WebGPU even if available
                offload_kqv: false, // Keep K/Q/V cache on CPU to avoid WebGPU backend usage
                n_threads: preferredThreads,
                progressCallback: ({ loaded, total }) => {
                    // Check if user cancelled loading
                    if (this.modelLoadingCancelled) {
                        return;
                    }

                    const percentage = Math.min(100, Math.max(15, Math.round((loaded / total) * 85) + 15));
                    const progress = loaded / total;

                    if (!isLazyLoad) {
                        this.updateProgress(
                            percentage,
                            `Loading model: ${Math.round((loaded / total) * 100)}%`
                        );
                    } else {
                        console.log(`Loading wllama: ${Math.round((loaded / total) * 100)}%`);
                        // Call the progress callback for lazy loading
                        if (progressCallback) {
                            progressCallback(progress);
                        }
                    }
                }
            };

            await this.withWebGpuDisabledForWorkers(async () => {
                try {
                    // Initialize wllama with CDN-hosted WASM files
                    this.wllama = new Wllama(CONFIG_PATHS);

                    // Load model from HuggingFace with optimized settings

                    await this.wllama.loadModelFromHF(
                        {
                            repo: 'ngxson/wllama-split-models',
                            file: 'Phi-3.1-mini-128k-instruct-Q3_K_M-00001-of-00008.gguf'
                        },
                        {
                            ...modelConfig,
                            progressCallback: modelConfig.progressCallback
                        }
                    );

                    // Check if cancelled before finalizing
                    if (this.modelLoadingCancelled) {
                        this.wllama = null;
                        throw new Error('Model loading cancelled by user');
                    }

                    console.log(`Wllama initialized successfully with ${preferredThreads} thread(s)`);

                    // Update to final ready state if requested
                    if (!isLazyLoad) {
                        this.updateProgress(100, 'Ready to chat! (CPU mode)');
                    }
                } catch (multiErr) {
                    if (preferredThreads > 1) {
                        console.warn(`Multi-threaded init failed (${multiErr.message}), falling back to single thread`);

                        // Retry with single thread
                        this.wllama = new Wllama(CONFIG_PATHS);
                        await this.wllama.loadModelFromHF(
                            {
                                repo: 'ngxson/wllama-split-models',
                                file: 'Phi-3.1-mini-128k-instruct-Q3_K_M-00001-of-00008.gguf'
                            },
                            {
                                ...modelConfig,
                                n_threads: 1,
                                progressCallback: modelConfig.progressCallback
                            }
                        );

                        // Check if cancelled before finalizing (fallback path)
                        if (this.modelLoadingCancelled) {
                            this.wllama = null;
                            throw new Error('Model loading cancelled by user');
                        }

                        console.log('Wllama initialized successfully with 1 thread (fallback)');

                        // Update to final ready state if requested
                        if (!isLazyLoad) {
                            this.updateProgress(100, 'Ready to chat! (CPU mode)');
                        }
                    } else {
                        throw multiErr;
                    }
                }
            });
            console.log('Wllama initialized successfully with Phi 3.1');
            this.availableModes.cpu = true;

            // Check cancellation before activating and showing interface
            if (!this.modelLoadingCancelled) {
                if (activateMode) {
                    this.setCurrentMode('cpu');
                }

                if (showChatInterface) {
                    setTimeout(() => {
                        this.showChatInterface();
                    }, 500);
                }
            }

        } catch (error) {
            console.error('Failed to initialize wllama:', error);
            this.availableModes.cpu = false;
            if (showFatalError) {
                this.showError('Failed to load AI model. Please refresh the page.');
            }
            throw error;
        }
    }

    // ============================================================================
    // UI STATE MANAGEMENT
    // ============================================================================

    /** Update the loading progress bar and status text. */
    updateProgress(percentage, text) {
        // Don't update if loading was cancelled
        if (this.modelLoadingCancelled) {
            return;
        }

        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = text;

        // Show cancel link when loading starts (after initial knowledge base load)
        if (percentage >= 15 && percentage < 100 && this.elements.cancelLoadingLink) {
            this.elements.cancelLoadingLink.style.display = 'inline-block';
        } else if (percentage === 100 && this.elements.cancelLoadingLink) {
            this.elements.cancelLoadingLink.style.display = 'none';
        }

        // Update progress bar ARIA attributes
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.setAttribute('aria-label', text);
        }
    }

    /** Hide the loading screen and reveal the chat UI. */
    showChatInterface() {
        // Don't show interface if already showing due to cancellation
        if (this.elements.chatContainer.style.display === 'flex') {
            return;
        }

        this.elements.progressSection.style.display = 'none';
        this.elements.chatContainer.style.display = 'flex';
        this.updateModeSelector();
        this.elements.userInput.focus();
    }

    /**
     * Enter Basic mode (no model). Used as the final fallback when both
     * WebLLM and wllama fail to initialize, and when the user explicitly
     * selects Basic from the mode dropdown.
     */
    initializeBasicMode(progressText = 'Ready to chat! (Basic mode)', notice = null) {
        this.setCurrentMode('basic');

        // Temporarily allow progress update for final state
        const wasCancelled = this.modelLoadingCancelled;
        this.modelLoadingCancelled = false;
        this.updateProgress(100, progressText);

        // Restore cancelled state if it was set
        if (wasCancelled) {
            this.modelLoadingCancelled = true;
        }

        setTimeout(() => {
            this.showChatInterface();
            if (notice) {
                this.addSystemMessage(notice);
            }
        }, 500);
    }

    /**
     * Cancel the ongoing model loading and switch to Basic mode.
     * Called when the user clicks the "Cancel and start in Basic Mode" link
     * during model download.
     */
    cancelModelLoading() {
        // Set the cancellation flag immediately
        this.modelLoadingCancelled = true;

        // Abort any ongoing loading operations
        if (this.modelLoadingAbortController) {
            this.modelLoadingAbortController.abort();
        }

        // Clean up any partially loaded models
        if (this.engine) {
            this.engine = null;
        }
        if (this.wllama) {
            this.wllama = null;
        }

        // Hide the cancel link immediately
        if (this.elements.cancelLoadingLink) {
            this.elements.cancelLoadingLink.style.display = 'none';
        }

        // Don't mark modes as unavailable - user cancelled, not failed
        // They should still be able to select GPU/CPU modes later

        // Immediately switch to Basic mode
        this.initializeBasicMode(
            'Ready to chat! (Basic mode)',
            'Model loading was cancelled. You can switch modes anytime from the mode selector.'
        );
    }

    /** Set the active inference mode. @param {'gpu'|'cpu'|'basic'} mode */
    setCurrentMode(mode) {
        this.currentMode = mode;
    }

    /** Human-readable label for the mode selector and aria descriptions. */
    getModeLabel(mode = this.currentMode) {
        if (mode === 'gpu') {
            return 'Phi 3.5 (GPU)';
        }

        if (mode === 'cpu') {
            return 'Phi 3.1 (CPU)';
        }

        return 'None (Basic Q&A)';
    }

    /** Show an error in the loading section (red progress bar). */
    showError(message) {
        this.elements.progressText.textContent = message;
        this.elements.progressFill.style.backgroundColor = '#dc3545';
    }

    /** Disable input controls while a model is loading. */
    disableInput() {
        this.elements.userInput.disabled = true;
        this.elements.sendBtn.disabled = true;
        this.elements.micBtn.disabled = true;
        this.elements.userInput.placeholder = 'Loading model...';
    }

    /** Re-enable input controls and refocus the text field. */
    enableInput() {
        this.elements.userInput.disabled = false;
        this.elements.sendBtn.disabled = false;
        this.elements.micBtn.disabled = false;
        this.elements.userInput.placeholder = 'Ask a question about AI...';
        this.elements.userInput.focus();
    }

    // ============================================================================
    // EVENT LISTENERS
    // ============================================================================

    /** Bind all DOM event listeners. Called once from {@link initialize}. */
    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => {
            if (this.isGenerating) {
                this.stopGeneration();
            } else {
                this.sendMessage();
            }
        });

        // Input keyboard handling: Enter to send (Shift+Enter for newline), Escape to stop.
        // Consolidated to a single listener to avoid double-firing sendMessage.
        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.isGenerating) {
                    this.sendMessage();
                }
            } else if (e.key === 'Escape' && this.isGenerating) {
                this.stopGeneration();
            }
        });

        // Auto-resize textarea
        this.elements.userInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus input
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.elements.userInput.focus();
            }
            // Ctrl/Cmd + N for new chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                this.restartConversation();
            }
        });

        // Microphone button
        this.elements.micBtn.addEventListener('click', () => {
            this.handleMicClick();
        });

        // Restart button
        this.elements.restartBtn.addEventListener('click', () => {
            this.restartConversation();
        });

        // Mode selector
        this.elements.modeSelect.addEventListener('change', (event) => {
            this.switchMode(event.target.value);
        });

        // Cancel loading link
        if (this.elements.cancelLoadingLink) {
            // Click handler
            this.elements.cancelLoadingLink.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.cancelModelLoading();
            });

            // Keyboard handler for accessibility
            this.elements.cancelLoadingLink.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    this.cancelModelLoading();
                }
            });
        }

        // About button
        this.elements.aboutBtn.addEventListener('click', () => {
            this.lastFocusedElement = this.elements.aboutBtn;
            this.showAboutModal();
        });

        // About modal handlers
        this.elements.aboutModalClose.addEventListener('click', () => {
            this.hideAboutModal();
        });

        this.elements.aboutModalOk.addEventListener('click', () => {
            this.hideAboutModal();
        });

        // Close about modal on overlay click
        this.elements.aboutModal.addEventListener('click', (e) => {
            if (e.target === this.elements.aboutModal || e.target.classList.contains('modal-overlay')) {
                this.hideAboutModal();
            }
        });

        // Modal handlers
        this.elements.modalClose.addEventListener('click', () => {
            this.hideAiModeModal();
        });

        this.elements.modalOk.addEventListener('click', () => {
            this.hideAiModeModal();
        });

        // Close modal on overlay click
        this.elements.aiModeModal.addEventListener('click', (e) => {
            if (e.target === this.elements.aiModeModal || e.target.classList.contains('modal-overlay')) {
                this.hideAiModeModal();
            }
        });

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.elements.aiModeModal.style.display === 'flex') {
                    this.hideAiModeModal();
                } else if (this.elements.aboutModal.style.display === 'flex') {
                    this.hideAboutModal();
                }
            }
        });

        // Example question buttons
        const exampleBtns = document.querySelectorAll('.example-btn');
        exampleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.getAttribute('data-question');
                this.elements.userInput.value = question;
                this.elements.userInput.focus();
                this.autoResizeTextarea();
            });
        });

        this.elements.chatMessages.addEventListener('click', (e) => {
            const videoLink = e.target.closest('.video-link');
            if (!videoLink) {
                return;
            }

            e.preventDefault();
            this.openVideoPopup(videoLink.href);
        });

        this.elements.chatMessages.addEventListener('keydown', (e) => {
            if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('video-link')) {
                e.preventDefault();
                this.openVideoPopup(e.target.href);
            }
        });
    }

    // ============================================================================
    // CONTENT MODERATION & TEXT PROCESSING
    // ============================================================================

    /** Grow the user textarea to fit its content, up to its CSS max-height. */
    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    /**
     * Case-insensitive whole-word check against the (decoded) moderation
     * word list loaded by {@link loadProhibitedWords}.
     * @returns {boolean}
     */
    containsProhibitedWords(text) {
        const lowerText = text.toLowerCase();
        for (const { word, regex } of this.prohibitedPatterns) {
            if (regex.test(lowerText)) {
                console.log(`Content moderation: blocked word "${word}" detected`);
                return true;
            }
        }
        return false;
    }

    /** Lowercase + collapse whitespace + strip punctuation for search. */
    normalizeSearchText(text) {
        return text.toLowerCase().trim().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    }

    /**
     * Detect explicit web-search intent in a user message ("search for X",
     * "find docs about Y", "how do I ...") and return the keywords to
     * forward to a search engine, or null when no intent is detected.
     */
    getSearchIntentQuery(text) {
        const trimmedText = text.trim();
        const lowerText = trimmedText.toLowerCase();

        if (lowerText.startsWith('search ')) {
            return trimmedText.slice(7).trim();
        }

        if (lowerText.startsWith('find ')) {
            return trimmedText.slice(5).trim();
        }

        if (lowerText.includes('documentation') || lowerText.includes(' docs ') || lowerText.includes('microsoft learn ') || lowerText.includes('how to ') || lowerText.includes('how do i ') || lowerText.includes('how can i') || lowerText.includes(' me how ') || lowerText.includes('sample code') || lowerText.includes('example code') || lowerText.includes('code sample') || lowerText.includes('code example')) {
            return trimmedText;
        }

        return null;
    }

    /** Strip stop words and dedupe to a compact keyword string for Bing. */
    extractBingSearchKeywords(text) {
        const normalizedText = this.normalizeSearchText(text);
        const words = normalizedText.split(' ').filter(Boolean);
        const uniqueWords = [];
        const seenWords = new Set();

        words.forEach(word => {
            if (word.length < 2 || BING_STOP_WORDS.has(word) || seenWords.has(word)) {
                return;
            }

            seenWords.add(word);
            uniqueWords.push(word);
        });

        return uniqueWords.join(' ');
    }

    // ============================================================================
    // SEARCH & CONTEXT RETRIEVAL
    // ============================================================================

    /**
     * Match the user question against the knowledge-base keyword map
     * using 3-gram, 2-gram and unigram lookups, then drop keyword hits
     * that are subsets of longer ones (e.g. drop "language model" if
     * "large language model" also matched).
     * @returns {{matches:Array, matchedKeywords:string[]}}
     */
    performSearch(userQuestion) {
        const lowerQuestion = userQuestion.toLowerCase().trim();

        // Normalize the question: remove punctuation, extra spaces
        const normalizedQuestion = this.normalizeSearchText(lowerQuestion);
        const words = normalizedQuestion.split(' ');

        // Extract all n-grams (trigrams, bigrams, unigrams)
        const nGrams = [];

        // Trigrams (3-word phrases)
        for (let i = 0; i <= words.length - 3; i++) {
            nGrams.push({
                text: words.slice(i, i + 3).join(' '),
                length: 3
            });
        }

        // Bigrams (2-word phrases)
        for (let i = 0; i <= words.length - 2; i++) {
            nGrams.push({
                text: words.slice(i, i + 2).join(' '),
                length: 2
            });
        }

        // Unigrams (single words) - filter out very short words and common stop words
        const stopWords = SEARCH_STOP_WORDS;
        words.forEach(word => {
            if (word.length >= 2 && !stopWords.has(word)) {
                nGrams.push({
                    text: word,
                    length: 1
                });
            }
        });

        console.log('Extracted n-grams:', nGrams.map(ng => `"${ng.text}" (${ng.length})`));

        // Match n-grams to keywords in the index. Each keyword may map to
        // multiple documents; record a per-document match for each one.
        const matchedKeywords = new Set();
        const documentMatches = new Map(); // doc id -> {doc, category, link, matchedKeywords[]}

        nGrams.forEach(ngram => {
            const matches = this.keywordMap.get(ngram.text);
            if (!matches) return;
            matchedKeywords.add(ngram.text);

            matches.forEach(match => {
                const docId = match.document.id;
                if (!documentMatches.has(docId)) {
                    documentMatches.set(docId, {
                        document: match.document,
                        category: match.category,
                        link: match.link,
                        matchedKeywords: []
                    });
                }
                const matchRecord = documentMatches.get(docId);
                if (!matchRecord.matchedKeywords.includes(ngram.text)) {
                    matchRecord.matchedKeywords.push(ngram.text);
                }
            });
        });

        // Filter out keywords that are subsets of longer matched keywords
        // Example: if "large language model" is matched, remove "language model" and "language"
        const filteredKeywords = new Set();
        const sortedKeywords = Array.from(matchedKeywords).sort((a, b) => {
            const aWords = a.split(' ').length;
            const bWords = b.split(' ').length;
            return bWords - aWords; // Longer phrases first
        });

        sortedKeywords.forEach(keyword => {
            // Check if this keyword is a subset of any already-added keyword
            let isSubset = false;
            for (const existing of filteredKeywords) {
                if (existing !== keyword && existing.includes(keyword)) {
                    isSubset = true;
                    break;
                }
            }
            if (!isSubset) {
                filteredKeywords.add(keyword);
            }
        });

        console.log('Matched keywords (before filtering):', Array.from(matchedKeywords));
        console.log('Filtered keywords (after removing subsets):', Array.from(filteredKeywords));

        // Rebuild document matches using only filtered keywords
        const finalDocumentMatches = [];
        documentMatches.forEach((match, docId) => {
            // Only include if at least one of its keywords survived filtering
            const validKeywords = match.matchedKeywords.filter(kw => filteredKeywords.has(kw));
            if (validKeywords.length > 0) {
                finalDocumentMatches.push({
                    ...match,
                    matchedKeywords: validKeywords
                });
            }
        });

        console.log(`Found ${finalDocumentMatches.length} matching documents`);
        if (finalDocumentMatches.length > 0) {
            console.log('Matched documents:', finalDocumentMatches.map(m => ({
                id: m.document.id,
                title: m.document.title,
                category: m.category,
                keywords: m.matchedKeywords
            })));
        }

        return {
            matches: finalDocumentMatches,
            matchedKeywords: Array.from(filteredKeywords)
        };
    }

    /**
     * Build the retrieval context handed to the model: matching document
     * snippets, learn-more link metadata and any associated videos.
     * Returns `{context:null, ...}` when nothing matched.
     */
    searchContext(userQuestion) {
        const { matches, matchedKeywords } = this.performSearch(userQuestion);

        // If no matches, return null context
        if (matches.length === 0) {
            this.elements.searchStatus.textContent = '🔍 No specific context found';
            return { context: null, categories: [], links: [], documents: [], videos: [] };
        }

        // Rank documents by match quality (documents with longer/better keyword matches come first)
        const rankedMatches = matches.sort((a, b) => {
            // Calculate match quality score: sum of matched keyword lengths
            const aScore = a.matchedKeywords.reduce((sum, kw) => sum + kw.split(' ').length, 0);
            const bScore = b.matchedKeywords.reduce((sum, kw) => sum + kw.split(' ').length, 0);
            return bScore - aScore; // Higher score first
        });

        // Build context from all matched documents - use full content, no summarization
        const contextParts = rankedMatches.map(match => {
            return match.document.content;
        });

        const categories = [...new Set(rankedMatches.map(m => m.category))];
        const links = [...new Set(rankedMatches.map(m => m.link))];
        const documents = rankedMatches.map(m => m.document);
        const videos = documents.filter(doc => doc.video_id).map(doc => ({
            video_id: doc.video_id,
            title: doc.title
        }));

        this.elements.searchStatus.textContent = `🔍 Found context in: ${categories.join(', ')}`;

        return {
            context: contextParts.join('\n\n'),
            categories: categories,
            links: links,
            documents: documents,
            videos: videos
        };
    }

    // ============================================================================
    // MESSAGE HANDLING & RESPONSE GENERATION
    // ============================================================================

    /**
     * Handle a user submission end-to-end: validate, moderate, search the
     * knowledge base, then dispatch to the search-link, no-results or
     * model-generation response path depending on intent and matches.
     */
    async sendMessage() {
        const userMessage = this.elements.userInput.value.trim();

        // Validate input
        if (!userMessage || this.isGenerating || this.isLoadingModel || this.isStoppingGeneration) return;

        // Limit message length to prevent abuse
        const MAX_MESSAGE_LENGTH = 1000;
        if (userMessage.length > MAX_MESSAGE_LENGTH) {
            this.addSystemMessage(`Message too long. Please keep it under ${MAX_MESSAGE_LENGTH} characters.`);
            return;
        }

        // Store voice input flag before any processing
        const usedVoice = this.usedVoiceInput;
        this.usedVoiceInput = false;

        // Content moderation: check for prohibited words (whole words only)
        if (this.containsProhibitedWords(userMessage)) {
            // Clear input and reset height
            this.elements.userInput.value = '';
            this.elements.userInput.style.height = 'auto';

            // Add user message to chat
            this.addMessage('user', userMessage);

            // Play audio if voice input was used
            if (usedVoice) {
                this.playModerationAudio();
            }

            // Add moderation response
            this.addMessage('assistant', "I'm sorry, I can't help with that because it triggered a content-safety filtering policy. I can only help with information about AI and computing.");
            this.elements.userInput.focus();
            return;
        }

        // Check if wllama is still loading when in CPU mode
        if (this.currentMode === 'cpu' && !this.wllama) {
            this.addSystemMessage('CPU mode is still loading. Please wait...');
            return;
        }

        // Clear input and reset height
        this.elements.userInput.value = '';
        this.elements.userInput.style.height = 'auto';

        // Add user message to chat
        this.addMessage('user', userMessage);

        // Check if this is an initial greeting (only if no messages yet)
        const messageCount = this.elements.chatMessages.querySelectorAll('.message').length;
        if (messageCount <= 1) { // Only user's message is in chat
            const greetingPattern = /^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\s!?]*$/i;
            if (greetingPattern.test(userMessage)) {
                // Respond with greeting without searching
                const greetingResponse = "Hello, I'm Anton. I'm here to help you learn about AI concepts. What would you like to know?";
                this.addMessage('assistant', greetingResponse);
                this.elements.userInput.focus();
                return;
            }
        }

        const searchQuery = this.getSearchIntentQuery(userMessage);
        if (searchQuery) {
            await this.respondWithSearchLink(userMessage, searchQuery, usedVoice);
            return;
        }

        // Search for relevant context
        const searchResult = this.searchContext(userMessage);

        // If no results found, provide Microsoft Learn search link
        if (!searchResult.context || searchResult.documents.length === 0) {
            await this.respondWithNoResultsSearchLink(userMessage, usedVoice);
            return;
        }

        // Generate response
        await this.generateResponse(userMessage, searchResult, usedVoice);
    }

    /** Toggle the send button between its "Send" and "Stop" affordances. */
    updateSendButton(isGenerating) {
        const sendIcon = this.elements.sendBtn.querySelector('.send-icon');
        if (isGenerating) {
            sendIcon.textContent = '■';
            this.elements.sendBtn.title = 'Stop generation';
            this.elements.sendBtn.setAttribute('aria-label', 'Stop generation');
        } else {
            sendIcon.textContent = '▶';
            this.elements.sendBtn.title = 'Send message';
            this.elements.sendBtn.setAttribute('aria-label', 'Send message');
        }
    }

    /**
     * Cooperative cancel for the in-flight response. Signals the active
     * stream, aborts any HTTP request, and asks the engine to interrupt
     * generation; per-engine cleanup runs in `safeStop*` helpers.
     */
    stopGeneration() {
        this.stopRequested = true;

        // Abort the generation properly using AbortController
        if (this.currentAbortController) {
            console.log('Aborting generation via AbortController');
            this.currentAbortController.abort();
            this.currentAbortController = null;
        }

        const activeStream = this.currentStream;

        // WebLLM can keep an internal lock after interruption unless the stream is
        // explicitly drained. This prevents the next prompt from hanging.
        if (activeStream && this.currentMode === 'gpu' && this.engine) {
            this.isStoppingGeneration = true;
            this.safeStopWebLLMStream(activeStream)
                .catch((error) => {
                    console.warn('WebLLM stop cleanup failed:', error);
                })
                .finally(() => {
                    this.isStoppingGeneration = false;
                    if (this.currentStream === activeStream) {
                        this.currentStream = null;
                    }
                });
        } else if (activeStream && this.currentMode === 'cpu') {
            this.isStoppingGeneration = true;
            this.safeStopWllamaStream(activeStream)
                .catch((error) => {
                    console.warn('Wllama stop cleanup failed:', error);
                })
                .finally(() => {
                    this.isStoppingGeneration = false;
                    if (this.currentStream === activeStream) {
                        this.currentStream = null;
                    }
                });
        } else {
            this.currentStream = null;
        }

        this.updateSendButton(false);
        console.log('Stop requested');
    }

    /**
     * Drain a wllama stream after interruption so its WASM state is left
     * in a clean condition for the next prompt. Safe to call with a null
     * or already-closed stream.
     */
    async safeStopWllamaStream(stream) {
        if (!stream) {
            return;
        }

        for (let i = 0; i < 2; i++) {
            try {
                const nextPromise = stream.next?.();
                if (!nextPromise || typeof nextPromise.then !== 'function') {
                    break;
                }

                await Promise.race([
                    nextPromise,
                    new Promise((resolve) => setTimeout(resolve, 120))
                ]);
            } catch (error) {
                break;
            }
        }

        try {
            if (typeof stream.return === 'function') {
                await stream.return();
            }
        } catch (error) {
            console.warn('Stream return failed during CPU stop cleanup:', error);
        }

        // Try to reset state immediately
        try {
            await this.resetWllamaInterruptState();
            console.log('Wllama state reset successfully after stop');
        } catch (error) {
            console.warn('Error during state reset:', error.message || error.toString());
        }
    }

    /**
     * Clear lingering interrupt flags inside wllama after a stop so the
     * next prompt isn't immediately cancelled.
     */
    async resetWllamaInterruptState() {
        if (!this.wllama) {
            return;
        }

        // Wllama state is reset automatically after interruption
        console.log('Wllama interrupt state cleared');
    }

    /**
     * Drain a WebLLM stream after interruption and release any pipeline
     * locks so the engine accepts the next prompt.
     */
    async safeStopWebLLMStream(stream) {
        if (!this.engine || !stream) {
            return;
        }

        try {
            if (typeof this.engine.interruptGenerate === 'function') {
                await this.engine.interruptGenerate();
            }
        } catch (error) {
            console.warn('engine.interruptGenerate failed:', error);
        }

        // Workaround for WebLLM lock not always being released immediately.
        for (let i = 0; i < 3; i++) {
            try {
                const nextPromise = stream.next?.();
                if (!nextPromise || typeof nextPromise.then !== 'function') {
                    break;
                }

                await Promise.race([
                    nextPromise,
                    new Promise((resolve) => setTimeout(resolve, 150))
                ]);
            } catch (error) {
                break;
            }
        }

        try {
            if (typeof stream.return === 'function') {
                await stream.return();
            }
        } catch (error) {
            console.warn('Stream return failed during stop cleanup:', error);
        }

        await this.resetWebLLMInterruptState();
    }

    /**
     * Reset internal WebLLM state (interrupt signal, pipeline locks, KV
     * cache) after a cancelled generation.
     */
    async resetWebLLMInterruptState() {
        if (!this.engine) {
            return;
        }

        // Defensive reset for known WebLLM interruption edge cases.
        if (Object.prototype.hasOwnProperty.call(this.engine, 'interruptSignal')) {
            this.engine.interruptSignal = false;
        }

        const lockMap = this.engine.loadedModelIdToLock;
        if (lockMap && typeof lockMap.values === 'function') {
            for (const lock of lockMap.values()) {
                if (lock && lock.acquired && typeof lock.release === 'function') {
                    try {
                        await lock.release();
                    } catch (error) {
                        console.warn('Failed to release WebLLM lock:', error);
                    }
                }
            }
        }

        if (typeof this.engine.resetChat === 'function') {
            try {
                await this.engine.resetChat();
            } catch (error) {
                console.warn('engine.resetChat failed after interruption:', error);
            }
        }
    }

    /**
     * User-driven mode change. Lazily initializes wllama the first time
     * CPU mode is selected and shows a transient loading indicator.
     * @param {'gpu'|'cpu'|'basic'} targetMode
     */
    async switchMode(targetMode) {
        if (targetMode === this.currentMode) {
            this.updateModeSelector();
            return;
        }

        // Reset cancellation flag when user tries to load a model again
        if ((targetMode === 'gpu' || targetMode === 'cpu') && this.modelLoadingCancelled) {
            this.modelLoadingCancelled = false;
        }

        if (targetMode === 'gpu') {
            // If GPU is available but engine not loaded, try to load it
            if (this.availableModes.gpu && !this.engine) {
                this.isLoadingModel = true;
                this.elements.modeSelect.disabled = true;
                this.disableInput();
                const loadingMsg = this.addSystemMessage('Switching to GPU mode - loading model... 0%');
                const loadingMsgElement = loadingMsg.querySelector('p');

                try {
                    // Create new abort controller for this load attempt
                    this.modelLoadingAbortController = new AbortController();

                    await this.initializeWebLLM((progress) => {
                        if (loadingMsgElement) {
                            const percentage = Math.round(progress * 100);
                            loadingMsgElement.textContent = `Switching to GPU mode - loading model... ${percentage}%`;
                        }
                    }, {
                        isLazyLoad: true,
                        activateMode: true,
                        showChatInterface: false
                    });

                    if (loadingMsgElement) {
                        loadingMsgElement.textContent = 'Switched to GPU mode';
                    }
                } catch (error) {
                    console.error('Failed to load GPU mode:', error);
                    this.availableModes.gpu = false;

                    const fallbackMode = this.availableModes.cpu && this.wllama ? 'cpu' : 'basic';
                    this.setCurrentMode(fallbackMode);

                    if (loadingMsgElement) {
                        loadingMsgElement.textContent = `Failed to load GPU mode. Switched to ${this.getModeLabel(fallbackMode)} mode.`;
                    }
                } finally {
                    this.updateModeSelector();
                    this.isLoadingModel = false;
                    this.enableInput();
                    this.elements.modeSelect.disabled = false;
                }
                return;
            }

            // If GPU not available, show modal
            if (!this.availableModes.gpu) {
                this.updateModeSelector();
                this.lastFocusedElement = document.activeElement;
                this.showAiModeModal();
                return;
            }

            // GPU mode already loaded, just switch to it
            this.disableInput();
            this.setCurrentMode('gpu');
            this.updateModeSelector();
            this.addSystemMessage('Switched to GPU mode');
            this.enableInput();
            return;
        }

        if (targetMode === 'basic') {
            this.disableInput();
            this.setCurrentMode('basic');
            this.updateModeSelector();
            this.addSystemMessage('Switched to Basic mode');
            this.enableInput();
            return;
        }

        if (!this.availableModes.cpu) {
            this.updateModeSelector();
            this.addSystemMessage('CPU mode is unavailable on this device.');
            return;
        }

        if (this.wllama) {
            this.disableInput();
            this.setCurrentMode('cpu');
            this.updateModeSelector();
            this.addSystemMessage('Switched to CPU mode');
            this.enableInput();
            return;
        }

        this.isLoadingModel = true;
        this.elements.modeSelect.disabled = true;
        this.disableInput();
        const loadingMsg = this.addSystemMessage('Switching to CPU mode - loading model... 0% (first-time download may take a few minutes)');
        const loadingMsgElement = loadingMsg.querySelector('p');

        try {
            await this.initializeWllama((progress) => {
                if (loadingMsgElement) {
                    const percentage = Math.round(progress * 100);
                    loadingMsgElement.textContent = `Switching to CPU mode - loading model... ${percentage}% (first-time download may take a few minutes)`;
                }
            }, {
                activateMode: true,
                showChatInterface: false,
                showFatalError: false
            });

            if (loadingMsgElement) {
                loadingMsgElement.textContent = 'Switched to CPU mode';
            }
        } catch (error) {
            console.error('Failed to load CPU mode:', error);
            this.availableModes.cpu = false;

            const fallbackMode = this.availableModes.gpu && this.engine ? 'gpu' : 'basic';
            this.setCurrentMode(fallbackMode);

            if (loadingMsgElement) {
                loadingMsgElement.textContent = `Failed to load CPU mode. Switched to ${this.getModeLabel(fallbackMode)} mode.`;
            }
        } finally {
            this.updateModeSelector();
            this.isLoadingModel = false;
            this.enableInput();
            this.elements.modeSelect.disabled = false;
        }
    }

    /** Sync the <select> options and labels with `availableModes`/`currentMode`. */
    updateModeSelector() {
        const { modeSelect } = this.elements;
        if (!modeSelect) {
            return;
        }

        const modeIcons = {
            gpu: { enabled: '🟢', disabled: '◯' },
            cpu: { enabled: '🟠', disabled: '◯' },
            basic: { enabled: '⚪', disabled: '◯' }
        };

        Array.from(modeSelect.options).forEach(option => {
            const mode = option.value;
            const isAvailable = mode === 'basic' ? true : this.availableModes[mode];
            const icon = isAvailable ? modeIcons[mode].enabled : modeIcons[mode].disabled;
            option.textContent = `${icon} ${this.getModeLabel(mode)}`;
            option.disabled = !isAvailable;
        });

        modeSelect.value = this.currentMode;

        const modeDescriptions = {
            gpu: 'GPU mode uses WebLLM with Phi-3-mini.',
            cpu: 'CPU mode uses wllama with Phi-2.',
            basic: 'Basic mode returns matching content directly from the knowledge base.'
        };

        modeSelect.title = `Current mode: ${this.getModeLabel()}. ${modeDescriptions[this.currentMode]}`;
        modeSelect.setAttribute('aria-label', `Choose a response mode. Current mode: ${this.getModeLabel()}.`);
    }

    // ============================================================================
    // MESSAGE UI RENDERING
    // ============================================================================

    /**
     * Build the HTML for the "Anton is typing" indicator, with a mode-specific
     * note appended in CPU and Basic modes. Used to seed an assistant message
     * bubble before the real response is streamed/animated into it.
     * @param {('gpu'|'cpu'|'basic')} [mode=this.currentMode] Mode whose note to show.
     * @returns {string} HTML string to assign to `messageTextDiv.innerHTML`.
     */
    getTypingIndicatorHtml(mode = this.currentMode) {
        const dots = '<span class="typing-indicator" aria-label="Anton is typing">●●●</span>';
        if (mode === 'cpu') {
            return `${dots}<p style="${TYPING_NOTE_STYLE}">(Responses may be slow in CPU mode. Thanks for your patience!)</p>`;
        }
        if (mode === 'basic') {
            return `${dots}<p style="${TYPING_NOTE_STYLE}">(Basic mode returns matching knowledge-base content without model inference.)</p>`;
        }
        return '<span class="typing-indicator">●●●</span>';
    }

    /**
     * Schedule the "Searching..." hint to disappear after a short delay so the
     * user has time to read it once a response finishes.
     */
    scheduleSearchStatusClear() {
        setTimeout(() => {
            this.elements.searchStatus.textContent = '';
        }, SEARCH_STATUS_CLEAR_DELAY);
    }

    /** Append a status/system bubble (e.g. "Switched to Basic mode"). */
    addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        messageDiv.setAttribute('role', 'status');
        messageDiv.setAttribute('aria-live', 'polite');
        // Sanitize message to prevent XSS
        const p = document.createElement('p');
        p.textContent = message;
        messageDiv.appendChild(p);
        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }

    /**
     * Create and append a chat-message bubble for the given role.
     * @param {'assistant'|'user'} role
     * @param {string} content Raw text (escaped before insertion).
     * @param {boolean} [isTyping] If true, seed the bubble with the typing indicator.
     * @returns {HTMLElement} The created message element.
     */
    addMessage(role, content, isTyping = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        messageDiv.setAttribute('role', 'article');
        messageDiv.setAttribute('aria-label', `Message from ${role === 'assistant' ? 'Anton' : 'You'}`);

        if (role === 'assistant') {
            messageDiv.innerHTML = `
                <div class="avatar anton-avatar" aria-hidden="true">
                    <img src="images/anton-icon.png" alt="Anton the AI assistant avatar" class="avatar-image">
                </div>
                <div class="message-content">
                    <p class="message-author" aria-label="From Anton">Anton</p>
                    <div class="message-text" ${isTyping ? 'aria-live="polite" aria-busy="true"' : ''}>
                        ${isTyping
                    ? '<span class="typing-indicator" aria-label="Anton is typing">●●●</span>'
                    : this.escapeHtml(content)}
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <p class="message-author" aria-label="From You">You</p>
                    <div class="message-text">${this.escapeHtml(content)}</div>
                </div>
                <div class="avatar user-avatar" aria-hidden="true">👤</div>
            `;
        }

        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    /**
     * Finalize an assistant message bubble: append any video and
     * "Learn more" link sections, then substitute placeholder tokens
     * (e.g. `[[VIDEO_LINK_0]]`, `[[SEARCH_RESULT_LINK]]`) with HTML.
     */
    renderAssistantMessage(messageTextDiv, assistantMessage, categories = [], links = [], videos = [], placeholders = {}) {
        let displayMessage = assistantMessage;

        // Add video links if available (before Learn more links)
        if (videos && videos.length > 0) {
            if (videos.length === 1) {
                displayMessage += '\n\nWatch this video for more details: [[VIDEO_LINK_0]]';
            } else {
                displayMessage += '\n\nThese videos might provide more information:\n[[VIDEO_LINKS]]';
            }
        }

        // Add learn more links after videos
        if (links && links.length > 0 && categories && categories.length > 0) {
            displayMessage += '\n\n---\n\n**Learn more:** [[LEARN_MORE_LINKS]]';
        }

        let formattedMessage = this.formatResponse(displayMessage);

        // Replace video links - popup window avoids the blocked iframe embed path
        if (videos && videos.length > 0) {
            if (videos.length === 1) {
                const video = videos[0];
                const videoUrl = this.getSynthesiaVideoUrl(video.video_id);
                const videoLinkHtml = `<a href="${videoUrl}" target="_blank" rel="noopener noreferrer" class="video-link">${this.escapeHtml(video.title)}</a>`;
                formattedMessage = formattedMessage.replace(/\[\[VIDEO_LINK_0\]\]/g, videoLinkHtml);
            } else {
                const videoLinksHtml = videos.map(video => {
                    const videoUrl = this.getSynthesiaVideoUrl(video.video_id);
                    return `• <a href="${videoUrl}" target="_blank" rel="noopener noreferrer" class="video-link">${this.escapeHtml(video.title)}</a>`;
                }).join('<br>');
                formattedMessage = formattedMessage.replace(/\[\[VIDEO_LINKS\]\]/g, videoLinksHtml);
            }
        }

        if (links && links.length > 0 && categories && categories.length > 0) {
            const linkHtml = links.map((link, index) => {
                const categoryName = categories[Math.min(index, categories.length - 1)];
                return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
            }).join(' • ');
            formattedMessage = formattedMessage.replace(/\[\[LEARN_MORE_LINKS\]\]/g, linkHtml);
        }

        Object.entries(placeholders).forEach(([placeholder, replacement]) => {
            formattedMessage = formattedMessage.split(placeholder).join(replacement);
        });

        messageTextDiv.innerHTML = formattedMessage;
        this.scrollToBottom();
    }

    /** @returns {boolean} true if at least one user turn exists in history AND the previous response wasn't a "no results" message. */
    hasPreviousUserPrompt() {
        const userMessages = this.elements.chatMessages.querySelectorAll('.user-message');
        if (userMessages.length <= 1) {
            return false;
        }

        // Check if the last assistant message was the standard "no results" message
        const assistantMessages = this.elements.chatMessages.querySelectorAll('.assistant-message');
        if (assistantMessages.length > 0) {
            const lastAssistantMessage = assistantMessages[assistantMessages.length - 1];
            const messageText = lastAssistantMessage.querySelector('.message-text');
            if (messageText) {
                const text = messageText.textContent || messageText.innerText || '';
                // Check for the standard "no results" message text
                if (text.includes("I don't have any information about that specific topic")) {
                    return false;
                }
            }
        }

        return true;
    }

    // === Microsoft Learn MCP server integration ============================
    // Mirrors the streamable HTTP client used by learn-mcp-client: lazy
    // initialize → tools/list → tools/call, parses the returned JSON envelope,
    // and returns up to `max` deduplicated {title, url} article references.

    /**
     * Query the Microsoft Learn MCP server for documentation links
     * relevant to a user question. Best-effort: returns `[]` on any error.
     * @param {string} query Search text.
     * @param {number} [max] Maximum number of dedup'd links to return.
     */
    async queryLearnMcp(query, max = 5) {
        if (!query || !query.trim()) return [];
        const tool = await this.ensureLearnMcpReady();
        if (!tool) return [];

        const args = this.buildLearnMcpArgs(tool, query.trim());
        const result = await this.mcpRpc('tools/call', { name: tool.name, arguments: args });

        const items = this.extractLearnMcpItems(result);
        const links = [];
        const seenBase = new Set();
        for (const item of items) {
            const url = item.contentUrl || item.url || item.uri || item.link || '';
            if (!url) continue;
            const base = url.split('#')[0];
            if (seenBase.has(base)) continue;
            seenBase.add(base);
            const title = item.title || item.name || item.heading || base;
            links.push({ title, url: base });
            if (links.length >= max) break;
        }
        return links;
    }

    /**
     * Lazily initialize the Learn MCP connection (initialize + tools/list)
     * and cache the discovered search tool descriptor. Returns the tool or
     * null if MCP is unreachable.
     */
    async ensureLearnMcpReady() {
        if (!this._mcp) {
            this._mcp = {
                endpoint: 'https://learn.microsoft.com/api/mcp',
                protocolVersion: '2025-06-18',
                sessionId: null,
                nextId: 1,
                tool: null,
                initPromise: null,
            };
        }
        if (this._mcp.tool) return this._mcp.tool;
        if (!this._mcp.initPromise) {
            this._mcp.initPromise = (async () => {
                await this.mcpRpc('initialize', {
                    protocolVersion: this._mcp.protocolVersion,
                    capabilities: {},
                    clientInfo: { name: 'ask-anton', version: '0.1.0' },
                });
                await this.mcpRpc('notifications/initialized', undefined, { isNotification: true });
                const listed = await this.mcpRpc('tools/list', {});
                const tools = (listed && listed.tools) || [];
                this._mcp.tool = tools.find(t => /search/i.test(t.name)) || tools[0] || null;
                return this._mcp.tool;
            })().catch(err => {
                // Reset so a later question can retry.
                this._mcp.initPromise = null;
                throw err;
            });
        }
        return this._mcp.initPromise;
    }

    /** Map a free-form query into the MCP tool's expected input shape. */
    buildLearnMcpArgs(tool, query) {
        const props = (tool.inputSchema && tool.inputSchema.properties) || {};
        const candidates = ['query', 'question', 'q', 'search', 'searchQuery', 'text', 'prompt'];
        const key = candidates.find(k => k in props) || Object.keys(props)[0];
        const args = {};
        if (key) args[key] = query;
        return args;
    }

    /**
     * Pull search-result items out of an MCP tool/call response. Handles
     * both plain-array and `{results|items|data|value|hits|documents}`
     * envelope payloads, parsing JSON text parts as needed.
     */
    extractLearnMcpItems(result) {
        const items = [];
        const parts = (result && result.content) || [];
        for (const part of parts) {
            if (part.type !== 'text' || typeof part.text !== 'string') continue;
            const trimmed = part.text.trim();
            let parsed = null;
            if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
                try { parsed = JSON.parse(trimmed); } catch { /* not JSON */ }
            }
            if (parsed && !Array.isArray(parsed) && typeof parsed === 'object') {
                for (const key of ['results', 'items', 'data', 'value', 'hits', 'documents']) {
                    if (Array.isArray(parsed[key])) { parsed = parsed[key]; break; }
                }
            }
            if (Array.isArray(parsed)) items.push(...parsed);
            else if (parsed && typeof parsed === 'object') items.push(parsed);
        }
        return items;
    }

    /**
     * Send a JSON-RPC request over the MCP transport.
     * @param {string} method
     * @param {object} params
     * @param {{isNotification?:boolean}} [opts] When true, no id is sent and no response is awaited.
     */
    async mcpRpc(method, params, { isNotification = false } = {}) {
        const id = isNotification ? undefined : this._mcp.nextId++;
        const body = { jsonrpc: '2.0', method, ...(params !== undefined ? { params } : {}) };
        if (!isNotification) body.id = id;

        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'MCP-Protocol-Version': this._mcp.protocolVersion,
        };
        if (this._mcp.sessionId) headers['Mcp-Session-Id'] = this._mcp.sessionId;

        const res = await fetch(this._mcp.endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
        });

        const sid = res.headers.get('Mcp-Session-Id') || res.headers.get('mcp-session-id');
        if (sid) this._mcp.sessionId = sid;

        if (isNotification) {
            if (!res.ok && res.status !== 202) {
                throw new Error(`MCP notification ${method} failed: ${res.status}`);
            }
            return null;
        }

        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(`MCP ${method} failed: ${res.status} ${res.statusText} ${text}`);
        }

        const ct = (res.headers.get('Content-Type') || '').toLowerCase();
        if (ct.includes('text/event-stream')) {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let idx;
                while ((idx = buffer.indexOf('\n\n')) !== -1) {
                    const evt = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    const data = evt.split('\n')
                        .filter(l => l.startsWith('data:'))
                        .map(l => l.slice(5).trimStart())
                        .join('\n');
                    if (!data) continue;
                    let msg;
                    try { msg = JSON.parse(data); } catch { continue; }
                    if (msg.id === id) {
                        if (msg.error) throw new Error(`MCP ${method}: ${msg.error.message}`);
                        return msg.result;
                    }
                }
            }
            throw new Error(`MCP ${method}: stream ended without a response`);
        }

        const msg = await res.json();
        if (msg.error) throw new Error(`MCP ${method}: ${msg.error.message}`);
        return msg.result;
    }
    // === end MCP integration ===============================================

    /**
     * Reply to an explicit "search for X" intent with a curated link
     * (Learn MCP results when available, otherwise a Bing fallback).
     */
    async respondWithSearchLink(userMessage, searchQuery, usedVoiceInput = false) {
        const searchResult = this.searchContext(searchQuery);
        const bingKeywords = this.extractBingSearchKeywords(searchQuery) || this.normalizeSearchText(searchQuery);
        const encodedKeywords = encodeURIComponent(bingKeywords);
        const bingUrl = `https://learn.microsoft.com/en-us/search/?terms=${encodedKeywords}&category=Documentation`;

        // Try the Microsoft Learn MCP server first to get specific article links.
        let mcpLinks = [];
        try {
            mcpLinks = await this.queryLearnMcp(searchQuery, 5);
        } catch (err) {
            console.warn('Learn MCP query failed, falling back to search link:', err);
        }

        const useMcp = mcpLinks.length > 0;
        const historyAssistantMessage = useMcp
            ? `OK, I searched the Microsoft Learn documentation for "${bingKeywords}".\nCheck out the following documentation articles:`
            : `OK, I searched the Microsoft Learn documentation for "${bingKeywords}".\nHere's what I found.`;
        const assistantMessage = useMcp
            ? historyAssistantMessage.replace('Check out the following documentation articles:', '[[SEARCH_RESULT_LINK]]')
            : historyAssistantMessage.replace("Here's what I found.", '[[SEARCH_RESULT_LINK]]');

        const searchLinkHtml = useMcp
            ? 'Check out the following documentation articles:<ul class="mcp-results">' +
            mcpLinks.map(l =>
                `<li><a href="${this.escapeHtml(l.url)}" target="_blank" rel="noopener noreferrer">${this.escapeHtml(l.title)}</a></li>`
            ).join('') +
            '</ul>'
            : `<a href="${bingUrl}" target="_blank" rel="noopener noreferrer">Here's what I found.</a>`;

        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);

        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');

        messageTextDiv.innerHTML = this.getTypingIndicatorHtml();

        try {
            await new Promise(resolve => setTimeout(resolve, 250));

            if (usedVoiceInput) {
                this.playSearchResultsAudio();
            }

            const animationCompleted = await this.animateTyping(
                messageTextDiv,
                historyAssistantMessage,
                (partialMessage) => this.formatResponse(partialMessage),
                25
            );

            if (!animationCompleted) {
                return;
            }

            this.renderAssistantMessage(
                messageTextDiv,
                assistantMessage,
                searchResult.categories,
                searchResult.links,
                searchResult.videos || [],
                { '[[SEARCH_RESULT_LINK]]': searchLinkHtml }
            );

            this.conversationHistory.push({
                role: 'user',
                content: userMessage
            });
            this.conversationHistory.push({
                role: 'assistant',
                content: historyAssistantMessage
            });
        } finally {
            this.isGenerating = false;
            this.stopRequested = false;
            this.updateSendButton(false);
            this.elements.userInput.focus();

            this.scheduleSearchStatusClear();
        }
    }

    /**
     * Reply when the knowledge-base search returned nothing. May try a
     * brief contextual model continuation first (if a prior turn exists)
     * and otherwise offers a Bing search link.
     */
    async respondWithNoResultsSearchLink(userMessage, usedVoiceInput = false) {
        const bingKeywords = this.extractBingSearchKeywords(userMessage) || this.normalizeSearchText(userMessage);
        const encodedKeywords = encodeURIComponent(bingKeywords);
        const bingUrl = `https://www.bing.com/search?q=${encodedKeywords}`;
        const historyAssistantMessage = `I don't have any information about that specific topic; but you may find what you're looking for here.\n\nAsk me about AI-related topics and I'll do my best to help!`;
        const assistantMessage = historyAssistantMessage.replace('here.', 'here: [[SEARCH_RESULT_LINK]].');
        const shouldTryConversationFallback = (this.currentMode === 'gpu' || this.currentMode === 'cpu') && this.hasPreviousUserPrompt();
        const fallbackNote = '\n\n*Note: You can ask me to "Search for details about {X}" or "Find documentation for {Y}" to look for information about Microsoft AI technologies in Microsoft Learn.';

        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);

        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');
        const searchLinkHtml = `<a href="${bingUrl}" target="_blank" rel="noopener noreferrer">Bing search results</a>`;

        messageTextDiv.innerHTML = this.getTypingIndicatorHtml();

        try {
            await new Promise(resolve => setTimeout(resolve, 250));

            if (shouldTryConversationFallback) {
                let modelResponse = '';

                if (this.currentMode === 'gpu') {
                    modelResponse = await this.generateWithWebLLM(userMessage, null, messageTextDiv, usedVoiceInput);
                } else if (this.currentMode === 'cpu') {
                    modelResponse = await this.generateWithWllama(userMessage, null, messageTextDiv, usedVoiceInput);
                }

                if (this.stopRequested) {
                    return;
                }

                if (modelResponse.trim()) {
                    const displayedMessage = `${modelResponse.trim()}${fallbackNote}`;
                    messageTextDiv.innerHTML = this.formatResponse(displayedMessage);

                    this.conversationHistory.push({
                        role: 'user',
                        content: userMessage
                    });
                    this.conversationHistory.push({
                        role: 'assistant',
                        content: modelResponse.trim()
                    });
                    return;
                }

                if (this.currentMode === 'gpu' && this.lastWebLLMCompletionErrored) {
                    messageTextDiv.innerHTML = this.formatResponse(this.gpuModeFailureMessage);
                    return;
                }

                if (this.currentMode === 'cpu' && this.lastWllamaCompletionErrored) {
                    messageTextDiv.innerHTML = this.formatResponse(this.cpuModeFailureMessage);
                    return;
                }
            }

            if (usedVoiceInput) {
                this.playNoResultsAudio();
            }

            const animationCompleted = await this.animateTyping(
                messageTextDiv,
                historyAssistantMessage,
                (partialMessage) => this.formatResponse(partialMessage),
                25
            );

            if (!animationCompleted) {
                return;
            }

            this.renderAssistantMessage(
                messageTextDiv,
                assistantMessage,
                [],
                [],
                [],
                { '[[SEARCH_RESULT_LINK]]': searchLinkHtml }
            );

            // Don't add to conversation history when no context is found
            // to avoid the model repeating this message on the next turn
        } finally {
            this.isGenerating = false;
            this.stopRequested = false;
            this.updateSendButton(false);
            this.elements.userInput.focus();

            this.scheduleSearchStatusClear();
        }
    }

    /**
     * Stream a response for a question that has knowledge-base context.
     * Routes to WebLLM, wllama, or the basic fallback based on
     * `currentMode`, then appends learn-more links and videos.
     */
    async generateResponse(userMessage, searchResult, usedVoiceInput = false) {
        const { context, categories, links, videos } = searchResult;

        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);

        // Add empty message that we'll stream into
        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');

        // Show thinking indicator with a mode-specific note where applicable.
        messageTextDiv.innerHTML = this.getTypingIndicatorHtml();

        try {
            // Route to the appropriate engine
            let assistantMessage = '';

            if (this.currentMode === 'cpu') {
                assistantMessage = await this.generateWithWllama(userMessage, context, messageTextDiv, usedVoiceInput);
            } else if (this.currentMode === 'basic') {
                assistantMessage = await this.generateBasicResponse(searchResult, messageTextDiv, usedVoiceInput);
            } else {
                assistantMessage = await this.generateWithWebLLM(userMessage, context, messageTextDiv, usedVoiceInput);
            }

            if (!assistantMessage.trim()) {
                if (this.currentMode === 'gpu' && this.lastWebLLMCompletionErrored) {
                    messageTextDiv.innerHTML = this.formatResponse(this.gpuModeFailureMessage);
                    return;
                }

                if (this.currentMode === 'cpu' && this.lastWllamaCompletionErrored) {
                    messageTextDiv.innerHTML = this.formatResponse(this.cpuModeFailureMessage);
                    return;
                }
            }

            // Add learn more links and videos
            if (links && links.length > 0 && categories && categories.length > 0) {
                this.renderAssistantMessage(messageTextDiv, assistantMessage, categories, links, videos, {});
            } else if (videos && videos.length > 0) {
                this.renderAssistantMessage(messageTextDiv, assistantMessage, [], [], videos, {});
            }

            // Only add to conversation history if not stopped (to prevent corruption)
            if (!this.stopRequested && assistantMessage.trim()) {
                this.conversationHistory.push({
                    role: 'user',
                    content: userMessage // Store original question, not the one with context
                });
                this.conversationHistory.push({
                    role: 'assistant',
                    content: assistantMessage
                });
            } else if (this.stopRequested) {
                console.log('Stopped response not added to conversation history to prevent corruption');
            }

        } catch (error) {
            console.error('Error generating response:', error);
            responseMessage.remove();

            // Suggest switching to CPU mode if currently in GPU mode
            if (this.currentMode === 'gpu') {
                this.addMessage('assistant', this.gpuModeFailureMessage);
            } else if (this.currentMode === 'cpu') {
                this.addMessage('assistant', this.cpuModeFailureMessage);
            } else {
                this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        } finally {
            this.isGenerating = false;
            this.stopRequested = false;
            this.currentStream = null;
            this.updateSendButton(false);
            this.elements.userInput.focus();

            // Clear search status after response is complete
            this.scheduleSearchStatusClear();
        }
    }

    // ============================================================================
    // LLM RESPONSE GENERATION (WebLLM & Wllama)
    // ============================================================================


    /**
     * Run a streamed chat completion against the WebLLM engine, writing
     * tokens into `messageTextDiv` as they arrive. Honors stop requests
     * and records error state in `lastWebLLMCompletionErrored`.
     * @returns {Promise<string>} The full assistant text (may be empty on stop/error).
     */
    async generateWithWebLLM(userMessage, context, messageTextDiv, usedVoiceInput = false) {
        this.lastWebLLMCompletionErrored = false;

        let userPrompt = context
            ? `${userMessage}\n${this.PROMPT_WITH_CONTEXT}\n${context}`
            : `${userMessage} (${this.PROMPT_WITHOUT_CONTEXT})`;

        const recentHistory = this.conversationHistory.slice(-6);
        recentHistory.push({
            role: 'user',
            content: userPrompt
        });

        const messages = [
            { role: 'system', content: this.SYSTEM_PROMPT },
            ...recentHistory
        ];

        const completion = await this.engine.chat.completions.create({
            messages: messages,
            temperature: 0.7,
            max_tokens: 500,
            stream: true
        });

        this.currentStream = completion;
        let assistantMessage = '';
        let audioPlayed = false;

        try {
            for await (const chunk of completion) {
                if (this.stopRequested) {
                    console.log('Generation stopped by user');
                    break;
                }

                const delta = chunk.choices[0]?.delta?.content;
                if (delta) {
                    // Play audio on first chunk if voice input was used
                    if (!audioPlayed && usedVoiceInput) {
                        this.playRandomResponseAudio();
                        audioPlayed = true;
                    }

                    assistantMessage += delta;
                    messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
                    this.scrollToBottom();
                }
            }
        } catch (error) {
            const isInterrupted = this.stopRequested ||
                error?.name === 'AbortError' ||
                /abort|interrupted|canceled|cancelled/i.test(error?.message || '');

            if (!isInterrupted) {
                this.lastWebLLMCompletionErrored = true;
                console.error('WebLLM generation error:', error);
                return '';
            }

            console.log('WebLLM generation interrupted');
        } finally {
            if (this.currentStream === completion) {
                this.currentStream = null;
            }
        }

        // Clean up incomplete sentences after streaming completes
        const cleanedAssistantMessage = this.trimIncompleteSentenceForCPU(assistantMessage);
        if (cleanedAssistantMessage !== assistantMessage) {
            assistantMessage = cleanedAssistantMessage;
            messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
            this.scrollToBottom();
        }

        return assistantMessage;
    }

    // Helper function to extract first sentence or first 30 characters
    extractFirstSentence(text) {
        if (!text) return '';

        // Find the first occurrence of sentence-ending punctuation
        const match = text.match(/^[^.!?:]*[.!?:]/);
        if (match) {
            return match[0].trim();
        }

        // If no sentence-ending punctuation, use first 30 characters
        return text.substring(0, 30).trim();
    }

    /**
     * Convert messages array to prompt format for wllama.
     * Phi 3.1 uses a simple conversational format without special tokens.
     * @param {Array<{role: string, content: string}>} messages
     * @returns {string} Formatted prompt string
     */
    buildWllamaPrompt(messages) {
        let prompt = '';

        for (const msg of messages) {
            if (msg.role === 'system') {
                prompt += `${msg.content}\n\n`;
            } else if (msg.role === 'user') {
                prompt += `User: ${msg.content}\n\n`;
            } else if (msg.role === 'assistant') {
                prompt += `Assistant: ${msg.content}\n\n`;
            }
        }

        // Add final prompt for assistant response
        prompt += 'Assistant:';

        return prompt;
    }

    truncateParagraphsForCPU(context) {
        if (!context) return context;

        // Split context into sections by blank lines, then concatenate into one flow.
        const sections = context
            .split(/\n\s*\n+/)
            .map(section => section.replace(/\s+/g, ' ').trim())
            .filter(Boolean);

        const truncatedSections = sections.map(section => {
            // If section is <= 100 chars, keep it as is
            if (section.length <= 100) return section;

            // Find first sentence ending (., !, ?) after position 100
            const searchFrom = 100;
            let endPos = -1;

            for (let i = searchFrom; i < section.length; i++) {
                if (section[i] === '.' || section[i] === '!' || section[i] === '?') {
                    endPos = i + 1; // Include the punctuation
                    break;
                }
            }

            // If found a sentence ending, truncate there
            if (endPos > 0) {
                return section.substring(0, endPos);
            }

            // Otherwise, truncate at 100 chars with ellipsis
            return section.substring(0, 100) + '...';
        });

        return truncatedSections.reduce((combined, section) => {
            if (!combined) {
                return section;
            }

            const needsSentenceSeparator = !/[.!?]\s*$/.test(combined);
            return combined + (needsSentenceSeparator ? '. ' : ' ') + section;
        }, '');
    }

    trimIncompleteSentenceForCPU(text) {
        if (!text) return text;

        let trimmedText = text.trim();
        if (!trimmedText) return trimmedText;

        const hasCompleteSentenceEnding = (value) => /[.!?]["')\]]*\s*$/.test(value);

        if (hasCompleteSentenceEnding(trimmedText)) {
            return trimmedText;
        }

        const trailingLeadInPattern = /(?:\b(?:and|or|but|so|because|which|that|who|when|where|while|with|for|to|of|in|on|at|by|as|like|including)\b|\b(?:such as|for example|for instance|for example,|for instance,)\b|[,;:\-–—(]\s*)$/i;

        while (trailingLeadInPattern.test(trimmedText)) {
            trimmedText = trimmedText.replace(trailingLeadInPattern, '').trim();
        }

        if (!trimmedText) {
            return '';
        }

        if (hasCompleteSentenceEnding(trimmedText)) {
            return trimmedText;
        }

        let lastSentenceEnd = -1;

        for (let i = 0; i < trimmedText.length; i++) {
            if (trimmedText[i] === '.' || trimmedText[i] === '!' || trimmedText[i] === '?') {
                let endIndex = i + 1;
                while (endIndex < trimmedText.length && /["')\]]/.test(trimmedText[endIndex])) {
                    endIndex++;
                }
                lastSentenceEnd = endIndex;
            }
        }

        if (lastSentenceEnd > 0) {
            return trimmedText.substring(0, lastSentenceEnd).trim();
        }

        return trimmedText;
    }

    /**
     * Compose a Basic-mode reply directly from the search result (no
     * model inference) by stitching together matching document summaries.
     */
    buildBasicResponse(searchResult) {
        const { documents = [] } = searchResult;

        if (!documents.length) {
            return "Sorry, I couldn't find any specific information on that topic. Please try rephrasing your question or explore other AI concepts.";
        }

        return documents.map(document => document.content).join('\n\n');
    }

    /** Animate the Basic-mode reply into the message bubble. */
    async generateBasicResponse(searchResult, messageTextDiv, usedVoiceInput = false) {
        const assistantMessage = this.buildBasicResponse(searchResult);

        await new Promise(resolve => setTimeout(resolve, 250));

        if (this.stopRequested) {
            return '';
        }

        if (usedVoiceInput) {
            this.playRandomResponseAudio();
        }

        const animationCompleted = await this.animateTyping(
            messageTextDiv,
            assistantMessage,
            (partialMessage) => this.formatResponse(partialMessage),
            12
        );

        return animationCompleted ? assistantMessage : '';
    }

    /**
     * Run a streamed chat completion against the wllama (CPU) engine,
     * writing tokens into `messageTextDiv` as they arrive. Honors stop
     * requests and records error state in `lastWllamaCompletionErrored`.
     * @returns {Promise<string>} The full assistant text (may be empty on stop/error).
     */
    async generateWithWllama(userMessage, context, messageTextDiv, usedVoiceInput = false) {
        this.lastWllamaCompletionErrored = false;

        // Ensure wllama is loaded
        if (!this.wllama) {
            throw new Error('Wllama is not initialized. Please wait for CPU mode to finish loading.');
        }

        // Build messages array (same format as WebLLM for consistency)
        const messages = [
            { role: 'system', content: this.SYSTEM_PROMPT }
        ];

        // Add truncated previous conversation if available
        if (this.conversationHistory.length >= 2) {
            const prevUser = this.conversationHistory[this.conversationHistory.length - 2];
            const prevAssistant = this.conversationHistory[this.conversationHistory.length - 1];

            if (prevUser.role === 'user' && prevAssistant.role === 'assistant') {
                const prevUserSentence = this.extractFirstSentence(prevUser.content);
                const prevAssistantSentence = this.extractFirstSentence(prevAssistant.content);

                messages.push(
                    { role: 'user', content: prevUserSentence },
                    { role: 'assistant', content: prevAssistantSentence }
                );
            }
        }

        // Add current user message with context if available
        let userPrompt;
        if (context) {
            const truncatedContext = this.truncateParagraphsForCPU(context);
            userPrompt = `${userMessage}\n${this.PROMPT_WITH_CONTEXT}\n${truncatedContext}`;
        } else {
            userPrompt = `${userMessage} (${this.PROMPT_WITHOUT_CONTEXT})`;
        }
        messages.push({ role: 'user', content: userPrompt });

        // Convert messages to prompt format for wllama
        const prompt = this.buildWllamaPrompt(messages);
        console.log('Sending prompt to wllama (length:', prompt.length, 'chars)');
        console.log('Wllama prompt:', prompt);

        let assistantMessage = '';
        let audioPlayed = false;
        let completion = null;
        let firstChunkReceived = false;
        let timeoutMessageAdded = false;

        // Create AbortController for consistency with other generation paths.
        const controller = new AbortController();
        this.currentAbortController = controller;

        // Set up a 20-second timeout for slow responses
        const slowResponseTimeout = setTimeout(() => {
            if (!firstChunkReceived && !this.stopRequested) {
                timeoutMessageAdded = true;
                // Add a waiting message with animated dots
                const waitingHtml = 'I\'m looking for information on that...<br><span class="typing-indicator" aria-label="Searching">●●●</span>';
                messageTextDiv.innerHTML = waitingHtml;
                this.scrollToBottom();
                if (usedVoiceInput) {
                    this.playLookingAudio();
                }
                console.log('Wllama slow response: showing waiting message after 20 seconds');
            }
        }, 20000);

        // Use streaming with proper abort support
        try {
            completion = await this.wllama.createCompletion({
                prompt: prompt,
                max_tokens: 200,
                temperature: 0.2,
                top_k: 40,
                top_p: 0.9,
                frequency_penalty: 1.1,
                stop: ['\n\nUser:', '\nUser:', 'User:', '\n\nAssistant:'],
                signal: controller.signal,
                stream: true
            });

            this.currentStream = completion;

            for await (const chunk of completion) {
                if (this.stopRequested) {
                    console.log('Wllama generation stopped by user');
                    break;
                }

                if (chunk.choices && chunk.choices[0] && chunk.choices[0].text) {
                    // Clear timeout on first chunk
                    if (!firstChunkReceived) {
                        clearTimeout(slowResponseTimeout);
                        firstChunkReceived = true;
                        // If we showed the waiting message, clear it before showing the real response
                        if (timeoutMessageAdded) {
                            messageTextDiv.innerHTML = '';
                        }
                    }

                    // Play audio on first chunk if voice input was used
                    if (!audioPlayed && usedVoiceInput) {
                        this.playRandomResponseAudio();
                        audioPlayed = true;
                    }

                    assistantMessage += chunk.choices[0].text;
                    messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
                    this.scrollToBottom();
                }
            }

            const cleanedAssistantMessage = this.trimIncompleteSentenceForCPU(assistantMessage);
            if (cleanedAssistantMessage !== assistantMessage) {
                assistantMessage = cleanedAssistantMessage;
                messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
                this.scrollToBottom();
            }

            // Clear abort controller on successful completion
            this.currentAbortController = null;

        } catch (error) {
            // Clear timeout on error
            clearTimeout(slowResponseTimeout);

            // Check if this was an abort (expected when user clicks stop)
            if (this.stopRequested || error.name === 'AbortError' || error.message?.includes('abort')) {
                console.log('Generation aborted by user');
                this.lastWllamaCompletionErrored = false;
            } else {
                console.log('Wllama generation error:', error.message || 'unknown error');
                this.lastWllamaCompletionErrored = true;
            }
            this.currentAbortController = null;
        } finally {
            // Ensure timeout is cleared
            clearTimeout(slowResponseTimeout);

            if (this.currentStream === completion) {
                this.currentStream = null;
            }
        }

        console.log('Wllama response complete, length:', assistantMessage.length);

        return assistantMessage;
    }

    /** Convert model markdown-ish output to safe display HTML. */
    formatResponse(text) {
        // Split out the learn more section and note if they exist
        const learnMoreMatch = text.match(/([\s\S]*?)(---\s*\n\n\*\*Learn more:\*\*.*?)(\n\n\*Note:.*)?$/);

        if (learnMoreMatch) {
            const mainContent = learnMoreMatch[1];
            const learnMoreSection = learnMoreMatch[2];
            const noteSection = learnMoreMatch[3] || '';

            // Format main content (escape HTML)
            let formatted = this.escapeHtml(mainContent);

            // Convert **bold** to <strong>
            formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

            // Convert *italic* to <em>
            formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');

            // Convert line breaks to paragraphs
            formatted = formatted.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');

            // Add learn more section - preserve placeholders and HTML structure
            const learnMoreFormatted = learnMoreSection
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/---\s*\n\n/g, '<hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;">\n\n');

            // Format note section - preserve placeholders
            let noteFormatted = '';
            if (noteSection) {
                // Extract the note text (remove leading \n\n*Note: and trailing *)
                let noteText = noteSection.replace(/^\n\n\*Note:\s*/g, '').replace(/\*$/g, '');
                // Wrap in styled paragraph - placeholders will be replaced by caller
                noteFormatted = `<p style="font-style: italic; color: #666; font-size: 0.9em; margin-top: 10px;">Note: ${noteText}</p>`;
            }

            return formatted + learnMoreFormatted + noteFormatted;
        }

        // No learn more section, process normally
        let formatted = this.escapeHtml(text);

        // Convert **bold** to <strong>
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Convert *italic* to <em>
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Convert line breaks to paragraphs
        formatted = formatted.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');

        return formatted;
    }

    /** Escape `&<>"'` to prevent HTML injection. */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Type `text` into `element` one character at a time, re-applying
     * `formatter` after each step. Returns false if cancelled via
     * `stopRequested`, true on natural completion.
     */
    async animateTyping(element, text, formatter = null, speed = 5) {
        element.innerHTML = '';
        const segments = text.split(/(\s+)/).filter(segment => segment.length > 0);
        let currentText = '';

        for (const segment of segments) {
            if (this.stopRequested) {
                return false;
            }

            currentText += segment;
            element.innerHTML = formatter ? formatter(currentText) : this.escapeHtml(currentText);
            this.scrollToBottom();
            await new Promise(resolve => setTimeout(resolve, speed));
        }

        return true;
    }

    /** Scroll the chat view to its newest message. */
    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    // ============================================================================
    // SPEECH RECOGNITION HELPER METHODS
    // ============================================================================

    /** Toggle the mic button's pressed/listening styling and aria label. */
    setMicButtonState(isActive, label = null) {
        if (isActive) {
            this.elements.micBtn.style.opacity = '0.6';
            this.elements.micBtn.classList.add('active');
            this.elements.micBtn.title = label || 'Listening...';
            this.elements.micBtn.setAttribute('aria-label', label || 'Listening to your voice input');
        } else {
            this.elements.micBtn.style.opacity = '1';
            this.elements.micBtn.classList.remove('active');
            this.elements.micBtn.title = label || 'Voice input';
            this.elements.micBtn.setAttribute('aria-label', label || 'Voice input');
        }
    }

    /** Cancel the silence + no-speech timers used during voice capture. */
    clearSpeechTimers() {
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        if (this.noSpeechTimer) {
            clearTimeout(this.noSpeechTimer);
            this.noSpeechTimer = null;
        }
    }

    /** Close the AudioContext and stop the mic stream after voice capture. */
    cleanupAudioResources() {
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
    }

    // ============================================================================
    // AUDIO PLAYBACK
    // ============================================================================

    /** Play one of the 7 spoken response confirmations after voice input. */
    playRandomResponseAudio() {
        // Randomly select one of the 7 audio files
        const audioNumber = Math.floor(Math.random() * 7) + 1;
        const audioPath = `audio/response_${audioNumber}.wav`;

        const audio = new Audio(audioPath);
        audio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }

    /** Spoken cue to indicate the assistant is looking for information. */
    playLookingAudio() {
        const audio = new Audio('audio/looking.wav');
        audio.play().catch(error => {
            console.error('Error playing looking audio:', error);
        });
    }

    /** Spoken "sorry" cue when a prompt fails moderation. */
    playModerationAudio() {
        const audio = new Audio('audio/sorry.wav');
        audio.play().catch(error => {
            console.error('Error playing moderation audio:', error);
        });
    }

    /** Spoken cue when the search returns no results. */
    playNoResultsAudio() {
        const audio = new Audio('audio/no_results.wav');
        audio.play().catch(error => {
            console.error('Error playing no results audio:', error);
        });
    }

    /** Spoken cue when search-link results are about to be shown. */
    playSearchResultsAudio() {
        const audio = new Audio('audio/search_results.wav');
        audio.play().catch(error => {
            console.error('Error playing search results audio:', error);
        });
    }

    // ============================================================================
    // SPEECH RECOGNITION - WEB SPEECH API & VOSK
    // ============================================================================

    /**
     * Toggle voice input. Prefers the Web Speech API; on failure or in
     * unsupported browsers, lazy-loads Vosk and uses the WASM model.
     */
    async handleMicClick() {
        // Try Web Speech API first
        if (this.usingWebSpeech) {
            const webSpeechWorked = await this.tryWebSpeech();
            if (!webSpeechWorked) {
                // Web Speech failed, switch to Vosk
                console.log('Web Speech API not available, loading Vosk fallback...');
                this.usingWebSpeech = false;

                // Load Vosk model if not already loaded
                if (!this.voskLoaded) {
                    const loaded = await this.loadVoskModel();
                    if (!loaded) {
                        return; // Vosk failed to load
                    }
                }

                // Now try Vosk
                await this.startVoskRecording();
            }
        } else {
            // Already using Vosk
            if (this.isRecording) {
                this.stopVoskRecording(true);
            } else {
                await this.startVoskRecording();
            }
        }
    }

    /**
     * Attempt one capture using the browser's Web Speech API.
     * @returns {Promise<boolean>} true if recognition started successfully.
     */
    async tryWebSpeech() {
        return new Promise((resolve) => {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

            if (!SpeechRecognition) {
                resolve(false);
                return;
            }

            try {
                const recognition = new SpeechRecognition();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                let hasResolved = false;
                let noSpeechTimer = null;

                // Set active state
                this.setMicButtonState(true);

                // Start no-speech timeout
                noSpeechTimer = setTimeout(() => {
                    if (!hasResolved) {
                        console.log('No speech detected in 5 seconds, cancelling...');
                        recognition.stop();
                        if (!hasResolved) {
                            hasResolved = true;
                            this.setMicButtonState(false);
                            this.addMessage('assistant', 'No speech detected. Please try again.');
                            resolve(true); // Don't fallback, just inform user
                        }
                    }
                }, this.noSpeechTimeout);

                recognition.onresult = (event) => {
                    // Clear no-speech timer since we got speech
                    if (noSpeechTimer) {
                        clearTimeout(noSpeechTimer);
                        noSpeechTimer = null;
                    }

                    const transcript = event.results[0][0].transcript;
                    this.elements.userInput.value = transcript;
                    this.autoResizeTextarea();
                    this.usedVoiceInput = true;
                    this.sendMessage();

                    if (!hasResolved) {
                        hasResolved = true;
                        resolve(true);
                    }
                };

                recognition.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);

                    // Clear no-speech timer
                    if (noSpeechTimer) {
                        clearTimeout(noSpeechTimer);
                        noSpeechTimer = null;
                    }

                    // Reset visual state
                    this.setMicButtonState(false);

                    if (!hasResolved) {
                        hasResolved = true;
                        // If it's a permission error, don't fallback
                        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                            this.addMessage('assistant', 'Microphone access was denied.');
                            resolve(true); // Don't fallback, user denied permission
                        } else {
                            // Any other error (network, no-speech, etc.) triggers fallback
                            resolve(false); // Fallback to Vosk
                        }
                    }
                };

                recognition.onend = () => {
                    // Clear no-speech timer
                    if (noSpeechTimer) {
                        clearTimeout(noSpeechTimer);
                        noSpeechTimer = null;
                    }

                    this.setMicButtonState(false);
                };

                recognition.start();
                console.log('Web Speech recognition started');
                // Don't resolve here - wait for result or error
            } catch (error) {
                console.error('Error starting Web Speech recognition:', error);
                this.setMicButtonState(false);
                resolve(false);
            }
        });
    }

    resetSilenceTimer() {
        // Clear existing timer
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
        }

        // Set new timer to auto-stop after silence
        if (this.isRecording) {
            this.silenceTimer = setTimeout(() => {
                if (this.isRecording && this.hasSpeech) {
                    console.log('Silence detected, auto-stopping...');
                    this.stopVoskRecording(false);
                }
            }, this.silenceTimeout);
        }
    }

    /**
     * Open the mic, pipe PCM frames into the loaded Vosk recognizer, and
     * auto-stop on silence (or no-speech timeout). Results are written
     * into the input field via the recognizer callback in {@link loadVoskModel}.
     */
    async startVoskRecording() {
        if (!this.voskRecognizer) {
            this.addMessage('assistant', 'Speech input is not available.');
            return;
        }

        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1,
                    sampleRate: 16000
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);

            // Process audio data
            this.processorNode.onaudioprocess = (event) => {
                try {
                    if (this.isRecording && this.voskRecognizer) {
                        this.voskRecognizer.acceptWaveform(event.inputBuffer);
                    }
                } catch (e) {
                    console.error('Audio processing error:', e);
                }
            };

            this.sourceNode.connect(this.processorNode);
            this.processorNode.connect(this.audioContext.destination);

            this.isRecording = true;
            this.usedVoiceInput = true;
            this.hasSpeech = false;
            this.lastSpeechTime = Date.now();

            // Start silence detection timer
            this.resetSilenceTimer();

            // Start no-speech timeout
            this.noSpeechTimer = setTimeout(() => {
                if (this.isRecording && !this.hasSpeech) {
                    console.log('No speech detected in 5 seconds, cancelling...');
                    this.stopVoskRecording(true);
                    this.addMessage('assistant', 'No speech detected. Please try again.');
                }
            }, this.noSpeechTimeout);

            // Set active state
            this.setMicButtonState(true);

            console.log('Vosk recording started');
        } catch (error) {
            console.error('Microphone access denied:', error);
            this.addMessage('assistant', 'Microphone access was denied. Please allow microphone access to use voice input.');
        }
    }

    /**
     * Tear down the Vosk capture pipeline.
     * @param {boolean} [isCancelled] If true, discard any partial recognition result.
     */
    stopVoskRecording(isCancelled = false) {
        this.isRecording = false;

        // Clear all timers
        this.clearSpeechTimers();

        // Clean up audio resources
        this.cleanupAudioResources();

        // Reset button state
        this.setMicButtonState(false);

        console.log('Vosk recording stopped');

        // Auto-send the message if there's text and it wasn't manually cancelled
        if (!isCancelled && this.elements.userInput.value.trim()) {
            this.sendMessage();
        }
    }

    // ============================================================================
    // UI CONTROLS & MODAL MANAGEMENT
    // ============================================================================

    /** Clear chat history and the message list, keeping the welcome message. */
    restartConversation() {
        if (confirm('Are you sure you want to start a new conversation? This will clear the chat history.')) {
            // Clear conversation history
            this.conversationHistory = [];

            // Clear chat messages (keep welcome message)
            const messages = this.elements.chatMessages.querySelectorAll('.message:not(.welcome-message)');
            messages.forEach(msg => msg.remove());

            // Clear search status
            this.elements.searchStatus.textContent = '';

            // Reset wllama cache if in CPU mode
            if (this.currentMode === 'cpu' && this.wllama) {
                try {
                    this.wllama.samplingReset();
                    console.log('Wllama cache reset for new conversation');
                } catch (error) {
                    console.warn('Failed to reset wllama cache:', error);
                }
            }

            // Reset WebLLM chat if in GPU mode
            if (this.currentMode === 'gpu' && this.engine) {
                try {
                    if (typeof this.engine.resetChat === 'function') {
                        this.engine.resetChat();
                        console.log('WebLLM cache reset for new conversation');
                    }
                } catch (error) {
                    console.warn('Failed to reset WebLLM cache:', error);
                }
            }

            console.log('Conversation restarted');
        }
    }

    /** Open the AI-mode explainer modal and install a focus trap. */
    showAiModeModal() {
        this.elements.aiModeModal.style.display = 'flex';
        this.currentModal = this.elements.aiModeModal;
        this.elements.aiModeModal.setAttribute('aria-hidden', 'false');
        setTimeout(() => {
            this.elements.modalClose.focus();
            this.setupModalFocusTrap(this.elements.aiModeModal);
        }, 100);
    }

    /** Close the AI-mode modal and restore focus to the trigger. */
    hideAiModeModal() {
        this.elements.aiModeModal.style.display = 'none';
        this.elements.aiModeModal.setAttribute('aria-hidden', 'true');
        this.removeModalFocusTrap();
        this.currentModal = null;
        if (this.lastFocusedElement) {
            this.lastFocusedElement.focus();
        } else {
            this.elements.userInput.focus();
        }
    }

    /** Open the About modal and install a focus trap. */
    showAboutModal() {
        this.elements.aboutModal.style.display = 'flex';
        this.currentModal = this.elements.aboutModal;
        // Store the previously focused element
        this.lastFocusedElement = document.activeElement;
        // Announce modal to screen readers
        this.elements.aboutModal.setAttribute('aria-hidden', 'false');
        // Set focus to close button
        setTimeout(() => {
            this.elements.aboutModalClose.focus();
            this.setupModalFocusTrap(this.elements.aboutModal);
        }, 100);
    }

    /** Close the About modal and restore focus to the trigger. */
    hideAboutModal() {
        this.elements.aboutModal.style.display = 'none';
        this.elements.aboutModal.setAttribute('aria-hidden', 'true');
        this.removeModalFocusTrap();
        this.currentModal = null;
        // Restore focus to the element that opened the modal
        if (this.lastFocusedElement) {
            this.lastFocusedElement.focus();
        } else {
            this.elements.userInput.focus();
        }
    }

    /** Resolve a Synthesia video id (or already-full URL) to an embed URL. */
    getSynthesiaVideoUrl(videoId) {
        return videoId.startsWith('http') ? videoId : `https://share.synthesia.io/embeds/videos/${videoId}`;
    }

    /**
     * Open `videoUrl` in a centered popup window; if popups are blocked,
     * fall back to navigating the current tab.
     */
    openVideoPopup(videoUrl) {
        const left = Math.max(0, Math.round(window.screenX + ((window.outerWidth - this.videoPopupWidth) / 2)));
        const top = Math.max(0, Math.round(window.screenY + ((window.outerHeight - this.videoPopupHeight) / 2)));
        const popupFeatures = [
            'popup=yes',
            `width=${this.videoPopupWidth}`,
            `height=${this.videoPopupHeight}`,
            `left=${left}`,
            `top=${top}`,
            'resizable=yes',
            'scrollbars=yes'
        ].join(',');

        const popup = window.open(videoUrl, 'ask-anton-video', popupFeatures);
        if (!popup) {
            window.location.href = videoUrl;
            return;
        }

        popup.opener = null;
        popup.focus();
    }

    /**
     * Install a Tab/Shift+Tab focus trap that keeps keyboard focus inside
     * `modalElement` until {@link removeModalFocusTrap} is called.
     */
    setupModalFocusTrap(modalElement) {
        // Get all focusable elements within the modal
        const focusableElements = modalElement.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        // Store the trap handler for cleanup
        this.modalFocusTrapHandler = (e) => {
            if (e.key !== 'Tab') return;

            if (e.shiftKey) {
                // Shift+Tab - going backwards
                if (document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                }
            } else {
                // Tab - going forwards
                if (document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        };

        modalElement.addEventListener('keydown', this.modalFocusTrapHandler);
    }

    /** Remove the focus trap installed by {@link setupModalFocusTrap}. */
    removeModalFocusTrap() {
        if (this.currentModal && this.modalFocusTrapHandler) {
            this.currentModal.removeEventListener('keydown', this.modalFocusTrapHandler);
            this.modalFocusTrapHandler = null;
        }
    }
}

// Make instance globally accessible for onclick handler
window.askAnton = null;

// Initialize the app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.askAnton = new AskAnton();
    });
} else {
    window.askAnton = new AskAnton();
}
