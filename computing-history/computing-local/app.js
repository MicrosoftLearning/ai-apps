import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/index.js';

const chatContainer = document.getElementById('chat-messages');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const imageUpload = document.getElementById('image-upload');
const micBtn = document.getElementById('mic-btn');
const modeSelect = document.getElementById('mode-select');

// State
let model = null;
let featureExtractor = null;
let pendingFile = null;
let isVoiceInput = false; // Tracks if current message was spoken
let isResponding = false; // Tracks if bot is currently responding
let shouldStopResponse = false; // Flag to cancel ongoing response
let isStoppingResponse = false; // Prevent new prompts while stream cleanup is running
let wllama = null; // Wllama instance for CPU mode
let wllamaReady = false; // Track if wllama is initialized
let wllamaUsedGPU = false; // True if current wllama instance was loaded with GPU layers
let gpuFailed = false; // True after a GPU session crash; suppresses future GPU attempts
let mobilenetReady = false; // Track if MobileNet is initialized
let mobilenetLoadPromise = null; // Coalesce concurrent lazy-load requests
let conversationHistory = []; // Track conversation for context
let inappropriateWords = []; // Loaded from moderation file
let currentMode = 'basic'; // Track which engine is active: 'cpu' or 'basic'
const availableModes = { cpu: true, basic: true }; // Track which modes can be used
let currentAbortController = null; // Track abort controller for wllama
let currentStream = null; // Track active stream for proper cleanup
let typingAnimationsInProgress = 0; // Track active typewriter animations
let modelLoadingCancelled = false; // Track if user cancelled model loading
let modelLoadingAbortController = null; // Track abort controller for model loading
let usingPrerecordedVoice = false; // Track if using pre-recorded voice fallback
const CPU_MODE_FAILURE_MESSAGE = "I'm sorry, something went wrong in AI mode.\nIf this keeps happening, please try switching to Basic (Wikipedia) mode.";
let lastWllamaCompletionErrored = false; // Track whether last CPU completion failed with an error
let wllamaShouldFailoverToBasic = false; // Flag to trigger failover from wllama to basic mode

// Debug flags for testing failover (can be set via URL params or console)
let debugConfig = { enabled: false, forceWllamaGenerationFail: false };

// Shared prompt constants for Wllama and Wikipedia modes
const SYSTEM_PROMPT = 'You are an AI assistant that helps people find information about computing history. Always respond with a single paragraph, using short sentences.';

// Vosk speech recognition (lazy-loaded fallback)
let voskModel = null;
let voskRecognizer = null;
let voskLoaded = false;
let voskLoadingFailed = false;
let isRecording = false;
let mediaStream = null;
let audioContext = null;
let processorNode = null;
let sourceNode = null;
let silenceTimer = null;
let noSpeechTimer = null;
let lastSpeechTime = null;
let hasSpeech = false;
const silenceTimeout = 2000; // Auto-stop after 2 seconds of silence
const noSpeechTimeout = 7000; // Cancel after 7 seconds of no speech
let usingWebSpeech = true; // Try Web Speech API first
let voskLoadingMessage = null; // Track the loading message for Vosk

// Calculate speech model path dynamically based on current location
// This works both locally and on GitHub Pages
// From computing-local, go up 3 levels: computing-local -> computing-history -> ai-apps -> speech-model
const basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
const parentPath = basePath.substring(0, basePath.lastIndexOf('/'));
const rootPath = parentPath.substring(0, parentPath.lastIndexOf('/'));
const speechModelUrl = `${rootPath}/speech-model/speech-model.tar.gz`;

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================

/**
 * Parse URL parameters for debug mode.
 * Example: ?debug=true&forceWllamaFail=true
 */
function parseDebugConfig() {
    const params = new URLSearchParams(window.location.search);
    const config = {
        enabled: params.get('debug') === 'true',
        forceWllamaGenerationFail: params.get('forceWllamaFail') === 'true'
    };
    if (config.enabled) {
        console.log('🧪 DEBUG MODE ENABLED');
        console.log('Debug config:', config);
    }
    return config;
}

// ============================================================================
// GLOBAL ERROR HANDLER
// ============================================================================

/**
 * Handles unhandled promise rejections, particularly Vosk model loading errors
 */
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);

    // Check if this is a Vosk model loading error
    if (!voskLoaded && !voskLoadingFailed && event.reason) {
        const errorMsg = String(event.reason);
        const errorStack = event.reason?.stack || '';
        // Check for Vosk-related errors
        if (errorMsg.includes('404') ||
            errorMsg.includes('HTTP error') ||
            errorMsg.includes('model') ||
            errorMsg.includes('Cannot read properties of undefined') ||
            errorStack.includes('vosk')) {
            console.log('Detected Vosk model loading failure from unhandled rejection');
            voskLoadingFailed = true;

            // Remove the loading message if it exists
            if (voskLoadingMessage && voskLoadingMessage.message) {
                voskLoadingMessage.message.remove();
                voskLoadingMessage = null;
            }

            // Add error message to chat
            addMessage('I\'m sorry. Voice input is unavailable.', 'bot');
            micBtn.disabled = true;
            micBtn.title = 'Voice input unavailable';

            // Re-enable buttons
            sendBtn.disabled = false;
            textInput.disabled = false;

            event.preventDefault();
        }
    }
});

// ============================================================================
// IMAGE CLASSIFICATION CONSTANTS
// ============================================================================

// Vision model paths
const MODEL_URL = './image_model/retro-classifier-model.json';
const BASE_MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

// Classification classes
const CLASSES = [
    'Altair 8800', // 0
    'Commodore 64', // 1
    'Sinclair ZX Spectrum', // 2
    'Apple II', // 3
    'Computer', // 4
    'Printed Circuit Board (PCB)', // 5
    'Unknown' // 6
];

const CLASS_INFO = {
    0: [
        'Developed by: Micro Instrumentation and Telemetry Systems (MITS)',
        'Released: 1974',
        'Processor: Intel 8080',
        'Fact: It was the first commercially successful personal computer.'
    ].join('\n'),
    1: [
        'Developed by: Commodore Business Machines',
        'Released: 1982',
        'Processor: MOS 6510',
        'Fact: It was one of the best-selling desktop computers of all time, with over 12 million units sold.'
    ].join('\n'),
    2: [
        'Developed by: Sinclair Research',
        'Released: 1982',
        'Processor: Zilog Z-80A',
        'Fact: It played a pivotal role in the development of the computer games industry, especially in the United Kingdom.'
    ].join('\n'),
    3: [
        'Developed by: Apple Computer, Inc',
        'Released: 1977',
        'Processor: MOS 6502',
        'Fact: It was one of the first personal computers to feature color graphics.'
    ].join('\n')
};

/**
 * Builds a prompt for generating information about a classified computer
 * @param {number} classIndex - Index of the classification class
 * @returns {string|null} Formatted prompt or null if class has no info
 */
function buildClassInfoPrompt(classIndex) {
    const className = CLASSES[classIndex];
    const classInfo = CLASS_INFO[classIndex];

    if (!className || !classInfo) {
        return null;
    }

    return `Provide a concise paragraph describing the ${className} computer using the following information:\nINFORMATION:\n---\n${classInfo}\n---`;
}

// ============================================================================
// HARDWARE REQUIREMENTS CHECK
// ============================================================================

/**
 * Check if the device meets minimum hardware requirements for running
 * the Phi 3.5-mini model. Returns false if device memory or CPU cores
 * are below the minimum thresholds.
 * @returns {boolean} true if hardware meets requirements, false otherwise.
 */
function checkHardwareRequirements() {
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

// ============================================================================
// INITIALIZATION FUNCTIONS
// ============================================================================

/**
 * Initializes the application by loading ML models and AI engines
 */
async function init() {
    // Setup cancel link event listener first
    setupCancelLinkListener();

    // Parse debug configuration from URL params
    debugConfig = parseDebugConfig();

    // Show loading overlay
    updateLoadingStatus('mobilenet', 'loading', 'Loading...');
    updateLoadingStatus('phi', 'loading', 'Loading...');

    // Pin TFJS to the CPU backend before any tf.* calls so it never grabs a
    // WebGL context. The image classifier is small (~470K params) and runs
    // fine on CPU.
    try {
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('TFJS backend:', tf.getBackend());
    } catch (backendErr) {
        console.warn('Failed to pin TFJS to CPU backend:', backendErr);
    }

    try {
        await loadInappropriateWords();

        // Pre-load MobileNet + classifier up front so the first image upload
        // doesn't have to wait for it (and so any load failure surfaces here
        // rather than partway through a classification flow). Failure is
        // non-fatal: ensureImageModelLoaded will retry on demand.
        ensureImageModelLoaded().catch(err => {
            console.warn('Image model preload failed (will retry on demand):', err);
        });

        // Create abort controller for model loading
        modelLoadingAbortController = new AbortController();

        // Check hardware requirements before attempting to load model
        if (!checkHardwareRequirements()) {
            console.log('Hardware requirements not met, using Basic mode only');
            currentMode = 'basic';
            availableModes.cpu = false;
            updateModelName('Wikipedia API (Basic)');
            updateLoadingStatus('phi', 'ready', 'Basic');
            const infoMsg = 'Your device does not meet the minimum requirements (8GB RAM, 8 CPU cores) for running the Phi 3.5-mini model. Using Basic (Wikipedia) mode.';
            hideLoadingOverlay();
            updateModeSelect();
            setTimeout(() => {
                textInput.focus();
                addMessage(infoMsg, 'bot');
            }, 550);
            return;
        }

        console.log('Initializing wllama (CPU mode)...');
        try {
            await initWllama();
            currentMode = 'cpu';
        } catch (error) {
            if (modelLoadingCancelled) {
                return;
            }
            console.log('Wllama failed, using Basic mode (Wikipedia)');
            currentMode = 'basic';
            availableModes.cpu = false;
            updateModelName('Wikipedia API (Basic)');
            updateLoadingStatus('phi', 'ready', 'Basic');
        }

        hideLoadingOverlay();
        updateModeSelect();
        setTimeout(() => textInput.focus(), 550);
    } catch (e) {
        console.error("Initialization error:", e);
        // Show error but still try to hide overlay after a delay
        setTimeout(() => {
            hideLoadingOverlay();
            addMessage(`Error loading models: ${escapeHtml(e.message)}. Some features may be unavailable.`, "bot");
            setTimeout(() => textInput.focus(), 550);
        }, 2000);
    }
}

/**
 * Sets up the cancel loading link event listener
 */
function setupCancelLinkListener() {
    const cancelLink = document.getElementById('cancel-loading-link');
    if (cancelLink) {
        // Initially hide the cancel link
        cancelLink.style.display = 'none';

        // Click handler
        cancelLink.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            cancelModelLoading();
        });

        // Keyboard handler for accessibility
        cancelLink.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                e.stopPropagation();
                cancelModelLoading();
            }
        });
    }
}

/**
 * Cancels the ongoing model loading and switches to Basic mode
 */
function cancelModelLoading() {
    // Set the cancellation flag immediately
    modelLoadingCancelled = true;

    // Abort any ongoing loading operations
    if (modelLoadingAbortController) {
        modelLoadingAbortController.abort();
    }

    // Clean up any partially loaded wllama instance
    if (wllama) {
        const _old = wllama;
        wllama = null;
        _old.exit().catch(() => { });
    }

    // Hide the cancel link immediately
    const cancelLink = document.getElementById('cancel-loading-link');
    if (cancelLink) {
        cancelLink.style.display = 'none';
    }

    // Don't mark modes as unavailable - user cancelled, not failed
    // Switch to Basic mode
    currentMode = 'basic';
    updateModelName('Wikipedia API (Basic)');
    updateLoadingStatus('phi', 'ready', 'Basic');

    hideLoadingOverlay();
    updateModeSelect();
    setTimeout(() => textInput.focus(), 550);

    setTimeout(() => {
        addMessage('Model loading was cancelled. You can switch modes anytime from the mode selector.', 'bot');
    }, 500);
}

// ============================================================================
// SPEECH RECOGNITION FUNCTIONS
// ============================================================================

/**
 * Loads the Vosk speech model for offline speech recognition (lazy-loaded fallback)
 * @returns {Promise<boolean>} True if loaded successfully, false otherwise
 */
async function loadVoskModel() {
    if (voskLoaded || voskLoadingFailed) {
        return voskLoaded;
    }

    try {
        console.log('Loading Vosk speech model from', speechModelUrl);

        if (!window.Vosk || typeof Vosk.createModel !== 'function') {
            console.warn('Vosk library not loaded');
            voskLoadingFailed = true;
            return false;
        }

        voskLoadingMessage = addMessage('Loading offline speech model... This may take a moment.', 'bot');
        sendBtn.disabled = true;
        textInput.disabled = true;
        micBtn.disabled = true;

        // Load the model - no timeout, just let it load
        voskModel = await Vosk.createModel(speechModelUrl);

        // Validate that the model loaded
        if (!voskModel || typeof voskModel.KaldiRecognizer !== 'function') {
            throw new Error('Vosk model did not load properly');
        }

        voskRecognizer = new voskModel.KaldiRecognizer(16000);

        // Set up recognizer event handlers
        voskRecognizer.on("result", (message) => {
            const result = message.result;
            if (result && result.text) {
                // Clear no-speech timer since we got speech
                if (noSpeechTimer) {
                    clearTimeout(noSpeechTimer);
                    noSpeechTimer = null;
                }

                // Append the recognized text to the input
                const currentText = textInput.value;
                textInput.value = currentText + (currentText ? " " : "") + result.text;
                hasSpeech = true;
                lastSpeechTime = Date.now();
                resetSilenceTimer();
            }
        });

        voskRecognizer.on("partialresult", (message) => {
            // Reset silence timer on partial results too
            const result = message.result;
            if (result && result.partial && result.partial.trim()) {
                // Clear no-speech timer on partial results
                if (noSpeechTimer) {
                    clearTimeout(noSpeechTimer);
                    noSpeechTimer = null;
                }

                lastSpeechTime = Date.now();
                resetSilenceTimer();
            }
        });

        voskLoaded = true;
        console.log('Vosk speech model loaded successfully');

        // Update the loading message
        const msgBubble = voskLoadingMessage.bubble;
        if (msgBubble) {
            setBubbleContent(msgBubble, 'Offline speech model ready! Please try your voice input again.');
        }
        voskLoadingMessage = null;

        sendBtn.disabled = false;
        textInput.disabled = false;
        micBtn.disabled = false;

        return true;
    } catch (error) {
        console.error('Failed to load Vosk model:', error);
        voskLoadingFailed = true;

        // Remove the loading message if it exists
        if (voskLoadingMessage && voskLoadingMessage.message) {
            voskLoadingMessage.message.remove();
            voskLoadingMessage = null;
        }

        sendBtn.disabled = false;
        textInput.disabled = false;
        micBtn.disabled = false;
        return false;
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Shows the cancel loading link
 */
function showCancelLink() {
    const cancelLink = document.getElementById('cancel-loading-link');
    if (cancelLink && !modelLoadingCancelled) {
        cancelLink.style.display = 'inline-block';
    }
}

/**
 * Reverses a string character by character
 * @param {string} text - Text to reverse
 * @returns {string} Reversed text
 */
function reverseWord(text) {
    return text.split('').reverse().join('');
}

/**
 * Shifts characters in a string by a specified amount (Caesar cipher)
 * @param {string} text - Text to shift
 * @param {number} amount - Amount to shift each character code
 * @returns {string} Shifted text
 */
function shiftWord(text, amount) {
    return text
        .split('')
        .map(char => String.fromCharCode(char.charCodeAt(0) + amount))
        .join('');
}

/**
 * Escapes special regex characters in a string
 * @param {string} text - Text to escape
 * @returns {string} Escaped text safe for use in RegExp
 */
function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Escapes HTML special characters and converts newlines to <br> tags
 * @param {string} text - Text to escape
 * @returns {string} HTML-safe text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

/**
 * Trims incomplete sentences from AI-generated text that was cut off mid-sentence.
 * Only trims if the response ends with incomplete markers (comma, dash, "and", etc.).
 * Preserves lists and responses ending with proper names or numbers.
 * @param {string} text - Text to trim
 * @returns {string} Text with incomplete trailing sentence removed, or original text
 */
function trimIncompleteSentence(text) {
    if (!text || text.length <= 20) {
        return text;
    }

    // If text already ends with sentence-ending punctuation, it's complete
    if (/[.!?]$/.test(text)) {
        return text;
    }

    // Text doesn't end with sentence-ending punctuation - trim to the last complete sentence
    const lastCompleteMatch = text.match(/([\s\S]*[.!?])/);
    if (lastCompleteMatch) {
        const trimmed = lastCompleteMatch[1].trim();
        if (trimmed.length >= 20) {
            console.log(`Trimmed incomplete sentence: ${text.length} -> ${trimmed.length} chars`);
            return trimmed;
        }
    }

    return text;
}

// ============================================================================
// CONTENT MODERATION
// ============================================================================

/**
 * Loads and decodes the inappropriate words list for content filtering
 * @throws {Error} If file cannot be loaded
 */
async function loadInappropriateWords() {
    try {
        const response = await fetch('./moderation/mod.txt');
        if (!response.ok) throw new Error('Failed to load inappropriate words');

        const encodedWordsText = await response.text();
        inappropriateWords = encodedWordsText
            .split(/\r?\n/)
            .map(word => word.trim())
            .filter(word => word.length > 0)
            .map(word => shiftWord(reverseWord(word.toLowerCase()), 1));

        console.log('Loaded inappropriate words:', inappropriateWords.length);
    } catch (error) {
        console.error('Error loading inappropriate words:', error);
        throw error;
    }
}

// ============================================================================
// IMAGE CLASSIFICATION MODEL LOADING
// ============================================================================

/**
 * Ensures the image classification model (MobileNet + classifier) is loaded.
 * Coalesces concurrent callers to prevent duplicate loads.
 * @returns {Promise<void>} Resolves when model is ready
 */
function ensureImageModelLoaded() {
    if (mobilenetReady) return Promise.resolve();
    if (mobilenetLoadPromise) return mobilenetLoadPromise;
    updateLoadingStatus('mobilenet', 'loading', 'Loading...');
    mobilenetLoadPromise = loadModel()
        .catch(err => {
            mobilenetLoadPromise = null;
            updateLoadingStatus('mobilenet', 'error', 'Failed');
            throw err;
        });
    return mobilenetLoadPromise;
}

/**
 * Loads the MobileNet base model and custom classifier for image classification.
 * Pins TFJS to CPU backend for the image classifier.
 * @throws {Error} If models fail to load
 */
async function loadModel() {
    try {
        console.log("Loading base model...");
        updateLoadingStatus('mobilenet', 'loading', '25%');
        const mobilenet = await tf.loadLayersModel(BASE_MODEL_URL);
        const layer = mobilenet.getLayer('conv_pw_13_relu');
        featureExtractor = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        console.log("Base model loaded.");
        updateLoadingStatus('mobilenet', 'loading', '50%');

        console.log("Loading classifier model...");
        model = await tf.loadLayersModel(MODEL_URL);
        console.log("Classifier model loaded");
        updateLoadingStatus('mobilenet', 'loading', '75%');

        // Warmup is only useful to prime GPU kernels. On the CPU backend
        // (which we pin in init() to preserve VRAM for the WebGPU LLM) it
        // just wastes time and RAM, so skip it there.
        if (tf.getBackend() !== 'cpu') {
            console.log("Warming up model...");
            tf.tidy(() => {
                const dummyInput = tf.zeros([1, 224, 224, 3]);
                const dummyFeatures = featureExtractor.predict(dummyInput);
                model.predict(dummyFeatures);
            });
        }
        console.log("Model ready!");
        updateLoadingStatus('mobilenet', 'ready', '100%');
        mobilenetReady = true;
    } catch (e) {
        console.warn("Standard load failed, trying as GraphModel...", e);
        try {
            model = await tf.loadGraphModel(MODEL_URL);
            updateLoadingStatus('mobilenet', 'ready', '100%');
            mobilenetReady = true;
        } catch (e2) {
            console.error(e2);
            updateLoadingStatus('mobilenet', 'error', 'Failed');
            throw new Error(`Failed to load model from ${MODEL_URL}. \nLayers Error: ${e.message}\nGraph Error: ${e2.message}`);
        }
    }
}

/**
 * Initializes the Wllama language model for CPU mode text generation
 * @throws {Error} If Wllama initialization fails
 */
async function initWllama(progressCallback = null) {
    try {
        // Check if already initialized
        if (wllama) {
            console.log('Wllama already initialized');
            return;
        }

        console.log("Initializing wllama...");
        updateModelName('Phi 3.5-mini (AI mode)');
        updateLoadingStatus('phi', 'loading', '10%');

        // Configure WASM paths for CDN
        const CONFIG_PATHS = {
            default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/wasm/wllama.wasm',
        };

        const internalProgressCallback = ({ loaded, total }) => {
            // Check if user cancelled loading
            if (modelLoadingCancelled) {
                return;
            }

            const progress = loaded / total;
            const percentage = Math.round(progress * 100);
            const adjustedProgress = Math.round(20 + (percentage * 0.8)); // 20% to 100%
            updateLoadingStatus('phi', 'loading', `${adjustedProgress}%`);
            console.log(`Loading wllama: ${percentage}%`);

            // Show cancel link when loading starts
            showCancelLink();

            if (progressCallback) {
                progressCallback(progress);
            }
        };

        // Try multithreaded first if cross-origin isolated, fall back to single-threaded
        const useMultiThread = window.crossOriginIsolated === true;
        const availableThreads = navigator.hardwareConcurrency || 4;
        const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;
        console.log(`Cross-origin isolated: ${window.crossOriginIsolated}, available threads: ${availableThreads}, attempting ${preferredThreads} thread(s)`);

        const modelRef = {
            //repo: 'unsloth/Phi-4-mini-instruct-GGUF',
            repo: 'bartowski/Phi-3.5-mini-instruct-GGUF',
            quant: 'Q4_K_M'
        };

        // Detect GPU vendor; disable WebGPU for known-broken implementations or
        // if a previous GPU session crashed (gpuFailed=true).
        let GPU_ENABLED = !gpuFailed && !!navigator.gpu;
        if (GPU_ENABLED) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    // Chrome 121+: adapter.info is synchronous. Older: requestAdapterInfo().
                    const info = adapter.info ?? await adapter.requestAdapterInfo?.();
                    const vendor = (info?.vendor || '').toLowerCase();
                    if (vendor.includes('qualcomm') || vendor.includes('adreno')) {
                        // Open bug: ggml-org/llama.cpp#23558 — still unresolved upstream.
                        console.warn(`WebGPU disabled: Qualcomm/Adreno GPU detected (vendor="${info?.vendor}") — known precision issues cause hallucinations`);
                        GPU_ENABLED = false;
                    } else if (vendor.includes('amd') || vendor.includes('advanced micro')) {
                        // Fixed in llama.cpp PR #23040 (wllama 3.2.3+), but this app uses 3.1.1
                        // which predates the fix. Fall back to CPU until wllama is upgraded.
                        console.warn(`WebGPU disabled: AMD GPU detected (vendor="${info?.vendor}") — flashattention bug in wllama <3.2.3 causes garbled output on Linux/Vulkan`);
                        GPU_ENABLED = false;
                    }
                } else {
                    GPU_ENABLED = false; // requestAdapter returned null — no WebGPU
                }
            } catch (e) {
                console.warn('Could not query WebGPU adapter info:', e);
                GPU_ENABLED = false;
            }
        }

        const baseParams = {
            n_ctx: 712,
            progressCallback: internalProgressCallback
        };

        // Helper: create a fresh Wllama instance and load the model.
        const attemptLoad = async (n_gpu_layers, n_threads) => {
            wllama = new Wllama(CONFIG_PATHS);
            await wllama.loadModelFromHF(modelRef, { ...baseParams, n_gpu_layers, n_threads });
        };

        updateLoadingStatus('phi', 'loading', '20%');

        const loadWithFallback = async () => {
            if (GPU_ENABLED) {
                try {
                    // Full GPU offload (all 32 layers). Full offload avoids the precision
                    // mismatch at CPU/GPU layer boundaries that caused garbled tokens.
                    await attemptLoad(32, preferredThreads);
                    wllamaUsedGPU = true;
                    console.log(`Wllama initialized with GPU (32 layers) + ${preferredThreads} thread(s)`);
                    return;
                } catch (gpuErr) {
                    if (modelLoadingCancelled) throw gpuErr;
                    console.warn(`GPU initialization failed (${gpuErr.message}), falling back to CPU`);
                    if (wllama) { try { await wllama.exit(); } catch (_) { } wllama = null; }
                }
            } else {
                console.log('Skipping GPU: using CPU directly');
            }

            // CPU multi-threaded
            try {
                await attemptLoad(0, preferredThreads);
                wllamaUsedGPU = false;
                console.log(`Wllama initialized on CPU with ${preferredThreads} thread(s)`);
            } catch (cpuErr) {
                if (modelLoadingCancelled) throw cpuErr;
                if (preferredThreads > 1) {
                    console.warn(`Multi-thread CPU init failed (${cpuErr.message}), retrying with 1 thread`);
                    if (wllama) { try { await wllama.exit(); } catch (_) { } wllama = null; }
                    // Final attempt: CPU single-threaded
                    await attemptLoad(0, 1);
                    wllamaUsedGPU = false;
                    console.log('Wllama initialized on CPU with 1 thread');
                } else {
                    throw cpuErr;
                }
            }
        };

        await loadWithFallback();

        // Check if cancelled before finalizing
        if (modelLoadingCancelled) {
            if (wllama) { try { await wllama.exit(); } catch (_) { } wllama = null; }
            throw new Error('Model loading cancelled by user');
        }

        wllamaReady = true;
        updateLoadingStatus('phi', 'ready', '100%');
        console.log(`Wllama initialized successfully (GPU: ${wllamaUsedGPU})`);
    } catch (error) {
        console.error('Failed to initialize wllama:', error);
        if (wllama) { try { await wllama.exit(); } catch (_) { } wllama = null; }
        updateLoadingStatus('phi', 'error', 'Failed');
        wllamaReady = false;
        throw error;
    }
}

/**
 * Updates the model name in the loading status
 * @param {string} modelName - The name of the model being loaded
 */
function updateModelName(modelName) {
    const statusTextElement = document.getElementById('phiStatusText');
    if (statusTextElement) {
        statusTextElement.textContent = modelName;
    }
}

/**
 * Updates the loading status display for a specific model
 * @param {string} modelType - Either 'mobilenet' or 'phi'
 * @param {string} status - One of 'loading', 'ready', or 'error'
 * @param {string} progress - Progress text to display
 */
function updateLoadingStatus(modelType, status, progress) {
    const statusId = modelType === 'mobilenet' ? 'mobilenetStatus' : 'phiStatus';
    const progressId = modelType === 'mobilenet' ? 'mobilenetProgress' : 'phiProgress';

    const statusElement = document.getElementById(statusId);
    const progressElement = document.getElementById(progressId);

    if (!statusElement || !progressElement) return;

    const iconSpan = statusElement.querySelector('.status-icon');

    if (status === 'loading') {
        iconSpan.textContent = '⏳';
        statusElement.classList.remove('ready', 'error');
        statusElement.classList.add('loading');
    } else if (status === 'ready') {
        iconSpan.textContent = '✓';
        statusElement.classList.remove('loading', 'error');
        statusElement.classList.add('ready');
    } else if (status === 'error') {
        iconSpan.textContent = '✗';
        statusElement.classList.remove('loading', 'ready');
        statusElement.classList.add('error');
    }

    progressElement.textContent = progress;
}

/**
 * Hides the loading overlay with a fade-out animation
 */
function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 500);
    }
}

/**
 * Adds a message to the chat interface
 * @param {string} text - The message text
 * @param {string} sender - Either 'user' or 'bot'
 * @param {string|null} imageUrl - Optional image URL to display
 */
function setBubbleContent(bubble, text) {
    if (typeof DOMPurify !== 'undefined') {
        bubble.innerHTML = DOMPurify.sanitize(text, {
            ALLOWED_TAGS: ['b', 'i', 'br', 'small', 'a', 'div', 'p'],
            ALLOWED_ATTR: ['href', 'target', 'style', 'class'],
            ALLOW_DATA_ATTR: false
        });
    } else {
        bubble.textContent = text;
    }
}

function addMessage(text, sender, imageUrl = null, options = {}) {
    const { deferCompletion = false } = options;
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.setAttribute('role', 'article');
    div.setAttribute('aria-label', `${sender === 'user' ? 'User' : 'Assistant'} message`);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    setBubbleContent(bubble, text);

    if (imageUrl) {
        const br = document.createElement('br');
        const img = document.createElement('img');
        img.src = imageUrl;
        img.className = 'message-image';
        img.alt = 'Uploaded image';
        img.onload = scrollToBottom;
        bubble.appendChild(br);
        bubble.appendChild(img);
    }

    div.appendChild(bubble);

    const timestamp = document.createElement('span');
    timestamp.className = 'timestamp';
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    div.appendChild(timestamp);

    chatContainer.appendChild(div);
    scrollToBottom();

    // Text-to-Speech for bot responses when input was spoken
    if (sender === 'bot' && isVoiceInput && !deferCompletion) {
        speakText(bubble);
        isVoiceInput = false; // Reset after speaking
    } else if (sender === 'bot' && !deferCompletion) {
        // If no speech, we can end the response state
        endResponse();
    }

    return { message: div, bubble };
}

function getBoardIdentificationMessage(text) {
    const lowerText = String(text || '').toLowerCase();

    if (lowerText.includes('assy 250')) {
        return 'The assembly number indicates that the board may have come from a Commodore 64.';
    }

    if (lowerText.includes('820-')) {
        return 'Serial numbers beginning 820- are commonly found in Apple computers.';
    }

    if (lowerText.includes('z-80')) {
        return 'The Zilog Z-80 processor is common in Sinclair computers, such as the ZX-80, ZX-81, and ZX Spectrum.';
    }

    if ((lowerText.includes('88-') || lowerText.includes('880-')) && lowerText.includes('mits')) {
        return 'The markings on the board are consistent with an Altair 8800.';
    }

    return "I can't determine what kind of computer this came from.";
}

/**
 * Scrolls the chat container to the bottom
 */
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Animates typing text into a bubble character by character
 * @param {HTMLElement} bubble - The bubble element to type into
 * @param {string} text - The complete HTML text (including any starting text)
 * @param {number} speed - Milliseconds per character (default: 20)
 * @param {string} startingText - Text that's already displayed (won't be animated)
 * @returns {Promise<void>} Resolves when animation completes or is stopped
 */
async function typeTextInBubble(bubble, text, speed = 20, startingText = '') {
    return new Promise((resolve) => {
        typingAnimationsInProgress++;

        const finishTyping = () => {
            typingAnimationsInProgress = Math.max(0, typingAnimationsInProgress - 1);
            resolve();
        };

        let currentIndex = 0;
        let displayText = startingText;

        // If starting text equals full text, nothing to type
        if (startingText === text) {
            finishTyping();
            return;
        }

        // Parse only the new text that comes after startingText
        const newText = text.substring(startingText.length);
        const htmlChunks = [];
        let inTag = false;
        let currentChunk = '';

        for (let i = 0; i < newText.length; i++) {
            const char = newText[i];

            if (char === '<') {
                if (currentChunk && !inTag) {
                    htmlChunks.push({ type: 'text', content: currentChunk });
                    currentChunk = '';
                }
                inTag = true;
                currentChunk += char;
            } else if (char === '>') {
                currentChunk += char;
                if (inTag) {
                    htmlChunks.push({ type: 'tag', content: currentChunk });
                    currentChunk = '';
                    inTag = false;
                }
            } else {
                currentChunk += char;
            }
        }

        if (currentChunk) {
            htmlChunks.push({ type: inTag ? 'tag' : 'text', content: currentChunk });
        }

        const typeInterval = setInterval(() => {
            if (checkStopResponse() || currentIndex >= htmlChunks.length) {
                clearInterval(typeInterval);
                setBubbleContent(bubble, text);
                scrollToBottom();
                finishTyping();
                return;
            }

            const chunk = htmlChunks[currentIndex];

            if (chunk.type === 'tag') {
                // Add entire tag at once
                displayText += chunk.content;
                currentIndex++;
            } else {
                // Type out text character by character
                if (!chunk.charIndex) chunk.charIndex = 0;

                if (chunk.charIndex < chunk.content.length) {
                    displayText += chunk.content[chunk.charIndex];
                    chunk.charIndex++;
                } else {
                    currentIndex++;
                }
            }

            setBubbleContent(bubble, displayText);
            scrollToBottom();

        }, speed);
    });
}

/**
 * Shows a typing indicator in the chat
 */
function showTyping() {
    const div = document.createElement('div');
    div.id = 'typing-indicator';
    div.className = 'message bot';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    // Add CPU mode patience message if in CPU mode
    if (currentMode === 'cpu') {
        bubble.innerHTML = `
            <div class="typing">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
            <p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow one some devices. Thanks for your patience!)</p>
        `;
    } else {
        bubble.innerHTML = `
            <div class="typing">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        `;
    }

    div.appendChild(bubble);
    chatContainer.appendChild(div);
    scrollToBottom();

    // Enable stop button when bot starts responding
    startResponse();
}

/**
 * Removes the typing indicator from the chat
 */
function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

/**
 * Marks the bot as responding (enables stop button)
 */
function startResponse() {
    isResponding = true;
    shouldStopResponse = false;
    sendBtn.classList.add('stop-mode');
    sendBtn.textContent = '⬜';
    sendBtn.title = 'Stop';
}

/**
 * Marks the bot as done responding (disables stop button)
 */
function endResponse() {
    // Only end when both speech and typing animation are finished
    if (!speechSynthesis.speaking && typingAnimationsInProgress === 0) {
        isResponding = false;
        shouldStopResponse = false;
        sendBtn.classList.remove('stop-mode');
        sendBtn.textContent = '▶';
        sendBtn.title = 'Send';
    }
}

/**
 * Checks if the user has requested to stop the response
 * @returns {boolean} True if response should be stopped
 */
function checkStopResponse() {
    return shouldStopResponse;
}

// ============================================================================
// KEYWORD EXTRACTION AND SEARCH CONSTANTS
// ============================================================================

// Stopwords for keyword extraction (common words to filter out)
const STOPWORDS = new Set([
    // Articles, prepositions, conjunctions
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'with',
    'or', 'but', 'if', 'than', 'then', 'so', 'yet',
    'after', 'before', 'between', 'during', 'into', 'through', 'over',
    'under', 'until', 'up', 'down', 'out', 'off', 'above', 'below',
    // Pronouns
    'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'our', 'their', 'i\'m',
    'you\'re', 'he\'s', 'she\'s', 'we\'re', 'they\'re',
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
    'whose', 'whether', 'what\'s', 'whats', 'who\'s', 'whos', 'how\'s',
    'hows',
    // Common adverbs
    'also', 'just', 'now', 'here', 'there', 'very', 'too',
    'really', 'still', 'always', 'never', 'often', 'sometimes', 'maybe',
    'perhaps', 'about',
    // Other common words
    'yes', 'no', 'thing', 'something', 'anything', 'nothing',
    'everything', 'someone', 'anyone', 'everyone', 'understand',
    'think', 'believe', 'feel', 'appear',
    'search', 'look', 'information', 'info',
    'ebay', 'sale', 'buy', 'price', 'cost', 'one'
]);

// Search trigger patterns
const SHOPPING_TRIGGERS = ['ebay', 'for sale', 'buy', 'purchase', 'shop'];
const WEB_SEARCH_TRIGGERS = ['bing', 'search', 'find'];
const SEARCH_TRIGGER_WORDS = new Set([
    ...SHOPPING_TRIGGERS.join(' ').split(/\s+/),
    ...WEB_SEARCH_TRIGGERS.join(' ').split(/\s+/)
]);

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

/**
 * Main message handler - processes text input, images, and commands
 */
async function handleSend() {
    if (isStoppingResponse) {
        return;
    }

    const text = textInput.value.trim();

    // Check if we have a file or text
    if (!text && !pendingFile) return;

    // Clear input immediately
    textInput.value = '';
    textInput.style.height = 'auto';

    // 1. Text Processing
    if (text) {
        addMessage(escapeHtml(text), "user");

        // Check for inappropriate content
        const lowerText = text.toLowerCase();
        const containsInappropriate = inappropriateWords.some(word => {
            const regex = new RegExp(`\\b${escapeRegex(word)}\\b`, 'i');
            return regex.test(lowerText);
        });

        if (containsInappropriate) {
            addMessage("I'm sorry, I can't help with that because it triggered a content-safety filtering policy.\nI can only help with information about the history of computing.", "bot");
            return;
        }

        // Text Analysis (Basic mode only - GPU and CPU modes send to model)
        if (currentMode === 'basic') {
            const lines = text.split('\n');
            const firstLine = lines[0].trim().toLowerCase();

            // Check for Summarization Command
            if (lines.length > 1 && firstLine.startsWith('summarize')) {
                const contentToAnalyze = lines.slice(1).join('\n');
                showTyping();
                // Simulate "reading"
                setTimeout(() => {
                    // Check if user stopped the response
                    if (checkStopResponse()) {
                        removeTyping();
                        return;
                    }

                    const summary = summarizeText(contentToAnalyze);
                    removeTyping();
                    addMessage(`<b>Summary:</b><br><br>${summary}`, "bot");
                }, 1000);

                return;
            }

            // Check for People Extraction Command
            const peoplePattern = /^(list|extract).*(people|persons|names).*in this text:$/i;
            if (lines.length > 1 && peoplePattern.test(firstLine)) {
                const contentToAnalyze = lines.slice(1).join('\n');
                showTyping();
                setTimeout(() => {
                    if (checkStopResponse()) {
                        removeTyping();
                        return;
                    }

                    const doc = nlp(contentToAnalyze);
                    const people = doc.people().out('array');
                    const uniquePeople = [...new Set(people)];

                    let response = "<b>People mentioned:</b><br><br>";
                    if (uniquePeople.length > 0) {
                        response += uniquePeople.join(', ');
                    } else {
                        response += "No people found in the text.";
                    }

                    removeTyping();
                    addMessage(response, "bot");
                }, 1000);

                return;
            }

            // Check for Places/Locations Extraction Command
            const placesPattern = /^(list|extract).*(places|locations).*in this text:$/i;
            if (lines.length > 1 && placesPattern.test(firstLine)) {
                const contentToAnalyze = lines.slice(1).join('\n');
                showTyping();
                setTimeout(() => {
                    if (checkStopResponse()) {
                        removeTyping();
                        return;
                    }

                    const doc = nlp(contentToAnalyze);
                    const places = doc.places().out('array');
                    const uniquePlaces = [...new Set(places)];

                    let response = "<b>Places/Locations mentioned:</b><br><br>";
                    if (uniquePlaces.length > 0) {
                        response += uniquePlaces.join(', ');
                    } else {
                        response += "No places or locations found in the text.";
                    }

                    removeTyping();
                    addMessage(response, "bot");
                }, 1000);

                return;
            }
        }
    }

    // 2. Handle Image
    if (pendingFile) {

        // Read file to data URL for display
        const reader = new FileReader();
        reader.onload = async (event) => {
            const imageUrl = event.target.result;

            addMessage("", "user", imageUrl);

            // Start Analysis
            showTyping();

            try {
                await ensureImageModelLoaded();
            } catch (modelErr) {
                removeTyping();
                addMessage("I'm sorry, the image classifier failed to load. Please try again later.", "bot");
                return;
            }

            // Create an invisible image element for TF.js to read
            const imgEl = new Image();
            imgEl.src = imageUrl;
            imgEl.onload = async () => {
                await performClassification(imgEl, text); // Pass text context
            };
        };
        reader.readAsDataURL(pendingFile);

        // Reset pending
        removeImage();
        return;
    }

    // 3. Text Only Response (language model or search)
    if (tryHandleSearchRequest(text)) {
        return;
    }

    const keywords = extractKeywords(text);
    if (!keywords) {
        addMessage("Please enter a more specific query.", "bot");
        return;
    }

    // CPU and Basic mode response handling
    showTyping();

    try {
        let bubble = null;

        // For CPU mode, create the bubble before calling generateComputingInfo
        // so it can update it with the waiting message if needed
        if (currentMode === 'cpu') {
            removeTyping();
            const msgResult = addMessage('', "bot", null, { deferCompletion: true });
            bubble = msgResult.bubble;
            startResponse();

            // Show initial message with typing indicator and CPU patience note
            let initialMessage = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>' +
                '<p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow on some devices. Thanks for your patience!)</p>';
            setBubbleContent(bubble, initialMessage);
            scrollToBottom();

            const summary = await generateComputingInfo(text, null, bubble, '');

            if (checkStopResponse()) return;

            // Check for wllama failover
            if (wllamaShouldFailoverToBasic && !shouldStopResponse) {
                console.log('Wllama failed, switching to Basic mode and retrying...');
                wllamaShouldFailoverToBasic = false;
                wllamaReady = false;
                currentMode = 'basic';
                availableModes.cpu = false;
                updateModelName('Wikipedia API (Basic)');
                updateModeSelect();

                // Update the bubble with the switch notification
                setBubbleContent(bubble, '<i>I experienced an error using the model on this device; so I\'m switching to Basic mode...</i>');
                await new Promise(resolve => setTimeout(resolve, 800));

                // Create a NEW bubble for the Wikipedia response (keep error message visible)
                const retryMsgResult = addMessage('', "bot", null, { deferCompletion: true });
                const retryBubble = retryMsgResult.bubble;

                // Show typing indicator in the new bubble
                setBubbleContent(retryBubble, '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>');
                scrollToBottom();

                const retryText = await generateComputingInfo(text, null, retryBubble, '');

                if (checkStopResponse()) return;

                if (retryText) {
                    // Display the Wikipedia response in the new bubble
                    await typeTextInBubble(retryBubble, escapeHtml(retryText), 20);

                    if (checkStopResponse()) return;

                    // Handle voice output if needed
                    if (isVoiceInput) {
                        speakText(retryBubble);
                        isVoiceInput = false;
                    }
                    endResponse();

                    // Store in conversation history
                    conversationHistory = [
                        { user: truncateToFirstSentence(text), assistant: truncateToFirstSentence(retryText) }
                    ];
                    return;
                } else {
                    setBubbleContent(retryBubble, `I'm sorry. I don't know about that topic.`);
                    endResponse();
                    return;
                }
            }

            if (summary) {
                // Summary already streamed to bubble, just finalize
                // Handle voice output if needed
                if (isVoiceInput) {
                    speakText(bubble);
                    isVoiceInput = false;
                }

                endResponse();
            } else {
                if (lastWllamaCompletionErrored) {
                    setBubbleContent(bubble, CPU_MODE_FAILURE_MESSAGE);
                } else {
                    setBubbleContent(bubble, `I'm sorry. I don't know about that topic.`);
                }
                endResponse();
            }
        } else {
            // Basic mode - original behavior
            const summary = await generateComputingInfo(text);
            removeTyping();

            if (checkStopResponse()) return;

            if (summary) {
                bubble = addMessage('', "bot", null, { deferCompletion: true }).bubble;
                startResponse();

                // In voice mode, speak immediately while the text animates.
                if (isVoiceInput) {
                    speakTextContent(summary);
                    isVoiceInput = false;
                }

                // Animate the response text (escape HTML and convert newlines to <br>)
                await typeTextInBubble(bubble, escapeHtml(summary), 20);

                if (checkStopResponse()) return;

                endResponse();
            } else {
                addMessage(`I'm sorry. I don't know about that topic.`, "bot");
            }
        }

        // Store in conversation history (if we got a summary)
        if (bubble && bubble.textContent && bubble.textContent.trim().length > 0) {
            const summary = bubble.textContent.trim();
            conversationHistory = [
                { user: truncateToFirstSentence(text), assistant: truncateToFirstSentence(summary) }
            ];
        }
    } catch (e) {
        removeTyping();
        if (checkStopResponse()) return;
        addMessage("Sorry, I had trouble searching via text. " + escapeHtml(e.message), "bot");
    }
}

/**
 * Extracts keywords from text by removing stopwords
 * @param {string} text - The text to extract keywords from
 * @returns {string} Space-separated keywords
 */
function extractKeywords(text, excludedWords = null) {
    // Remove punctuation and split
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);

    // Filter using global STOPWORDS
    return words.filter(w => !STOPWORDS.has(w) && w.length > 0 && !(excludedWords && excludedWords.has(w))).join(' ');
}

function getSearchIntent(text) {
    const lowerText = text.toLowerCase();

    if (hasBoundaryKeyword(lowerText, SHOPPING_TRIGGERS)) {
        return 'shopping';
    }

    if (hasBoundaryKeyword(lowerText, WEB_SEARCH_TRIGGERS)) {
        return 'web';
    }

    return null;
}

function tryHandleSearchRequest(text, options = {}) {
    const { queryOverride = null, prefixMessage = null } = options;
    const searchIntent = getSearchIntent(text);

    if (!searchIntent) {
        return false;
    }

    const searchKeywords = queryOverride || extractKeywords(text, SEARCH_TRIGGER_WORDS);
    if (!searchKeywords) {
        addMessage(
            searchIntent === 'shopping'
                ? "Please enter a more specific shopping query."
                : "Please enter a more specific search query.",
            "bot"
        );
        return true;
    }

    if (prefixMessage) {
        addMessage(prefixMessage, "bot");
    }

    const isShoppingSearch = searchIntent === 'shopping';
    const url = isShoppingSearch
        ? `https://www.bing.com/shop/topics?q=${searchKeywords.replace(/ /g, '+')}`
        : `https://www.bing.com/search?q=${searchKeywords.replace(/ /g, '+')}`;

    addMessage(
        isShoppingSearch
            ? `Searching Bing Shopping for <b>"${searchKeywords}"</b>...`
            : `Searching Bing for <b>"${searchKeywords}"</b>...`,
        "bot"
    );

    setTimeout(() => {
        if (!checkStopResponse()) {
            addMessage(
                isShoppingSearch
                    ? `Here's what I found: <a href="${url}" target="_blank" style="color: #64185e; text-decoration: underline;">Click here to see results for ${searchKeywords}</a>`
                    : `Here's what I found: <a href="${url}" target="_blank" style="color: #64185e; text-decoration: underline;">Click here to see results for ${searchKeywords}</a>`,
                "bot"
            );
        }
    }, 600);

    return true;
}

function hasBoundaryKeyword(text, keywords) {
    return keywords.some(keyword => {
        const pattern = keyword
            .split(/\s+/)
            .map(escapeRegex)
            .join('\\s+');

        const regex = new RegExp(
            `(^\\s*${pattern}(?=$|\\s|[.?!:]))|([.?!:]\\s*${pattern}(?=$|\\s|[.?!:]))|( ${pattern}(?= ))|(\\b${pattern}(?=[.?!:]))`
        );

        return regex.test(text);
    });
}

/**
 * Extracts up to maxSentences from the start of text, treating periods between digits
 * (for example, 12.5) as part of a number instead of sentence boundaries.
 * @param {string} text - Source text
 * @param {number} maxSentences - Maximum number of sentences to include
 * @returns {string} Extracted leading sentences, or original trimmed text if no boundary found
 */
function extractLeadingSentences(text, maxSentences = 1) {
    if (!text) return '';

    let sentenceCount = 0;
    let cutIndex = -1;

    for (let i = 0; i < text.length; i++) {
        const char = text[i];

        if (char === '.' || char === '!' || char === '?') {
            const prev = i > 0 ? text[i - 1] : '';
            const next = i < text.length - 1 ? text[i + 1] : '';

            // Ignore decimal points like 12.5
            if (char === '.' && /\d/.test(prev) && /\d/.test(next)) {
                continue;
            }

            sentenceCount++;
            cutIndex = i + 1;

            if (sentenceCount >= maxSentences) {
                break;
            }
        }
    }

    if (cutIndex === -1) {
        return text.trim();
    }

    return text.slice(0, cutIndex).trim();
}

/**
 * Extracts the first sentence from a given text string
 * @param {string} text - The text to truncate
 * @returns {string} The first sentence or first 100 characters
 */
function truncateToFirstSentence(text) {
    const firstSentence = extractLeadingSentences(text, 1);
    if (firstSentence) return firstSentence;

    // No sentence-ending punctuation found, take first 100 characters
    return text.substring(0, 100).trim();
}

// ============================================================================
// TEXT ANALYSIS FUNCTIONS
// ============================================================================

/**
 * Summarizes text using TextRank algorithm
 * @param {string} text - The text to summarize
 * @returns {string} Summary of top 3 sentences
 */
function summarizeText(text) {
    // 1. Split into sentences (simple approximation)
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

    if (sentences.length <= 3) return sentences.join(' ');

    // 2. Tokenize sentences
    const tokenizedSentences = sentences.map(s => {
        return s.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => !STOPWORDS.has(w) && w.length > 0);
    });

    // 3. Build Similarity Matrix (Graph)
    // We'll calculate score for each sentence based on overlaps with others
    const scores = new Array(sentences.length).fill(0);

    for (let i = 0; i < sentences.length; i++) {
        for (let j = 0; j < sentences.length; j++) {
            if (i === j) continue;

            const wordsI = new Set(tokenizedSentences[i]);
            const wordsJ = new Set(tokenizedSentences[j]);

            // Jaccard similarity or simple intersection
            // TextRank uses intersection / (log(|Si|) + log(|Sj|))
            let intersection = 0;
            for (let w of wordsI) {
                if (wordsJ.has(w)) intersection++;
            }

            if (intersection > 0) {
                const norm = Math.log(wordsI.size || 1) + Math.log(wordsJ.size || 1);
                // Prevent div by zero if empty
                if (norm > 0) {
                    scores[i] += intersection / norm;
                }
            }
        }
    }

    // 4. Sort and Pick Top 3
    // We want to keep original order for readability, so we'll pick indices
    const indicesWithScores = scores.map((score, i) => ({ index: i, score }));
    indicesWithScores.sort((a, b) => b.score - a.score);

    const topIndices = indicesWithScores.slice(0, 3).map(item => item.index).sort((a, b) => a - b);

    return topIndices.map(i => sentences[i].trim()).join(' ');
}

// ============================================================================
// IMAGE UPLOAD AND CLASSIFICATION
// ============================================================================

/**
 * Handles image file selection from input
 * @param {Event} e - The change event
 */
function handleImageInput(e) {
    const file = e.target.files[0];
    if (!file) return;

    pendingFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (event) => {
        document.getElementById('preview-img').src = event.target.result;
        document.getElementById('preview-container').classList.remove('hidden');
        textInput.focus();
    };
    reader.readAsDataURL(file);

    // Reset file input so same file can be selected again
    imageUpload.value = '';
}

/**
 * Removes the selected image and hides preview
 */
function removeImage() {
    pendingFile = null;
    document.getElementById('preview-img').src = '';
    document.getElementById('preview-container').classList.add('hidden');
    imageUpload.value = '';
}

/**
 * Performs image classification and generates response
 * @param {HTMLImageElement} imgEl - The image element to classify
 * @param {string} userText - Optional user text for context
 */
async function performClassification(imgEl, userText = "") {
    try {
        const result = await classify(imgEl);
        removeTyping();

        if (checkStopResponse()) return;

        const topMatch = result[0];
        const classIndex = CLASSES.indexOf(topMatch.className);

        // Class 6: Unknown - Simple "don't know" message
        if (classIndex === 6) {
            addMessage("I'm sorry. I don't know what this is. I can only recognize computers and circuit boards.", "bot");
            return;
        }

        // Class 5: Printed Circuit Board - Show prediction + OCR
        if (classIndex === 5) {
            const confidence = (topMatch.probability * 100).toFixed(1);
            const reply = `I am <b>${confidence}%</b> sure this is a <b>${topMatch.className}</b>.<br><small>Scanning for text...</small>`;

            // Perform OCR using the same approach as info-extractor
            startResponse();
            const { bubble } = addMessage(reply, "bot", null, { deferCompletion: true });
            try {
                console.log('Starting OCR for image');

                // Initialize Tesseract with progress tracking (same as info-extractor)
                const worker = await Tesseract.createWorker('eng', 1, {
                    logger: m => {
                        console.log('Tesseract log:', m);
                    }
                });

                // Perform OCR directly on the image element (same as info-extractor)
                console.log('Performing OCR recognition...');
                const result = await worker.recognize(imgEl);
                const data = result.data;

                console.log('OCR completed. Text length:', data.text?.length || 0);
                console.log('OCR text preview:', data.text?.substring(0, 200) || 'No text');

                // Clean up worker (same as info-extractor)
                await worker.terminate();

                removeTyping();

                if (checkStopResponse()) return;

                // Use the extracted text
                const rawText = data.text || '';
                const baseReply = `I am <b>${confidence}%</b> sure this is a <b>${topMatch.className}</b>.`;
                let ocrMessage = '';
                let boardIdMessage = '';

                if (!rawText || rawText.trim().length === 0) {
                    ocrMessage = `I couldn't extract any text from the board.`;
                    boardIdMessage = getBoardIdentificationMessage('');
                } else {
                    // Clean the text - preserve hyphens, underscores, dots for part numbers
                    const cleanText = rawText
                        .split(/\s+/)
                        .map(word => word.replace(/[^a-zA-Z0-9\-_.]/g, ''))
                        .filter(word => {
                            const alphanumeric = word.replace(/[\-_.]/g, '');
                            return alphanumeric.length >= 2;
                        })
                        .join(' ')
                        .trim();

                    // Validate: require at least 3 total alphanumeric characters
                    if (cleanText && cleanText.replace(/[^a-zA-Z0-9]/g, '').length >= 3) {
                        ocrMessage = `There are details printed on the board.`;
                        boardIdMessage = getBoardIdentificationMessage(cleanText);
                    } else {
                        ocrMessage = `I couldn't extract any text from the board.`;
                        boardIdMessage = getBoardIdentificationMessage('');
                    }
                }

                // Update bubble to show confidence message without scanning indicator
                setBubbleContent(bubble, baseReply);
                scrollToBottom();

                // Type only the OCR results below the confidence message
                const ocrResults = `<br><br>${ocrMessage}<br><br>${boardIdMessage}`;
                await typeTextInBubble(bubble, baseReply + ocrResults, 20, baseReply);

                if (isVoiceInput) {
                    speakText(bubble);
                    isVoiceInput = false;
                } else {
                    endResponse();
                }
            } catch (e) {
                console.error("OCR Failed", e);
                removeTyping();
                if (!checkStopResponse()) {
                    const baseReply = `I am <b>${confidence}%</b> sure this is a <b>${topMatch.className}</b>.`;

                    // Update bubble to show confidence message
                    setBubbleContent(bubble, baseReply);
                    scrollToBottom();

                    // Type only the error message and identification
                    const errorResults = `<br><br>I couldn't extract any text from the board.<br><br>${getBoardIdentificationMessage('')}`;
                    await typeTextInBubble(bubble, baseReply + errorResults, 20, baseReply);

                    if (isVoiceInput) {
                        speakText(bubble);
                        isVoiceInput = false;
                    } else {
                        endResponse();
                    }
                }
            }
            return;
        }

        // Standard prediction message for all other classes
        const confidence = (topMatch.probability * 100).toFixed(1);
        let reply = `I am <b>${confidence}%</b> sure this is a <b>${topMatch.className}</b>.`;

        // Add secondary guess if close
        if (result[1] && result[1].probability > 0.1) {
            reply += `<br><small>Second guess: ${result[1].className} (${(result[1].probability * 100).toFixed(1)}%)</small>`;
        }

        // Classes 0, 1, 2, 3, 4: eBay search or information
        if ([0, 1, 2, 3, 4].includes(classIndex)) {
            if (tryHandleSearchRequest(userText, {
                queryOverride: topMatch.className,
                prefixMessage: reply
            })) {
                return;
            }

            // AI-generated info for classes 0, 1, 2, 3
            if ([0, 1, 2, 3].includes(classIndex)) {
                // CPU and Basic modes
                const { bubble } = addMessage('', "bot", null, { deferCompletion: true });
                startResponse();

                try {
                    // Step 1: Type the classification message slowly
                    await typeTextInBubble(bubble, reply, 30);

                    if (checkStopResponse()) {
                        return;
                    }

                    // Step 2: Add "I'm researching details..." and thinking dots (with CPU patience message if CPU mode)
                    let researchingMessage = reply + '<br><br>I\'m researching details...<br><br>' +
                        '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';

                    if (currentMode === 'cpu') {
                        researchingMessage += '<p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow in AI mode. Thanks for your patience!)</p>';
                    }

                    setBubbleContent(bubble, researchingMessage);
                    scrollToBottom();

                    const historyUserPrompt = `Tell me about the ${topMatch.className} computer`;
                    const infoPrompt = buildClassInfoPrompt(classIndex);
                    const modelQuery = currentMode === 'basic'
                        ? topMatch.className
                        : (infoPrompt || topMatch.className);
                    console.log('[Image Classification] Sending query to model:', modelQuery);

                    // Step 3: Generate the response
                    // For CPU mode, pass bubble and prefix so waiting message can be shown
                    const summary = await generateComputingInfo(
                        modelQuery,
                        null,
                        currentMode === 'cpu' ? bubble : null,
                        currentMode === 'cpu' ? reply + '<br><br>' : ''
                    );

                    if (checkStopResponse()) {
                        return;
                    }

                    // Check for wllama failover (same as text-only flow)
                    if (wllamaShouldFailoverToBasic && !shouldStopResponse) {
                        console.log('Wllama failed during image classification, switching to Basic mode and retrying...');
                        wllamaShouldFailoverToBasic = false;
                        wllamaReady = false;
                        currentMode = 'basic';
                        availableModes.cpu = false;
                        updateModelName('Wikipedia API (Basic)');
                        updateModeSelect();

                        // Update the bubble with the switch notification
                        setBubbleContent(bubble, reply + '<br><br><i>I experienced an error using the model on this device; so I\'m switching to Basic mode...</i>');
                        await new Promise(resolve => setTimeout(resolve, 800));

                        // Show typing indicator in the same bubble
                        setBubbleContent(bubble, reply + '<br><br><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>');
                        scrollToBottom();

                        // Retry with Basic mode
                        const retryQuery = topMatch.className; // Basic mode uses simple query
                        const retrySummary = await generateComputingInfo(retryQuery, null, null, '');

                        if (checkStopResponse()) {
                            return;
                        }

                        if (retrySummary) {
                            // Update bubble with final Wikipedia response
                            const finalMessage = reply + `<br><br>${escapeHtml(retrySummary)}`;
                            setBubbleContent(bubble, finalMessage);
                            scrollToBottom();

                            conversationHistory = [
                                { user: historyUserPrompt, assistant: truncateToFirstSentence(retrySummary) }
                            ];

                            // Handle voice output if needed
                            if (isVoiceInput) {
                                speakText(bubble);
                                isVoiceInput = false;
                            } else {
                                endResponse();
                            }
                            return;
                        } else {
                            setBubbleContent(bubble, reply + `<br><br>I'm sorry. I don't know about that topic.`);
                            endResponse();
                            return;
                        }
                    }

                    if (summary) {
                        // For Basic mode, update bubble with final result
                        // For CPU mode, already streamed to bubble
                        if (currentMode === 'basic') {
                            const finalMessage = reply + `<br><br>${escapeHtml(summary)}`;
                            setBubbleContent(bubble, finalMessage);
                            scrollToBottom();
                        }

                        conversationHistory = [
                            { user: historyUserPrompt, assistant: truncateToFirstSentence(summary) }
                        ];
                    }

                    // Handle voice output if needed
                    if (isVoiceInput) {
                        speakText(bubble);
                        isVoiceInput = false;
                    } else {
                        endResponse();
                    }
                } catch (e) {
                    console.warn("Info generation failed", e);
                    endResponse();
                }
                return; // Exit early since we already added the message
            }

            // Class 4: Computer - Add uncertainty message
            if (classIndex === 4) {
                reply += `<br><br>Unfortunately, I'm not sure what kind of computer this is.`;
            }
        }

        addMessage(reply, "bot");

    } catch (err) {
        removeTyping();
        addMessage("Oops, I had trouble analyzing that image. " + err.message, "bot");
        console.error(err);
    }
}

// Cap incoming queries at a safe length and cut at a word boundary.
const MAX_QUERY_CHARS = 1200;

function clampQueryLength(query) {
    if (typeof query !== 'string' || query.length <= MAX_QUERY_CHARS) {
        return query;
    }
    const slice = query.slice(0, MAX_QUERY_CHARS);
    const lastSpace = slice.lastIndexOf(' ');
    const cut = lastSpace > MAX_QUERY_CHARS - 100 ? slice.slice(0, lastSpace) : slice;
    console.log(`Truncated query from ${query.length} to ${cut.length} chars`);
    return cut.trim();
}

// ============================================================================
// AI TEXT GENERATION
// ============================================================================

/**
 * Generates computing-related information using AI (Wllama or Wikipedia)
 * @param {string} query - The query to generate information about
 * @param {*} _onChunk - Unused (kept for call-site compatibility)
 * @param {HTMLElement} bubbleElement - Optional bubble element for CPU mode waiting message
 * @param {string} bubblePrefix - Optional HTML prefix for bubble (e.g., classification result)
 * @returns {Promise<string|null>} Generated text or null if unavailable
 */
async function generateComputingInfo(query, _onChunk = null, bubbleElement = null, bubblePrefix = '') {
    const safeQuery = clampQueryLength(query);

    if (currentMode === 'cpu' && wllamaReady && wllama) {
        return await generateWithWllama(safeQuery, bubbleElement, bubblePrefix);
    } else if (currentMode === 'basic') {
        return await generateWithWikipedia(safeQuery);
    } else {
        console.warn("No AI engine ready, skipping generation");
        return null;
    }
}

/**
 * Generates text using Wllama (CPU mode)
 * @param {string} query - The query to generate information about
 * @param {HTMLElement} bubbleElement - Optional bubble element to update with waiting message
 * @param {string} bubblePrefix - Optional HTML prefix to preserve in bubble (e.g., classification result)
 */
/**
 * Generates text using Wllama (CPU mode)
 * @param {string} query - The query to generate information about
 * @param {HTMLElement} bubbleElement - Optional bubble element to update with waiting message
 * @param {string} bubblePrefix - Optional HTML prefix to preserve in bubble (e.g., classification result)
 */
async function generateWithWllama(query, bubbleElement = null, bubblePrefix = '') {
    let slowResponseTimeout;
    let stallDetectionTimer = null;  // Fires if no new token arrives mid-stream
    let stalledMidStream = false;    // Set when the stall timer fires
    let responseText = '';           // Declared here so catch block can read partial content
    try {
        lastWllamaCompletionErrored = false;

        // Build messages array
        const messages = [
            { role: 'system', content: SYSTEM_PROMPT }
        ];

        // Only include previous exchange when the query looks like a follow-up
        // (contains a pronoun or demonstrative that refers back to prior context)
        const isFollowUp = /\b(it|its|they|their|them|that|this|those|these|he|she|him|her)\b/i.test(query);
        if (isFollowUp && conversationHistory.length > 0) {
            const lastExchange = conversationHistory[conversationHistory.length - 1];
            messages.push({ role: 'user', content: lastExchange.user });
            messages.push({ role: 'assistant', content: lastExchange.assistant });
        }

        messages.push({ role: 'user', content: query });

        console.log('Generating info with Wllama for:', query);

        // Debug mode: force failure if requested
        if (debugConfig.enabled && debugConfig.forceWllamaGenerationFail) {
            console.log('🧪 DEBUG: Forcing wllama generation failure');
            throw new Error('Debug mode: forced wllama failure');
        }

        // Create AbortController for cancellation
        currentAbortController = new AbortController();

        // Track if first chunk has been received
        let firstChunkReceived = false;
        let waitingMessageShown = false;

        // Set up 20-second timeout for slow responses
        slowResponseTimeout = setTimeout(() => {
            if (!firstChunkReceived && !shouldStopResponse && bubbleElement) {
                waitingMessageShown = true;
                const waitingHtml = bubblePrefix + '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>' +
                    '<p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">I\'m working on a response. Please continue to wait...</p>';
                setBubbleContent(bubbleElement, waitingHtml);
                scrollToBottom();
                console.log('Wllama slow response: showing waiting message after 20 seconds');
            }
        }, 20000);

        // Generate response using createChatCompletion with streaming
        const completion = await wllama.createChatCompletion({
            messages,
            max_tokens: 512,
            temperature: 0.3,
            top_k: 30,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            cache_prompt: false,
            stop: ['\n\n', '\nUser:', '\nUser :', 'User:', 'User :', '\nAssistant:', 'Assistant:'],
            abortSignal: currentAbortController.signal,
            stream: true
        });

        // Store stream reference for cleanup
        currentStream = completion;

        // Throttle DOM updates to the browser's render cycle to prevent
        // layout-reflow stuttering as the chat history grows.
        let rafPending = false;

        for await (const chunk of completion) {
            if (shouldStopResponse) {
                console.log('Wllama generation stopped by user');
                break;
            }
            if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta?.content) {
                const text = chunk.choices[0].delta.content;
                // Clear timeout on first chunk
                if (!firstChunkReceived) {
                    clearTimeout(slowResponseTimeout);
                    firstChunkReceived = true;
                    // If we showed the waiting message, clear it before showing the real response
                    if (waitingMessageShown && bubbleElement) {
                        setBubbleContent(bubbleElement, bubblePrefix);
                    }
                }

                // Reset the mid-stream stall timer on every content chunk.
                // If no new token arrives within 30 s the generation is aborted
                // and whatever partial text was received is returned to the caller.
                if (stallDetectionTimer) clearTimeout(stallDetectionTimer);
                stallDetectionTimer = setTimeout(() => {
                    stalledMidStream = true;
                    console.warn('Wllama stream stalled (no new token for 30s), aborting');
                    if (currentAbortController) {
                        currentAbortController.abort();
                    }
                }, 30000);

                responseText += text;

                // Stream to bubble if provided, throttled to one DOM update per frame
                if (bubbleElement && !rafPending) {
                    rafPending = true;
                    requestAnimationFrame(() => {
                        setBubbleContent(bubbleElement, bubblePrefix + escapeHtml(responseText));
                        scrollToBottom();
                        rafPending = false;
                    });
                }
            }
        }

        currentAbortController = null;
        currentStream = null;

        // Clear both timeouts now that the stream has finished
        clearTimeout(slowResponseTimeout);
        if (stallDetectionTimer) clearTimeout(stallDetectionTimer);

        // If stopped by user, return null
        if (shouldStopResponse) {
            return null;
        }

        // Clean up the response
        responseText = trimIncompleteSentence(responseText.trim());

        // Update bubble with trimmed content to remove any incomplete fragment streamed earlier
        if (bubbleElement && responseText) {
            setBubbleContent(bubbleElement, bubblePrefix + escapeHtml(responseText));
            scrollToBottom();
        }

        if (!responseText || responseText.length < 10) {
            return null;
        }

        console.log('Wllama final response:', responseText);

        return responseText;

    } catch (error) {
        clearTimeout(slowResponseTimeout);
        if (stallDetectionTimer) clearTimeout(stallDetectionTimer);
        currentStream = null;
        if (error.name === 'AbortError') {
            if (stalledMidStream) {
                // Stream stalled after partial content was received.
                // Return whatever we have so the user sees something useful.
                const partial = trimIncompleteSentence(responseText.trim());
                if (partial && partial.length >= 10) {
                    console.warn('Returning partial response from stalled stream');
                    if (bubbleElement) {
                        setBubbleContent(bubbleElement, bubblePrefix + escapeHtml(partial));
                        scrollToBottom();
                    }
                    return partial;
                }
                // No usable partial content – fall through to the error path
                console.warn('Stream stalled with no usable partial response');
                lastWllamaCompletionErrored = true;
                return null;
            }
            console.log('Generation aborted by user');
            lastWllamaCompletionErrored = false;
            return null;
        }
        console.error('Error generating info with Wllama:', error);
        currentAbortController = null;
        lastWllamaCompletionErrored = true;

        // Set failover flag to switch to basic mode
        console.log('Setting failover flag: wllama → basic mode');
        wllamaShouldFailoverToBasic = true;

        // Clean up failed wllama instance
        if (wllama) {
            const _deadWllama = wllama;
            wllama = null;
            _deadWllama.exit().catch(() => { });
        }

        return null;
    }
}

/**
 * Safely stop and drain a wllama stream to leave WASM state clean
 * @param {AsyncIterator} stream - The wllama completion stream to stop
 */
async function safeStopWllamaStream(stream) {
    if (!stream) {
        return;
    }

    // Try to drain a couple pending chunks
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

    // Call return() to properly close the iterator
    try {
        if (typeof stream.return === 'function') {
            await stream.return();
        }
    } catch (error) {
        console.warn('Stream return() failed:', error);
    }
}

/**
 * Generates information using Wikipedia API (Basic mode)
 * Extracts keywords from the query and returns the first paragraph of the best-matching article.
 * @param {string} query - The query to look up
 * @returns {Promise<string|null>} First paragraph of the Wikipedia article, or null
 */
async function generateWithWikipedia(query) {
    try {
        // Use only the first line for keyword extraction (handles multi-line prompts)
        const firstLine = query.split('\n')[0];
        const keywords = extractKeywords(firstLine);
        if (!keywords) return null;

        // Search Wikipedia for a matching article
        const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(keywords)}&format=json&origin=*&srlimit=1`;
        const searchResponse = await fetch(searchUrl);
        if (!searchResponse.ok) throw new Error('Wikipedia search request failed');

        const searchData = await searchResponse.json();
        const results = searchData?.query?.search;
        if (!results || results.length === 0) return null;

        // Fetch the article summary
        const title = results[0].title;
        const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
        const summaryResponse = await fetch(summaryUrl);
        if (!summaryResponse.ok) throw new Error('Wikipedia summary request failed');

        const summaryData = await summaryResponse.json();
        const extract = summaryData?.extract;
        if (!extract || extract.length < 20) return null;

        // Return the first non-empty paragraph, trimmed to 2 sentences for conciseness
        const firstParagraph = extract.split('\n').find(p => p.trim().length > 0) || extract;
        const extractedSentences = extractLeadingSentences(firstParagraph, 2);
        const result = extractedSentences || firstParagraph.substring(0, 300).trim();

        return result.length >= 20 ? result : null;
    } catch (error) {
        console.error('Wikipedia lookup failed:', error);
        return null;
    }
}

/**
 * Classifies an image using TensorFlow.js models
 * @param {HTMLImageElement} imgElement - The image to classify
 * @returns {Promise<Array>} Array of classification results sorted by probability
 */
async function classify(imgElement) {
    if (!model || !featureExtractor) throw new Error("Model not loaded");

    // Use async data extraction for better reliability
    const predictions = tf.tidy(() => {
        // Preprocessing must match Training!
        // Usually MobileNet expects: 224x224, float32, normalized to [-1, 1]
        let tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224]) // MobileNet default
            .toFloat();

        // Normalize: (x / 127.5) - 1
        const offset = tf.scalar(127.5);
        const normalized = tensor.sub(offset).div(offset).expandDims();

        // 1. Extract features
        const features = featureExtractor.predict(normalized);

        // 2. Classify - return predictions tensor (it won't be disposed by tidy)
        return model.predict(features);
    });

    // Extract values asynchronously after tidy, then dispose the predictions tensor
    const values = await predictions.data();
    predictions.dispose();

    // Process results
    const results = Array.from(values).map((p, i) => ({
        className: CLASSES[i] || `Class ${i}`,
        probability: p
    }));

    // Sort descending
    return results.sort((a, b) => b.probability - a.probability);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Send button - handles both send and stop actions
sendBtn.addEventListener('click', () => {
    if (sendBtn.classList.contains('stop-mode')) {
        handleStopResponse();
    } else {
        handleSend();
    }
});

// Text input - submit on Enter (Shift+Enter for new line)
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Auto-resize textarea as user types
textInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

uploadBtn.addEventListener('click', () => imageUpload.click());
imageUpload.addEventListener('change', handleImageInput);
document.getElementById('remove-img-btn').addEventListener('click', removeImage);

// Modal and UI buttons
const aboutBtn = document.getElementById('aboutBtn');
if (aboutBtn) {
    aboutBtn.addEventListener('click', showAbout);
}

const viewDetailsBtn = document.getElementById('viewDetailsBtn');
if (viewDetailsBtn) {
    viewDetailsBtn.addEventListener('click', showAppDetails);
}

const restartBtn = document.getElementById('restartBtn');
if (restartBtn) {
    restartBtn.addEventListener('click', restartConversation);
}

const closeAppDetailsBtn = document.getElementById('closeAppDetailsBtn');
if (closeAppDetailsBtn) {
    closeAppDetailsBtn.addEventListener('click', closeAppDetails);
}

const closeAboutBtn = document.getElementById('closeAboutBtn');
if (closeAboutBtn) {
    closeAboutBtn.addEventListener('click', closeAbout);
}

// Voice Input (Speech-to-Text)
micBtn.addEventListener('click', handleVoiceInput);

// Mode select
if (modeSelect) {
    modeSelect.addEventListener('change', selectMode);
}

/**
 * Handles mode selection from the dropdown
 */
function selectMode() {
    const selected = modeSelect.value;

    // Reset cancellation flag when switching modes
    modelLoadingCancelled = false;

    if (selected === 'cpu') {
        if (!availableModes.cpu) {
            addMessage('AI mode is not available. The model failed to load.', 'bot');
            updateModeSelect(); // revert dropdown to current mode
            return;
        }
        if (!wllama) {
            // Lazy-load wllama on first switch to CPU
            modeSelect.disabled = true;
            sendBtn.disabled = true;
            textInput.disabled = true;
            micBtn.disabled = true;
            uploadBtn.disabled = true;
            const loadingMsg = addMessage('Switching to AI mode - loading model... 0%', 'bot');
            const loadingBubble = loadingMsg.bubble;

            initWllama((progress) => {
                if (loadingBubble) {
                    const percentage = Math.round(progress * 100);
                    setBubbleContent(loadingBubble, `Switching to AI mode - loading model... ${percentage}%`);
                }
            }).then(() => {
                currentMode = 'cpu';
                if (loadingBubble) {
                    setBubbleContent(loadingBubble, 'Switched to AI mode (Phi 3.5-mini)');
                }
                modeSelect.disabled = false;
                sendBtn.disabled = false;
                textInput.disabled = false;
                micBtn.disabled = false;
                uploadBtn.disabled = false;
                updateModeSelect();
            }).catch(error => {
                console.error('Failed to load wllama:', error);
                availableModes.cpu = false;
                currentMode = 'basic';
                if (loadingBubble) {
                    setBubbleContent(loadingBubble, 'Failed to load AI mode. Switched to Basic (Wikipedia) mode.');
                }
                modeSelect.disabled = false;
                sendBtn.disabled = false;
                textInput.disabled = false;
                micBtn.disabled = false;
                uploadBtn.disabled = false;
                updateModeSelect();
            });
            return;
        }

        // Model is loaded, but verify it's ready before enabling input
        if (!wllamaReady) {
            // Disable input while we wait for model to be ready
            modeSelect.disabled = true;
            sendBtn.disabled = true;
            textInput.disabled = true;
            micBtn.disabled = true;
            uploadBtn.disabled = true;
            const loadingMsg = addMessage('Switching to AI mode - preparing model...', 'bot');
            const loadingBubble = loadingMsg.bubble;

            // Poll for readiness (should be quick since model is already loaded)
            const checkReady = setInterval(() => {
                if (wllamaReady) {
                    clearInterval(checkReady);
                    currentMode = 'cpu';
                    if (loadingBubble) {
                        setBubbleContent(loadingBubble, 'Switched to AI mode (Phi 3.5-mini)');
                    }
                    modeSelect.disabled = false;
                    sendBtn.disabled = false;
                    textInput.disabled = false;
                    micBtn.disabled = false;
                    uploadBtn.disabled = false;
                    updateModeSelect();
                }
            }, 100);

            // Timeout after 5 seconds if model doesn't become ready
            setTimeout(() => {
                if (!wllamaReady) {
                    clearInterval(checkReady);
                    if (loadingBubble) {
                        setBubbleContent(loadingBubble, 'AI mode unavailable - model not ready');
                    }
                    availableModes.cpu = false;
                    currentMode = 'basic';
                    modeSelect.disabled = false;
                    sendBtn.disabled = false;
                    textInput.disabled = false;
                    micBtn.disabled = false;
                    uploadBtn.disabled = false;
                    updateModeSelect();
                }
            }, 5000);
            return;
        }

        currentMode = 'cpu';
        addMessage('Switched to AI mode (Phi 3.5-mini)', 'bot');
        updateModeSelect();
    } else {
        currentMode = 'basic';
        addMessage('Switched to Basic mode (Wikipedia)', 'bot');
        updateModeSelect();
    }
}

/**
 * Updates the mode select dropdown to reflect the current mode and availability
 */
function updateModeSelect() {
    if (!modeSelect) return;

    // Reflect current mode in the dropdown
    modeSelect.value = currentMode;

    // Dynamically update each option's text and disabled state based on availability
    const cpuOption = modeSelect.querySelector('option[value="cpu"]');
    const basicOption = modeSelect.querySelector('option[value="basic"]');

    if (cpuOption) {
        const cpuReady = availableModes.cpu;
        cpuOption.disabled = !cpuReady;
        cpuOption.textContent = cpuReady ? '🟢 AI mode (Phi 3.5-mini)' : '⚫ AI mode (unavailable)';
    }
    if (basicOption) {
        basicOption.textContent = '⚪ Basic (Wikipedia)';
    }

    // Update tooltip to reflect current mode
    const modeLabel = currentMode === 'cpu' ? 'AI mode (Phi 3.5-mini)' : 'Basic (Wikipedia)';
    modeSelect.title = `AI mode: ${modeLabel}`;
    modeSelect.setAttribute('aria-label', `Select AI mode. Currently: ${modeLabel}`);
}

/**
 * Handles voice input - tries Web Speech API first, falls back to Vosk if needed
 */
async function handleVoiceInput() {
    // Try Web Speech API first
    if (usingWebSpeech) {
        const webSpeechWorked = await tryWebSpeech();
        if (!webSpeechWorked) {
            // Web Speech failed, try Vosk fallback
            console.log('Web Speech API not available, loading Vosk fallback...');

            // Load Vosk model if not already loaded
            if (!voskLoaded && !voskLoadingFailed) {
                console.log('Attempting to load Vosk model...');
                const loaded = await loadVoskModel();
                console.log('loadVoskModel returned:', loaded);
                if (!loaded) {
                    // Vosk genuinely failed - inform user and permanently disable voice input
                    console.log('Vosk fallback failed to load, disabling voice input');
                    addMessage('I\'m sorry. Voice input is unavailable.', 'bot');
                    micBtn.disabled = true;
                    micBtn.title = 'Voice input unavailable';
                    return;
                }
            } else if (voskLoadingFailed) {
                // Previously failed - voice input unavailable
                console.log('Vosk previously failed, voice input disabled');
                addMessage('I\'m sorry. Voice input is unavailable.', 'bot');
                micBtn.disabled = true;
                micBtn.title = 'Voice input unavailable';
                return;
            }

            // Vosk loaded successfully, switch to it
            usingWebSpeech = false;
            await startVoskRecording();
        }
    } else {
        // Already using Vosk
        if (isRecording) {
            stopVoskRecording(true);
        } else {
            await startVoskRecording();
        }
    }
}

/**
 * Tries to use Web Speech API for voice input
 * @returns {Promise<boolean>} True if Web Speech worked, false if should fallback to Vosk
 */
async function tryWebSpeech() {
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
            micBtn.classList.add('listening');

            // Start no-speech timeout - resolve quietly without fallback
            noSpeechTimer = setTimeout(() => {
                if (!hasResolved) {
                    hasResolved = true;
                    console.log('No speech detected, quietly resetting...');
                    micBtn.classList.remove('listening');
                    resolve(true); // Don't fallback to Vosk
                    recognition.abort();
                }
            }, noSpeechTimeout);

            recognition.onresult = (event) => {
                // Clear no-speech timer since we got speech
                if (noSpeechTimer) {
                    clearTimeout(noSpeechTimer);
                    noSpeechTimer = null;
                }

                const transcript = event.results[0][0].transcript;
                textInput.value = transcript;
                isVoiceInput = true;
                handleSend();

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
                micBtn.classList.remove('listening');

                if (!hasResolved) {
                    hasResolved = true;
                    if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                        addMessage('Microphone access was denied.', 'bot');
                        resolve(true); // Don't fallback, user denied permission
                    } else if (event.error === 'no-speech') {
                        // No speech detected - quietly reset, no fallback
                        console.log('No speech detected, quietly resetting...');
                        resolve(true);
                    } else {
                        // Actual error (network, audio-capture, etc.) - fallback to Vosk
                        resolve(false);
                    }
                }
            };

            recognition.onend = () => {
                // Clear no-speech timer
                if (noSpeechTimer) {
                    clearTimeout(noSpeechTimer);
                    noSpeechTimer = null;
                }

                micBtn.classList.remove('listening');

                // If we haven't resolved yet, recognition ended cleanly without a result
                // (e.g. after abort from no-speech timer) - quietly reset, no fallback
                if (!hasResolved) {
                    hasResolved = true;
                    console.log('Web Speech ended without result, quietly resetting');
                    resolve(true);
                }
            };

            recognition.start();
            console.log('Web Speech recognition started');
            // Don't resolve here - wait for result or error
        } catch (error) {
            console.error('Error starting Web Speech recognition:', error);
            micBtn.classList.remove('listening');
            resolve(false);
        }
    });
}

/**
 * Reset the silence detection timer
 */
function resetSilenceTimer() {
    // Clear existing timer
    if (silenceTimer) {
        clearTimeout(silenceTimer);
    }

    // Set new timer to auto-stop after silence
    if (isRecording) {
        silenceTimer = setTimeout(() => {
            if (isRecording && hasSpeech) {
                console.log('Silence detected, auto-stopping...');
                stopVoskRecording(false);
            }
        }, silenceTimeout);
    }
}

/**
 * Start Vosk offline speech recognition
 */
async function startVoskRecording() {
    if (!voskRecognizer) {
        addMessage('Speech input is not available.', 'bot');
        return;
    }

    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                channelCount: 1,
                sampleRate: 16000
            }
        });

        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);

        // Process audio data
        processorNode.onaudioprocess = (event) => {
            try {
                if (isRecording && voskRecognizer) {
                    voskRecognizer.acceptWaveform(event.inputBuffer);
                }
            } catch (e) {
                console.error('Audio processing error:', e);
            }
        };

        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);

        isRecording = true;
        isVoiceInput = true;
        hasSpeech = false;
        lastSpeechTime = Date.now();

        // Start silence detection timer
        resetSilenceTimer();

        // Start no-speech timeout
        noSpeechTimer = setTimeout(() => {
            if (isRecording && !hasSpeech) {
                console.log('No speech detected in 5 seconds, cancelling...');
                stopVoskRecording(true);
                addMessage('No speech detected. Please try again.', 'bot');
            }
        }, noSpeechTimeout);

        // Update mic button state
        micBtn.classList.add('listening');

        console.log('Vosk recording started');
    } catch (error) {
        console.error('Error starting Vosk recording:', error);
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            addMessage('Microphone access was denied.', 'bot');
        } else {
            addMessage('Could not access microphone.', 'bot');
        }
    }
}

/**
 * Stop Vosk offline speech recognition
 * @param {boolean} cancel - If true, don't send the recognized text
 */
function stopVoskRecording(cancel = false) {
    if (!isRecording) return;

    // Stop timers
    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
    if (noSpeechTimer) {
        clearTimeout(noSpeechTimer);
        noSpeechTimer = null;
    }

    // Stop audio processing
    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    isRecording = false;
    micBtn.classList.remove('listening');

    // Get final results if not cancelled
    if (!cancel && voskRecognizer && hasSpeech) {
        handleSend();
    }

    console.log('Vosk recording stopped');
}

/**
 * Converts text to speech using Web Speech Synthesis API
 * @param {HTMLElement} element - The element containing text to speak
 */
function speakText(element) {
    // Extract text content from DOM element (safe, no HTML parsing needed)
    const cleanText = (element.textContent || '').replace(/\s+/g, ' ').trim();
    speakTextContent(cleanText);
}

/**
 * Converts plain text to speech using Web Speech Synthesis API
 * @param {string} text - The text to speak
 */
function speakTextContent(text) {
    if (!('speechSynthesis' in window)) {
        endResponse();
        return;
    }

    let cleanText = (text || '').replace(/\s+/g, ' ').trim();
    if (!cleanText) {
        endResponse();
        return;
    }

    // Replace URLs with speakable format
    cleanText = cleanText.replace(/https?:\/\/(?:www\.)?[^\s]+/gi, function (url) {
        // Remove protocol
        let cleaned = url.replace(/^https?:\/\//i, '');
        // Remove www. if present
        cleaned = cleaned.replace(/^www\./i, '');
        // Remove trailing period (if URL ends a sentence)
        cleaned = cleaned.replace(/\.$/, '');
        // Replace dots with " dot "
        cleaned = cleaned.replace(/\./g, ' dot ');
        // Replace slashes with " slash "
        cleaned = cleaned.replace(/\//g, ' slash ');
        // Replace hyphens with spaces
        cleaned = cleaned.replace(/-/g, ' ');
        return cleaned;
    });
    cleanText = cleanText.replace(/www\.[^\s]+/gi, function (url) {
        // Remove www. if present
        let cleaned = url.replace(/^www\./i, '');
        // Remove trailing period (if URL ends a sentence)
        cleaned = cleaned.replace(/\.$/, '');
        // Replace dots with " dot "
        cleaned = cleaned.replace(/\./g, ' dot ');
        // Replace slashes with " slash "
        cleaned = cleaned.replace(/\//g, ' slash ');
        // Replace hyphens with spaces
        cleaned = cleaned.replace(/-/g, ' ');
        return cleaned;
    });

    const utterance = new SpeechSynthesisUtterance(cleanText);

    // Check for debug parameter to simulate no voices
    const urlParams = new URLSearchParams(window.location.search);
    const debugNoVoices = urlParams.get('debug_no_voices') === 'true';

    // Use default voice
    const voices = speechSynthesis.getVoices();
    if (voices.length > 0 && !debugNoVoices) {
        utterance.voice = voices[0];
        usingPrerecordedVoice = false;
    } else {
        // No voices available - use pre-recorded voice
        usingPrerecordedVoice = true;
    }

    // Keep stop button enabled (already set by startResponse)
    sendBtn.classList.add('stop-mode');
    sendBtn.textContent = '⬜';
    sendBtn.title = 'Stop';

    utterance.onend = () => {
        endResponse();
    };

    utterance.onerror = () => {
        endResponse();
    };

    // Use pre-recorded audio if no voices available, otherwise use speech synthesis
    if (usingPrerecordedVoice) {
        const audio = new Audio('./audio/response.wav');
        audio.onended = () => {
            endResponse();
        };
        audio.onerror = () => {
            endResponse();
        };
        audio.play().catch(error => {
            console.error('Error playing pre-recorded audio:', error);
            endResponse();
        });
    } else {
        speechSynthesis.speak(utterance);
    }
}

/**
 * Stops any ongoing bot response (text generation or speech)
 */
async function handleStopResponse() {
    // Set flag to stop any ongoing text generation
    shouldStopResponse = true;
    isStoppingResponse = true;

    // Abort Wllama generation if active
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    // Drain wllama stream to clean up WASM state
    if (currentStream && currentMode === 'cpu') {
        safeStopWllamaStream(currentStream).catch(err => {
            console.warn('Wllama stream cleanup failed:', err);
        }).finally(() => {
            currentStream = null;
        });
    }

    // Stop speech if playing
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }

    // Remove typing indicator
    removeTyping();

    // Force clear typing animation counter
    typingAnimationsInProgress = 0;

    // Reset flags and button state
    isStoppingResponse = false;
    shouldStopResponse = false;
    isResponding = false;

    sendBtn.classList.remove('stop-mode');
    sendBtn.textContent = '▶';
    sendBtn.title = 'Send';
}

/**
 * Restarts the conversation by clearing chat history and state
 */
async function restartConversation() {
    if (confirm('Are you sure you want to clear the conversation history?')) {
        // Stop any ongoing response
        await handleStopResponse();

        // Clear conversation history
        conversationHistory = [];

        // Clear the chat UI (keep welcome message)
        chatContainer.innerHTML = '<div class="welcome-message">Let\'s chat about computing history...</div>';

        // Remove any selected image
        removeImage();

        // Reset voice input flag
        isVoiceInput = false;

        console.log('Conversation restarted');
    }
}

// ============================================================================
// MODAL UI FUNCTIONS
// ============================================================================

/**
 * Shows the About modal with focus management
 */
function showAbout() {
    const modal = document.getElementById('aboutModal');
    const closeBtn = document.getElementById('closeAboutBtn');
    modal.style.display = 'flex';
    if (closeBtn) {
        closeBtn.focus();
    }
}

/**
 * Closes the About modal and returns focus to the trigger button
 */
function closeAbout() {
    const modal = document.getElementById('aboutModal');
    const aboutBtn = document.getElementById('aboutBtn');
    modal.style.display = 'none';
    if (aboutBtn) {
        aboutBtn.focus();
    }
}

/**
 * Shows the App Details modal and updates model information
 */
function showAppDetails() {
    const modal = document.getElementById('appDetailsModal');
    const closeBtn = document.getElementById('closeAppDetailsBtn');
    const modelNameElement = document.getElementById('modalModelName');

    // Update model name based on current mode
    if (modelNameElement) {
        if (currentMode === 'cpu' && wllamaReady && wllama) {
            modelNameElement.textContent = 'Phi 3.5-mini';
        } else if (currentMode === 'basic') {
            modelNameElement.textContent = 'None (Wikipedia API)';
        } else {
            modelNameElement.textContent = 'Loading...';
        }
    }

    modal.style.display = 'flex';

    if (closeBtn) {
        closeBtn.focus();
    }
}

/**
 * Closes the App Details modal and returns focus to the trigger button
 */
function closeAppDetails() {
    const modal = document.getElementById('appDetailsModal');
    const viewDetailsBtn = document.getElementById('viewDetailsBtn');

    modal.style.display = 'none';

    // Return focus to the button that opened the modal
    if (viewDetailsBtn) {
        viewDetailsBtn.focus();
    }
}

// Close modal when clicking outside of it
document.addEventListener('click', function (event) {
    const modal = document.getElementById('appDetailsModal');
    if (event.target === modal) {
        closeAppDetails();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        const aboutModal = document.getElementById('aboutModal');
        if (aboutModal && aboutModal.style.display === 'flex') {
            closeAbout();
            return;
        }
        const modal = document.getElementById('appDetailsModal');
        if (modal.style.display === 'flex') {
            closeAppDetails();
        }
    }
});

// ============================================================================
// APPLICATION INITIALIZATION
// ============================================================================

init();
