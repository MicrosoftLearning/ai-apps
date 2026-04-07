import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from '@wllama/wllama';

const chatContainer = document.getElementById('chat-messages');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const imageUpload = document.getElementById('image-upload');
const micBtn = document.getElementById('mic-btn');
const modeToggle = document.getElementById('mode-toggle');
const modeToggleText = document.getElementById('mode-toggle-text');

// State
let model = null;
let featureExtractor = null;
let pendingFile = null;
let isVoiceInput = false; // Tracks if current message was spoken
let isResponding = false; // Tracks if bot is currently responding
let shouldStopResponse = false; // Flag to cancel ongoing response
let engine = null; // WebLLM engine for GPU mode
let wllama = null; // Wllama instance for CPU mode
let wllamaReady = false; // Track if wllama is initialized
let mobilenetReady = false; // Track if MobileNet is initialized
let conversationHistory = []; // Track conversation for context
let inappropriateWords = []; // Loaded from moderation file
let webGPUAvailable = false; // Track if WebGPU is available
let usingWllama = false; // Track which engine is active
let currentAbortController = null; // Track abort controller for wllama
let currentStream = null; // Track current streaming completion

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
const noSpeechTimeout = 5000; // Cancel after 5 seconds of no speech
let usingWebSpeech = true; // Try Web Speech API first

// Calculate speech model path dynamically based on current location
// This works both locally and on GitHub Pages
// From computing-local, go up 3 levels: computing-local -> computing-history -> ai-apps -> speech-model
const basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
const parentPath = basePath.substring(0, basePath.lastIndexOf('/'));
const rootPath = parentPath.substring(0, parentPath.lastIndexOf('/'));
const speechModelUrl = `${rootPath}/speech-model/speech-model.tar.gz`;

// Vision model paths
const MODEL_URL = './image_model/retro-classifier-model.json'; // Path to your exported model
const BASE_MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

// We need to define the classes. 
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

function buildClassInfoPrompt(classIndex) {
    const className = CLASSES[classIndex];
    const classInfo = CLASS_INFO[classIndex];

    if (!className || !classInfo) {
        return null;
    }

    return `Tell me about the ${className} computer using ONLY the following information:\nINFORMATION:\n---\n${classInfo}\n---\nProvide a concise summary in 2-3 sentences. Do not add any details that are not in the provided information`;
}

/**
 * Checks if WebGPU is available in the browser
 */
function checkWebGPUSupport() {
    if (!navigator.gpu) {
        console.log('WebGPU not supported in this browser');
        return false;
    }
    return true;
}

/**
 * Initializes the application by loading ML models and AI engines
 */
async function init() {
    // Show loading overlay
    updateLoadingStatus('mobilenet', 'loading', 'Loading...');
    updateLoadingStatus('smollm', 'loading', 'Loading...');

    // Load MobileNet and moderation in parallel
    const mobilenetPromise = loadModel();
    const moderationPromise = loadInappropriateWords();

    try {
        await Promise.all([mobilenetPromise, moderationPromise]);

        // Check for WebGPU support
        const hasWebGPU = checkWebGPUSupport();

        if (!hasWebGPU) {
            console.log('WebGPU not available, using wllama (CPU mode)');
            await initWllama();
        } else {
            // Try WebLLM first (faster with GPU)
            try {
                await initializeWebLLM();
            } catch (error) {
                console.log('WebLLM initialization failed, falling back to wllama');
                await initWllama();
            }
        }

        // Both models loaded successfully
        hideLoadingOverlay();
        updateModeToggle();
    } catch (e) {
        console.error("Initialization error:", e);
        // Show error but still try to hide overlay after a delay
        setTimeout(() => {
            hideLoadingOverlay();
            addMessage(`Error loading models: ${e.message}. Some features may be unavailable.`, "bot");
        }, 2000);
    }
}

/**
 * Initializes WebLLM for GPU mode
 */
async function initializeWebLLM() {
    try {
        updateModelName('Phi-3-mini (WebGPU)');
        updateLoadingStatus('smollm', 'loading', 'Loading AI model (WebGPU)...');

        const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';

        engine = await webllm.CreateMLCEngine(
            targetModelId,
            {
                initProgressCallback: (progress) => {
                    const percentage = Math.round(progress.progress * 100);
                    updateLoadingStatus('smollm', 'loading', `${percentage}%`);
                }
            }
        );

        updateLoadingStatus('smollm', 'ready', '100%');
        console.log('WebLLM engine initialized successfully');
        webGPUAvailable = true;
        usingWllama = false;

    } catch (error) {
        console.error('Failed to initialize WebLLM:', error);
        throw error; // Re-throw to trigger fallback
    }
}

/**
 * Load Vosk speech model for offline speech recognition (lazy-loaded fallback)
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

        const loadingMsg = addMessage('Loading offline speech model... This may take a moment.', 'bot');
        sendBtn.disabled = true;
        textInput.disabled = true;
        micBtn.disabled = true;

        try {
            voskModel = await Vosk.createModel(speechModelUrl);
        } catch (modelError) {
            // Remove the loading message since we failed
            if (loadingMsg && loadingMsg.message) {
                loadingMsg.message.remove();
            }
            throw modelError; // Re-throw to be caught by outer catch
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
        const msgBubble = loadingMsg.bubble;
        if (msgBubble) {
            setBubbleContent(msgBubble, 'Offline speech model ready! Please try your voice input again.');
        }

        sendBtn.disabled = false;
        textInput.disabled = false;
        micBtn.disabled = false;

        return true;
    } catch (error) {
        console.error('Failed to load Vosk model:', error);
        voskLoadingFailed = true;
        // Don't add another error message here - handleVoiceInput will inform the user
        sendBtn.disabled = false;
        textInput.disabled = false;
        micBtn.disabled = false;
        return false;
    }
}

function reverseWord(text) {
    return text.split('').reverse().join('');
}

function shiftWord(text, amount) {
    return text
        .split('')
        .map(char => String.fromCharCode(char.charCodeAt(0) + amount))
        .join('');
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

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

/**
 * Loads the MobileNet and custom classifier models for image classification
 * @throws {Error} If models fail to load
 */
async function loadModel() {
    // Attempt to load the model
    // Note: If you used the Browser Trainer, it might be a LayersModel.
    // If you used the Python script export, it is also a LayersModel.
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

        // Warmup prediction to initialize GPU kernels and prevent first-run issues
        console.log("Warming up model...");
        tf.tidy(() => {
            const dummyInput = tf.zeros([1, 224, 224, 3]);
            const dummyFeatures = featureExtractor.predict(dummyInput);
            model.predict(dummyFeatures);
        });
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

        const isLazyLoad = webGPUAvailable; // If WebGPU is available, this is a lazy load

        if (!isLazyLoad) {
            console.log("Initializing wllama...");
            updateModelName('SmolLM2-360M (CPU)');
            updateLoadingStatus('smollm', 'loading', '10%');
        }

        // Configure WASM paths for CDN
        const CONFIG_PATHS = {
            'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm',
            'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm',
        };

        const internalProgressCallback = ({ loaded, total }) => {
            const progress = loaded / total;
            if (!isLazyLoad) {
                const percentage = Math.round((progress * 100));
                const adjustedProgress = Math.round(20 + (percentage * 0.8)); // 20% to 100%
                updateLoadingStatus('smollm', 'loading', `${adjustedProgress}%`);
                console.log(`Loading wllama: ${percentage}%`);
            }
            if (progressCallback) {
                progressCallback(progress);
            }
        };

        // Try multithreaded (4 threads) first, fall back to single-threaded
        const useMultiThread = window.crossOriginIsolated === true;
        const preferredThreads = useMultiThread ? 4 : 1;
        console.log(`Cross-origin isolated: ${window.crossOriginIsolated}, attempting ${preferredThreads} thread(s)`);

        try {
            wllama = new Wllama(CONFIG_PATHS);
            if (!isLazyLoad) {
                updateLoadingStatus('smollm', 'loading', '20%');
            }

            await wllama.loadModelFromHF(
                'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                'smollm2-360m-instruct-q8_0.gguf',
                {
                    n_ctx: 768,
                    n_threads: preferredThreads,
                    progressCallback: internalProgressCallback
                }
            );
            console.log(`Wllama initialized successfully with ${preferredThreads} thread(s)`);
        } catch (multiErr) {
            if (preferredThreads > 1) {
                console.warn(`Multi-threaded init failed (${multiErr.message}), falling back to single thread`);
                if (!isLazyLoad) {
                    updateLoadingStatus('smollm', 'loading', '20%');
                }

                wllama = new Wllama(CONFIG_PATHS);
                await wllama.loadModelFromHF(
                    'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                    'smollm2-360m-instruct-q8_0.gguf',
                    {
                        n_ctx: 768,
                        n_threads: 1,
                        progressCallback: internalProgressCallback
                    }
                );
                console.log("Wllama initialized successfully with 1 thread (fallback)");
            } else {
                throw multiErr;
            }
        }

        wllamaReady = true;
        if (!isLazyLoad) {
            updateLoadingStatus('smollm', 'ready', '100%');
            webGPUAvailable = false;
            usingWllama = true;
        }
    } catch (error) {
        console.error('Failed to initialize wllama:', error);
        if (!webGPUAvailable) {
            updateLoadingStatus('smollm', 'error', 'Failed');
        }
        wllamaReady = false;
        throw error;
    }
}

/**
 * Updates the model name in the loading status
 * @param {string} modelName - The name of the model being loaded
 */
function updateModelName(modelName) {
    const statusTextElement = document.getElementById('smollmStatusText');
    if (statusTextElement) {
        statusTextElement.textContent = modelName;
    }
}

/**
 * Updates the loading status display for a specific model
 * @param {string} modelType - Either 'mobilenet' or 'smollm'
 * @param {string} status - One of 'loading', 'ready', or 'error'
 * @param {string} progress - Progress text to display
 */
function updateLoadingStatus(modelType, status, progress) {
    const statusId = modelType === 'mobilenet' ? 'mobilenetStatus' : 'smollmStatus';
    const progressId = modelType === 'mobilenet' ? 'mobilenetProgress' : 'smollmProgress';

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
            ALLOWED_TAGS: ['b', 'i', 'br', 'small', 'a'],
            ALLOWED_ATTR: ['href', 'target', 'style'],
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
        let currentIndex = 0;
        let displayText = startingText;

        // If starting text equals full text, nothing to type
        if (startingText === text) {
            resolve();
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
                resolve();
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
    div.innerHTML = `
        <div class="typing">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
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
    // Only end if not speaking
    if (!speechSynthesis.speaking) {
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

const STOPWORDS = new Set([
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'of', 'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for',
    'about', 'as', 'that', 'this', 'these', 'those', 'it', 'they',
    'he', 'she', 'we', 'you', 'i', 'me', 'my', 'him', 'her', 'us',
    'them', 'which', 'who', 'whom', 'whose', 'what', 'where', 'when',
    'why', 'how', 'can', 'could', 'will', 'would', 'should',
    'may', 'might', 'must', 'find', 'search', 'show', 'tell', 'look',
    'ebay', 'sale', 'buy', 'price', 'cost', 'need', 'one'
]);

const SHOPPING_TRIGGERS = ['ebay', 'for sale', 'buy', 'purchase', 'shop'];
const WEB_SEARCH_TRIGGERS = ['bing', 'search', 'find'];
const SEARCH_TRIGGER_WORDS = new Set([
    ...SHOPPING_TRIGGERS.join(' ').split(/\s+/),
    ...WEB_SEARCH_TRIGGERS.join(' ').split(/\s+/)
]);

// Initialize
// ...

/**
 * Main message handler - processes text input, images, and commands
 */
async function handleSend() {
    const text = textInput.value.trim();

    // Check if we have a file or text
    if (!text && !pendingFile) return;

    // Clear input immediately
    textInput.value = '';
    textInput.style.height = 'auto';

    // 1. Text Processing
    if (text) {
        addMessage(text, "user");

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

        // Check for Summarization Command
        const lines = text.split('\n');
        if (lines.length > 1 && lines[0].trim().toLowerCase().startsWith('summarize')) {
            const contentToSummarize = lines.slice(1).join('\n');
            showTyping();
            // Simulate "reading"
            setTimeout(() => {
                // Check if user stopped the response
                if (checkStopResponse()) {
                    removeTyping();
                    return;
                }

                const summary = summarizeText(contentToSummarize);

                // Entity Extraction
                const doc = nlp(contentToSummarize);
                const people = doc.people().out('array');
                const places = doc.places().out('array');
                // Use .match for dates
                let dates = doc.match('#Date').out('array');

                // Custom regex for years (4 digits between 1900 and current year, allowing 's' suffix)
                const currentYear = new Date().getFullYear();
                const yearRegex = /\b(19\d{2}|20\d{2})s?\b/g;
                const matches = contentToSummarize.match(yearRegex);

                if (matches) {
                    matches.forEach(item => {
                        // Remove 's' for numeric check
                        const yearNum = parseInt(item.replace('s', ''));
                        if (yearNum >= 1900 && yearNum <= currentYear) {
                            dates.push(item);
                        }
                    });
                }

                // Deduplicate and Sort
                dates = [...new Set(dates)].sort();

                let entityInfo = "";

                if (people.length > 0) {
                    entityInfo += `<br><b>People:</b> ${[...new Set(people)].join(', ')}`;
                }
                if (places.length > 0) {
                    entityInfo += `<br><b>Places:</b> ${[...new Set(places)].join(', ')}`;
                }
                if (dates.length > 0) {
                    entityInfo += `<br><b>Dates/Years:</b> ${dates.join(', ')}`;
                }

                removeTyping();
                addMessage(`<b>Summary:</b><br><br>${summary}<br><br><b>Entities Found:</b>${entityInfo}`, "bot");
            }, 1000);

            return;
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

    // For GPU mode with streaming, create message first
    if (!usingWllama && webGPUAvailable && engine) {
        const { bubble } = addMessage('', "bot", null, { deferCompletion: true });
        startResponse();

        try {
            const summary = await generateComputingInfo(text, (chunk) => {
                if (bubble && chunk) {
                    const currentText = bubble.textContent || '';
                    setBubbleContent(bubble, currentText + chunk);
                    scrollToBottom();
                }
            });

            if (checkStopResponse()) {
                removeTyping();
                return;
            }

            if (summary) {
                // Store in conversation history
                conversationHistory.push({
                    user: truncateToFirstSentence(text),
                    assistant: truncateToFirstSentence(summary)
                });
                if (conversationHistory.length > 2) {
                    conversationHistory.shift();
                }

                // Handle voice output if needed
                if (isVoiceInput) {
                    speakText(bubble);
                    isVoiceInput = false;
                } else {
                    endResponse();
                }
            } else {
                setBubbleContent(bubble, `I'm sorry. I don't know about that topic.`);
                endResponse();
            }
        } catch (e) {
            if (checkStopResponse()) return;
            setBubbleContent(bubble, "Sorry, I had trouble searching via text. " + e.message);
            endResponse();
        }
    } else {
        // CPU mode without streaming - use existing approach
        showTyping();

        try {
            const summary = await generateComputingInfo(text);
            removeTyping();

            if (checkStopResponse()) return;

            if (summary) {
                addMessage(`${summary}`, "bot");
                // Store in conversation history
                conversationHistory.push({
                    user: truncateToFirstSentence(text),
                    assistant: truncateToFirstSentence(summary)
                });
                if (conversationHistory.length > 2) {
                    conversationHistory.shift();
                }
            } else {
                addMessage(`I'm sorry. I don't know about that topic.`, "bot");
            }
        } catch (e) {
            removeTyping();
            if (checkStopResponse()) return;
            addMessage("Sorry, I had trouble searching via text. " + e.message, "bot");
        }
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
 * Extracts the first sentence from a given text string
 * @param {string} text - The text to truncate
 * @returns {string} The first sentence or first 100 characters
 */
function truncateToFirstSentence(text) {
    // Find first sentence-ending punctuation
    const match = text.match(/^[^.!?]+[.!?]/);
    if (match) {
        return match[0].trim();
    }
    // No sentence-ending punctuation found, take first 100 characters
    return text.substring(0, 100).trim();
}

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
                // For GPU mode with streaming, add message and stream into it
                if (!usingWllama && webGPUAvailable && engine) {
                    const { bubble } = addMessage(reply, "bot", null, { deferCompletion: true });
                    startResponse();

                    try {
                        const historyUserPrompt = `Tell me about the ${topMatch.className} computer`;
                        const infoPrompt = buildClassInfoPrompt(classIndex);

                        let streamedText = '';
                        const summary = await generateComputingInfo(infoPrompt || topMatch.className, (chunk) => {
                            if (bubble && chunk) {
                                streamedText += chunk;
                                setBubbleContent(bubble, reply + `<br>` + streamedText);
                                scrollToBottom();
                            }
                        });

                        if (checkStopResponse()) {
                            return;
                        }

                        if (summary) {
                            conversationHistory.push({
                                user: historyUserPrompt,
                                assistant: truncateToFirstSentence(summary)
                            });
                            if (conversationHistory.length > 2) {
                                conversationHistory.shift();
                            }
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
                } else {
                    // CPU mode - use existing non-streaming approach
                    showTyping();
                    try {
                        const historyUserPrompt = `Tell me about the ${topMatch.className} computer`;
                        const infoPrompt = buildClassInfoPrompt(classIndex);
                        const summary = await generateComputingInfo(infoPrompt || topMatch.className);
                        if (checkStopResponse()) {
                            removeTyping();
                            return;
                        }
                        if (summary) {
                            reply += `<br>${summary}`;
                            conversationHistory.push({
                                user: historyUserPrompt,
                                assistant: truncateToFirstSentence(summary)
                            });
                            if (conversationHistory.length > 2) {
                                conversationHistory.shift();
                            }
                        }
                    } catch (e) {
                        console.warn("Info generation failed", e);
                    } finally {
                        removeTyping();
                    }
                }
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

/**
 * Generates computing-related information using AI (WebLLM or Wllama)
 * @param {string} query - The query to generate information about
 * @param {Function} onChunk - Optional callback for streaming chunks (GPU mode only)
 * @returns {Promise<string|null>} Generated text or null if unavailable
 */
async function generateComputingInfo(query, onChunk = null) {
    // Use WebLLM if available and enabled, otherwise use wllama
    if (!usingWllama && webGPUAvailable && engine) {
        return await generateWithWebLLM(query, onChunk);
    } else if (wllamaReady && wllama) {
        return await generateWithWllama(query);
    } else {
        console.warn("No AI engine ready, skipping generation");
        return null;
    }
}

/**
 * Generates text using WebLLM (GPU mode) with streaming support
 * @param {string} query - The query to generate information about
 * @param {Function} onChunk - Optional callback for streaming chunks
 */
async function generateWithWebLLM(query, onChunk = null) {
    try {
        // Build messages array for WebLLM chat format
        const messages = [
            {
                role: "system",
                content: 'You are a knowledgeable assistant about computing history. You follow the rules at all times.\n\nRules:\n- You may discuss computing and technology topics only\n- Respond with one or two clear sentences, using simple language\n- Focus on key facts and historical context\n- You must not provide assistance with activities that are illegal or may cause harm'
            }
        ];

        // Include conversation history for context (last 2 exchanges)
        if (conversationHistory.length > 0) {
            conversationHistory.forEach(exchange => {
                messages.push({ role: "user", content: exchange.user });
                messages.push({ role: "assistant", content: exchange.assistant });
            });
        }

        // Add current query
        messages.push({
            role: "user",
            content: query + "\nProvide a concise and factually accurate response."
        });

        console.log('Generating info with WebLLM for:', query);

        // Generate with streaming
        let responseText = '';
        currentStream = await engine.chat.completions.create({
            messages,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 250,
            stream: true
        });

        for await (const chunk of currentStream) {
            if (shouldStopResponse) {
                currentStream = null;
                return null;
            }
            const delta = chunk.choices[0]?.delta?.content || '';
            responseText += delta;

            // Call the streaming callback if provided
            if (onChunk && delta) {
                onChunk(delta);
            }
        }

        currentStream = null;

        // Clean up the response
        responseText = responseText.trim();

        // Remove incomplete last sentence
        if (responseText && !responseText.match(/[.!?]$/)) {
            const lastCompleteMatch = responseText.match(/(.*[.!?])/);
            if (lastCompleteMatch) {
                responseText = lastCompleteMatch[1].trim();
            }
        }

        if (!responseText || responseText.length < 10) {
            return null;
        }

        return responseText;

    } catch (error) {
        console.error('Error generating info with WebLLM:', error);
        currentStream = null;
        return null;
    }
}

/**
 * Generates text using Wllama (CPU mode)
 */
async function generateWithWllama(query) {
    try {
        // Build ChatML formatted prompt
        let chatMLPrompt = '<|im_start|>system\n';
        chatMLPrompt += 'You are a knowledgeable assistant about computing history. You follow the rules at all times.\n\n';
        chatMLPrompt += 'Rules:\n';
        chatMLPrompt += '- You may discuss computing and technology topics only\n';
        chatMLPrompt += '- Respond with one or two clear sentences, using simple language\n';
        chatMLPrompt += '- Focus on key facts and historical context\n';
        chatMLPrompt += '- You must not provide assistance with activities that are illegal or may cause harm\n';
        chatMLPrompt += '<|im_end|>\n\n';

        // Include conversation history for context (last 2 exchanges)
        if (conversationHistory.length > 0) {
            conversationHistory.forEach(exchange => {
                chatMLPrompt += '<|im_start|>user\n';
                chatMLPrompt += exchange.user + '\n';
                chatMLPrompt += '<|im_end|>\n\n';
                chatMLPrompt += '<|im_start|>assistant\n';
                chatMLPrompt += exchange.assistant + '\n';
                chatMLPrompt += '<|im_end|>\n\n';
            });
        }

        // Add current user query
        chatMLPrompt += '<|im_start|>user\n';
        chatMLPrompt += query + '\n' + "Provide a concise and factually accurate response.\n";
        chatMLPrompt += '<|im_end|>\n\n';
        chatMLPrompt += '<|im_start|>assistant\n';

        console.log('Generating info with Wllama for:', query);

        // Create AbortController for cancellation
        currentAbortController = new AbortController();

        // Generate response
        let responseText = '';
        const completion = await wllama.createCompletion(chatMLPrompt, {
            nPredict: 250,
            sampling: {
                temp: 0.3,
                top_k: 40,
                top_p: 0.9,
                penalty_repeat: 1.1
            },
            stopTokens: ['<|im_end|>', '<|im_start|>'],
            stream: true,
            abortSignal: currentAbortController.signal
        });

        for await (const chunk of completion) {
            if (shouldStopResponse) {
                currentAbortController = null;
                return null;
            }
            if (chunk.currentText) {
                responseText = chunk.currentText;
            }
        }

        currentAbortController = null;

        // Clear KV cache after generation
        try {
            await wllama.kvClear();
        } catch (error) {
            // Silently ignore
        }

        // Clean up the response
        responseText = responseText.trim();

        // Remove incomplete last sentence
        if (responseText && !responseText.match(/[.!?]$/)) {
            const lastCompleteMatch = responseText.match(/(.*[.!?])/);
            if (lastCompleteMatch) {
                responseText = lastCompleteMatch[1].trim();
            }
        }

        if (!responseText || responseText.length < 10) {
            return null;
        }

        return responseText;

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Generation aborted by user');
            return null;
        }
        console.error('Error generating info with Wllama:', error);
        currentAbortController = null;
        try {
            await wllama.kvClear();
        } catch (e) {
            // Silently ignore
        }
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

// Event Listeners
sendBtn.addEventListener('click', () => {
    if (sendBtn.classList.contains('stop-mode')) {
        handleStopResponse();
    } else {
        handleSend();
    }
});

textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Auto-resize textarea
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

// Mode toggle
if (modeToggle) {
    modeToggle.addEventListener('change', toggleMode);
}

/**
 * Toggles between GPU and CPU mode
 */
function toggleMode() {
    if (!webGPUAvailable) {
        // Show info that WebGPU is not available
        addMessage('WebGPU is not available in your browser. The app is running in CPU mode using SmolLM2.', 'bot');
        return;
    }

    usingWllama = !usingWllama;
    updateModeToggle();

    // If switching to wllama and it's not loaded yet, show loading message
    if (usingWllama && !wllama) {
        sendBtn.disabled = true;
        textInput.disabled = true;
        micBtn.disabled = true;
        const loadingMsg = addMessage('Switching to CPU mode - loading model... 0%', 'bot');
        const loadingBubble = loadingMsg.bubble;

        // Lazy load wllama
        initWllama((progress) => {
            if (loadingBubble) {
                const percentage = Math.round(progress * 100);
                setBubbleContent(loadingBubble, `Switching to CPU mode - loading model... ${percentage}%`);
            }
        }).then(() => {
            const mode = usingWllama ? 'CPU' : 'GPU';
            if (loadingBubble) {
                setBubbleContent(loadingBubble, `Switched to ${mode} mode`);
            }
            sendBtn.disabled = false;
            textInput.disabled = false;
            micBtn.disabled = false;
        }).catch(error => {
            console.error('Failed to load wllama:', error);
            if (loadingBubble) {
                setBubbleContent(loadingBubble, 'Failed to load CPU mode. Reverting to GPU mode.');
            }
            usingWllama = false;
            updateModeToggle();
            sendBtn.disabled = false;
            textInput.disabled = false;
            micBtn.disabled = false;
        });
    } else {
        const mode = usingWllama ? 'CPU' : 'GPU';
        console.log(`Switched to ${mode} mode`);
        addMessage(`Switched to ${mode} mode`, 'bot');
    }
}

/**
 * Updates the mode toggle UI
 */
function updateModeToggle() {
    if (!modeToggle || !modeToggleText) return;

    const isGpuMode = !usingWllama;

    // Update checkbox state
    modeToggle.checked = usingWllama;
    modeToggle.setAttribute('aria-checked', usingWllama ? 'true' : 'false');

    // Update text label
    modeToggleText.textContent = isGpuMode ? 'GPU' : 'CPU';

    // Update title and aria-label
    const modeTitle = isGpuMode ?
        'Currently using GPU (Phi-3). Click to switch to CPU mode (SmolLM2).' :
        'Currently using CPU (SmolLM2). Click to switch to GPU mode (Phi-3).';
    const ariaLabel = isGpuMode ?
        'Toggle AI mode. Currently in GPU mode. Click to switch to CPU mode.' :
        'Toggle AI mode. Currently in CPU mode. Click to switch to GPU mode.';

    const toggleContainer = modeToggle.parentElement;
    if (toggleContainer) {
        toggleContainer.title = modeTitle;
    }
    modeToggle.setAttribute('aria-label', ariaLabel);

    // Disable toggle if WebGPU not available
    if (!webGPUAvailable) {
        modeToggle.disabled = true;
        modeToggle.checked = true; // CPU mode
        modeToggle.setAttribute('aria-checked', 'true');
        modeToggleText.textContent = 'CPU';
        if (toggleContainer) {
            toggleContainer.title = 'WebGPU not available - CPU mode only';
        }
        modeToggle.setAttribute('aria-label', 'AI mode set to CPU only. WebGPU not available.');
    }
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
                const loaded = await loadVoskModel();
                if (!loaded) {
                    // Vosk also failed - inform user and stay on Web Speech
                    console.log('Vosk fallback unavailable, voice input disabled');
                    addMessage('Voice input is unavailable. The online speech service is not accessible, and the offline speech model could not be loaded.', 'bot');
                    return;
                }
            } else if (voskLoadingFailed) {
                // Previously failed to load Vosk - inform user
                addMessage('Voice input is unavailable. The online speech service is not accessible, and the offline speech model is not available.', 'bot');
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

            // Start no-speech timeout
            noSpeechTimer = setTimeout(() => {
                if (!hasResolved) {
                    console.log('No speech detected in 5 seconds, cancelling...');
                    recognition.stop();
                    if (!hasResolved) {
                        hasResolved = true;
                        micBtn.classList.remove('listening');
                        addMessage('No speech detected. Please try again.', 'bot');
                        resolve(true); // Don't fallback, just inform user
                    }
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
                    // If it's a permission error, don't fallback
                    if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                        addMessage('Microphone access was denied.', 'bot');
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

                micBtn.classList.remove('listening');
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
    if (!('speechSynthesis' in window)) {
        endResponse();
        return;
    }

    // Extract text content from DOM element (safe, no HTML parsing needed)
    const cleanText = (element.textContent || '').replace(/\s+/g, ' ').trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);

    // Use default voice
    const voices = speechSynthesis.getVoices();
    if (voices.length > 0) {
        utterance.voice = voices[0];
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

    speechSynthesis.speak(utterance);
}

/**
 * Stops any ongoing bot response (text generation or speech)
 */
function handleStopResponse() {
    // Set flag to stop any ongoing text generation
    shouldStopResponse = true;

    // Abort WebLLM streaming if active
    if (currentStream) {
        currentStream = null;
    }

    // Abort Wllama generation if active
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    // Stop speech if playing
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }

    // Remove typing indicator
    removeTyping();

    // Reset button state
    endResponse();
}

/**
 * Restarts the conversation by clearing chat history and state
 */
async function restartConversation() {
    if (confirm('Are you sure you want to clear the conversation history?')) {
        // Stop any ongoing response
        handleStopResponse();

        // Clear the chat UI
        chatContainer.innerHTML = '<div class="welcome-message">Let\'s chat about computing history...</div>';

        // Remove any selected image
        removeImage();

        // Reset voice input flag
        isVoiceInput = false;

        // Clear conversation history (browser-side cache)
        conversationHistory = [];

        // Clear model's KV cache for Wllama
        if (wllama && wllamaReady) {
            try {
                await wllama.kvClear();
                console.log('Wllama KV cache cleared');
            } catch (error) {
                // Silently ignore - kvClear can throw harmless warnings
                console.debug('KV cache clear warning (harmless):', error);
            }
        }

        // Note: WebLLM doesn't maintain persistent state across messages in the same way,
        // so we just clear the conversation history array
    }
}

/**
 * Shows the app details modal with focus management
 */
function showAbout() {
    const modal = document.getElementById('aboutModal');
    const closeBtn = document.getElementById('closeAboutBtn');
    modal.style.display = 'flex';
    if (closeBtn) {
        closeBtn.focus();
    }
}

function closeAbout() {
    const modal = document.getElementById('aboutModal');
    const aboutBtn = document.getElementById('aboutBtn');
    modal.style.display = 'none';
    if (aboutBtn) {
        aboutBtn.focus();
    }
}

function showAppDetails() {
    const modal = document.getElementById('appDetailsModal');
    const closeBtn = document.getElementById('closeAppDetailsBtn');
    const modelNameElement = document.getElementById('modalModelName');

    // Update model name based on current mode
    if (modelNameElement) {
        if (!usingWllama && webGPUAvailable && engine) {
            modelNameElement.textContent = 'Phi-3-mini-4k-instruct (WebGPU - running locally)';
        } else if (wllamaReady && wllama) {
            modelNameElement.textContent = 'SmolLM2-360M-Instruct (CPU - running locally)';
        } else {
            modelNameElement.textContent = 'Loading...';
        }
    }

    modal.style.display = 'flex';

    // Focus the close button for keyboard accessibility
    if (closeBtn) {
        closeBtn.focus();
    }
}

/**
 * Closes the app details modal
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

// Start
init();
