import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/index.js';

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class ChatPlayground {
    constructor() {
        // Core state
        this.wllama = null; // wllama engine for AI mode
        this.usingWllama = false; // Track if wllama is active
        this.usingWikipedia = false; // Track if no-model Wikipedia mode is active
        this.currentMode = 'none'; // 'phi-cpu' or 'none'
        this.wllamaLoaded = false; // Track if wllama is initialized
        this.wllamaFailed = false; // Track if Phi 3.5-mini failed to load
        this.wllamaUsedGPU = false; // True if current wllama instance was loaded with GPU layers
        this.gpuFailed = false; // True after a GPU session crash; suppresses future GPU attempts
        this.isModelLoaded = false;
        this.conversationHistory = [];
        this.isGenerating = false;
        this.stopRequested = false;
        this.currentStream = null; // Track current streaming completion
        this.currentAbortController = null; // Track abort controller for wllama
        this.typingState = null;
        this.currentSystemMessage = "You are an AI assistant that helps people find information.";
        this.currentModelId = null;
        this.voiceMode = false; // Track if voice mode is enabled
        this.isSpeaking = false; // Track if TTS is speaking
        this.isListening = false; // Track if speech recognition is active
        this.avatarEnabled = false; // Track if avatar is enabled
        this.selectedAvatar = null; // Track selected avatar filename
        this.availableAvatars = ['Boris.svg', 'Doris.svg']; // Available avatars
        this.voiceInteractionCancelled = false; // Track if user explicitly cancelled voice interaction
        this.isStartingRecognition = false; // Track if we're intentionally starting recognition
        this.modelLoadingCancelled = false; // Track if user cancelled initial model loading
        this.modelLoadingAbortController = null; // Track abort controller for model loading

        // Configuration objects
        this.config = {
            modelParameters: {
                temperature: 0.5,
                top_p: 0.7,
                max_tokens: 400,
                repetition_penalty: 1.1
            },
            fileUpload: {
                content: null,
                fileName: null,
                maxSize: 3 * 1024, // 3KB
                allowedTypes: ['.txt']
            },
            visionSettings: {
                imageAnalysis: false,
                maxImageSize: 5 * 1024 * 1024, // 5MB
                allowedImageTypes: ['image/jpeg', 'image/jpg', 'image/png']
            }
        };

        // Initialize vision settings directly (for backward compatibility)
        this.visionSettings = {
            imageAnalysis: false,
            maxImageSize: 5 * 1024 * 1024, // 5MB
            allowedImageTypes: ['image/jpeg', 'image/jpg', 'image/png']
        };

        // Initialize speech settings
        this.speechSettings = {
            voice: '',
            textToSpeech: true
        };

        // Speech recognition state
        this.recognition = null;
        this.voicesAvailable = false;
        this.voicesLoaded = false;
        this.voiceHealthCacheKey = 'chat-playground-voice-health-v1';
        this.voiceHealthCacheTtlMs = 7 * 24 * 60 * 60 * 1000;
        this.voiceHealthCache = {};
        this.voiceHealthCheckPromise = null;
        this.voiceSelectionProbeNonce = 0;
        this.voiceSelectionProbePromise = null;
        this.showCaptions = false; // Track whether to show conversation text
        this.prohibitedTerms = []; // Content moderation terms loaded from file

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
        this.silenceTimer = null;
        this.noSpeechTimer = null;
        this.lastSpeechTime = null;
        this.hasSpeech = false;
        this.voskTranscript = ''; // Buffer for Vosk transcript
        this.voskPartialTranscript = ''; // Buffer for partial results
        this.silenceTimeout = 2000; // Auto-stop after 2 seconds of silence
        this.noSpeechTimeoutDuration = 5000; // Cancel after 5 seconds of no speech
        this.usingWebSpeech = true; // Try Web Speech API first

        // Calculate speech model path relative to the base path
        const basePath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
        const parentPath = basePath.substring(0, basePath.lastIndexOf('/'));
        this.speechModelUrl = `${parentPath}/speech-model/speech-model.tar.gz`;

        this._voicesChangedListenerAdded = false; // guard against voiceschanged listener stacking

        this.loadVoiceHealthCache();

        // Initialize DOM element registry
        this.elements = {};
        this.eventListeners = [];

        // Initialize app
        this.initialize();
    }

    // Constants for error messages and UI text
    static MESSAGES = {
        ERRORS: {
            FILE_TYPE: 'Please select a valid file type',
            FILE_SIZE: 'File too large. Please select a smaller file',
            IMAGE_LOAD: 'Error loading image. Please try a different file',
            IMAGE_PROCESS: 'Error processing image. Please try again',
            MODEL_DOWNLOAD: 'Model is downloading. Please wait...',
            MODEL_NOT_READY: 'Model not ready. Please try enabling again',
            SPEECH_NOT_AVAILABLE: 'Speech recognition not available',
            SPEECH_ERROR: 'Speech recognition error. Please try again',
            VOICE_INPUT_FAILED: 'Could not start voice input. Please try again'
        },
        SUCCESS: {
            FILE_UPLOADED: 'File uploaded successfully',
            FILE_REMOVED: 'File removed',
            IMAGE_READY: 'Image ready to send with next message',
            SYSTEM_MESSAGE_UPDATED: 'System message updated',
            PARAMETERS_RESET: 'Parameters reset to defaults',
            SETTINGS_UPDATED: 'Chat settings updated'
        },
        MODERATION: {
            BLOCKED: "I'm sorry. I can't help with that because it triggered a content-safety filter."
        }
    };

    // Stopwords for text summarization
    static STOPWORDS = new Set([
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
        'search', 'look', 'information', 'info'
    ]);

    // Centralized initialization
    async initialize() {
        await this.loadProhibitedTerms();
        this.initializeElements();
        this.attachEventListeners();
        this.initializeParameterControls();
        this.initializeFileUpload();
        this.setupImageAnalysisToggle();
        this.populateVoices();
        this.initializeSpeechRecognition();
        this.initializeAvatars();
        this.initializeModel();
    }

    reverseWord(text) {
        return text.split('').reverse().join('');
    }

    shiftWord(text, amount) {
        return text
            .split('')
            .map(char => String.fromCharCode(char.charCodeAt(0) + amount))
            .join('');
    }

    escapeRegex(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    async loadProhibitedTerms() {
        try {
            const response = await fetch('moderation/mod.txt');
            if (!response.ok) throw new Error('Failed to load prohibited terms');

            const encodedTermsText = await response.text();
            this.prohibitedTerms = encodedTermsText
                .split(/\r?\n/)
                .map(term => term.trim())
                .filter(term => term.length > 0)
                .map(term => this.shiftWord(this.reverseWord(term.toLowerCase()), 1));

            console.log('Loaded prohibited terms:', this.prohibitedTerms.length);
        } catch (error) {
            console.error('Error loading prohibited terms:', error);
            throw error;
        }
    }

    initializeElements() {
        // Define all element selectors in one place for easier maintenance
        const elementSelectors = {
            // Progress elements
            progressContainer: 'progress-container',
            progressFill: 'progress-fill',
            progressText: 'progress-text',

            // Model and system elements
            modelSelect: 'model-select',
            systemMessage: 'system-message',

            // Chat elements
            chatMessages: 'chat-messages',
            userInput: 'user-input',
            sendBtn: 'send-btn',
            stopBtn: 'stop-btn',
            attachBtn: 'attach-btn',

            // File upload elements
            fileInput: 'file-input',
            fileInfo: 'file-info',
            fileName: 'file-name',
            fileSize: 'file-size',
            addDataBtn: 'add-data-btn',

            // Vision elements
            imageAnalysisToggle: 'image-analysis-toggle',
            visionProgressContainer: 'vision-progress-container',
            visionProgressFill: 'vision-progress-fill',
            visionProgressText: 'vision-progress-text',

            // Input image elements
            inputThumbnailContainer: 'input-thumbnail-container',
            inputThumbnail: 'input-thumbnail',
            removeThumbnailBtn: 'remove-thumbnail-btn'
        };

        // Populate elements object with actual DOM references
        Object.entries(elementSelectors).forEach(([key, id]) => {
            this.elements[key] = document.getElementById(id);
        });

        // Set legacy references for backward compatibility
        this.progressContainer = this.elements.progressContainer;
        this.progressFill = this.elements.progressFill;
        this.progressText = this.elements.progressText;

        this.modelSelect = this.elements.modelSelect;
        this.systemMessage = this.elements.systemMessage;
        this.chatMessages = this.elements.chatMessages;
        this.userInput = this.elements.userInput;
        this.sendBtn = this.elements.sendBtn;
        this.stopBtn = this.elements.stopBtn;
        this.attachBtn = this.elements.attachBtn;

        // Initialize vision state
        this.mobileNetModel = null;
        this.isModelDownloading = false;
        this.pendingImage = null;
    }

    // Getter for backward compatibility
    get modelParameters() {
        return this.config.modelParameters;
    }

    set modelParameters(value) {
        this.config.modelParameters = value;
    }

    // Get model-specific default parameters
    getModelDefaults() {
        return {
            temperature: 0.5,
            top_p: 0.7,
            max_tokens: 768,
            repetition_penalty: 1.1
        };
    }

    // Update UI sliders to reflect current parameter values
    updateParameterUI() {
        const params = this.config.modelParameters;
        const updates = [
            { slider: 'temperature-slider', value: 'temperature-value', param: 'temperature' },
            { slider: 'top-p-slider', value: 'top-p-value', param: 'top_p' },
            { slider: 'max-tokens-slider', value: 'max-tokens-value', param: 'max_tokens' },
            { slider: 'repetition-penalty-slider', value: 'repetition-penalty-value', param: 'repetition_penalty' }
        ];

        updates.forEach(({ slider, value, param }) => {
            const sliderEl = document.getElementById(slider);
            const valueEl = document.getElementById(value);
            if (sliderEl && valueEl) {
                sliderEl.value = params[param];
                valueEl.textContent = params[param];
                sliderEl.setAttribute('aria-valuetext', params[param].toString());
            }

            // Also update modal sliders if they exist
            const modalSlider = 'modal-' + slider;
            const modalValue = 'modal-' + value;
            const modalSliderEl = document.getElementById(modalSlider);
            const modalValueEl = document.getElementById(modalValue);
            if (modalSliderEl && modalValueEl) {
                modalSliderEl.value = params[param];
                modalValueEl.textContent = params[param];
                modalSliderEl.setAttribute('aria-valuetext', params[param].toString());
            }
        });
    }

    // Utility functions to reduce code duplication
    getElement(id) {
        return this.elements[id] || document.getElementById(id);
    }

    setElementProperty(elementId, property, value) {
        const element = this.getElement(elementId);
        if (element) {
            element[property] = value;
        }
        return element;
    }

    setElementText(elementId, text) {
        return this.setElementProperty(elementId, 'textContent', text);
    }

    setElementStyle(elementId, property, value) {
        const element = this.getElement(elementId);
        if (element) {
            element.style[property] = value;
        }
        return element;
    }

    showElement(elementId) {
        return this.setElementStyle(elementId, 'display', 'block');
    }

    hideElement(elementId) {
        return this.setElementStyle(elementId, 'display', 'none');
    }

    toggleElement(elementId, show = null) {
        const element = this.getElement(elementId);
        if (element) {
            const isVisible = element.style.display !== 'none';
            const shouldShow = show !== null ? show : !isVisible;
            element.style.display = shouldShow ? 'block' : 'none';
        }
        return element;
    }

    addEventListenerTracked(element, event, handler, options = false) {
        if (typeof element === 'string') {
            element = this.getElement(element);
        }
        if (element) {
            element.addEventListener(event, handler, options);
            this.eventListeners.push({ element, event, handler, options });
        }
    }

    // Cleanup method to remove all tracked event listeners
    cleanup() {
        this.eventListeners.forEach(({ element, event, handler, options }) => {
            if (element && element.removeEventListener) {
                element.removeEventListener(event, handler, options);
            }
        });
        this.eventListeners = [];
    }

    updateProgress(containerId, fillId, textId, percentage, text) {
        this.showElement(containerId);
        this.setElementStyle(fillId, 'width', `${percentage}%`);
        this.setElementText(textId, text);
    }

    validateFileType(file, allowedTypes, maxSize = null) {
        if (!allowedTypes.some(type =>
            file.name.toLowerCase().endsWith(type.toLowerCase()) ||
            file.type === type
        )) {
            return { valid: false, error: `Please select a ${allowedTypes.join(', ')} file.` };
        }

        if (maxSize && file.size > maxSize) {
            const sizeMB = (maxSize / (1024 * 1024)).toFixed(1);
            return { valid: false, error: `File too large. Maximum size: ${sizeMB}MB` };
        }

        return { valid: true };
    }

    initializeParameterControls() {
        // Centralized parameter configuration - using modal slider IDs (only parameter
        // sliders present in the HTML)
        this.parameterConfig = [
            {
                id: 'modal-temperature-slider',
                valueId: 'modal-temperature-value',
                param: 'temperature',
                type: 'float',
                displayName: 'Temperature'
            },
            {
                id: 'modal-top-p-slider',
                valueId: 'modal-top-p-value',
                param: 'top_p',
                type: 'float',
                displayName: 'Top P'
            },
            {
                id: 'modal-max-tokens-slider',
                valueId: 'modal-max-tokens-value',
                param: 'max_tokens',
                type: 'int',
                displayName: 'Max Tokens'
            },
            {
                id: 'modal-repetition-penalty-slider',
                valueId: 'modal-repetition-penalty-value',
                param: 'repetition_penalty',
                type: 'float',
                displayName: 'Repetition Penalty'
            }
        ];

        this.parameterConfig.forEach(config => {
            this.initializeSlider(config);
        });
    }

    initializeSlider({ id, valueId, param, type, displayName }) {
        const slider = this.getElement(id);
        const valueDisplay = this.getElement(valueId);

        if (!slider || !valueDisplay) return;

        const initialValue = this.config.modelParameters[param];

        // Set initial values
        slider.value = initialValue;
        valueDisplay.textContent = initialValue;
        slider.setAttribute('aria-valuetext', initialValue.toString());

        // Add event listener
        this.addEventListenerTracked(slider, 'input', (e) => {
            const value = type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value);
            this.config.modelParameters[param] = value;
            valueDisplay.textContent = value;
            slider.setAttribute('aria-valuetext', value.toString());
            this.showToast(`${displayName}: ${value}`);

            // Also update modal slider if it exists
            updateModalSliderFromSource(id, valueId, value);
        });
    }

    formatParameterName(param) {
        const names = {
            'temperature': 'Temperature',
            'top_p': 'Top P',
            'max_tokens': 'Max Tokens',
            'repetition_penalty': 'Repetition Penalty'
        };
        return names[param] || param;
    }

    initializeFileUpload() {
        this.addEventListenerTracked('fileInput', 'change', (e) => this.handleFileUpload(e));
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Use centralized file validation
        const validation = this.validateFileType(
            file,
            this.config.fileUpload.allowedTypes,
            this.config.fileUpload.maxSize
        );

        if (!validation.valid) {
            alert(validation.error);
            event.target.value = '';
            return;
        }

        // Read file content
        const reader = new FileReader();
        reader.onload = (e) => {
            this.config.fileUpload.content = e.target.result;
            this.config.fileUpload.fileName = file.name;
            this.displayFileInfo(file);
            this.showToast(`${ChatPlayground.MESSAGES.SUCCESS.FILE_UPLOADED}: ${file.name}`);

            // Restart conversation to apply the new file data to system message
            this.restartConversation('file-upload');
        };

        reader.onerror = () => {
            alert('Error reading file');
            event.target.value = '';
        };

        reader.readAsText(file);
    }

    displayFileInfo(file) {
        this.setElementText('fileName', '🗒 ' + file.name);
        this.setElementText('fileSize', `${(file.size / 1024).toFixed(1)}KB`);
        this.setElementStyle('fileInfo', 'display', 'flex');
        this.hideElement('addDataBtn');
        const fileSearchOption = document.getElementById('file-search-option');
        if (fileSearchOption) fileSearchOption.style.display = 'none';
    }

    isWebSearchActive() {
        const el = document.getElementById('web-search-info');
        return !!(el && el.style.display !== 'none');
    }

    extractWebSearchKeywords(text, extraTerms = null) {
        const stopwords = new Set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'with',
            'or', 'but', 'if', 'than', 'then', 'so', 'yet',
            'after', 'before', 'between', 'during', 'into', 'through', 'over',
            'under', 'until', 'up', 'down', 'out', 'off', 'above', 'below',
            'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'this', 'these', 'those', 'some', 'any', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'other', 'another', 'much', 'many',
            'am', 'was', 'were', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'may', 'might', 'must', 'shall', 'will',
            'get', 'make', 'know', 'see', 'take', 'come', 'go', 'want',
            'use', 'find', 'need', 'try', 'ask', 'work', 'help', 'like',
            'what', 'when', 'where', 'who', 'how', 'why', 'which',
            'also', 'just', 'now', 'here', 'there', 'very', 'too',
            'really', 'still', 'always', 'never', 'often', 'sometimes',
            'search', 'look', 'information', 'info', 'please', 'tell',
            'about', 'regarding', 'related', 'anything'
        ]);
        const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
        const keywords = words.filter(w => w.length > 0 && !stopwords.has(w)).join(' ');
        const combined = [keywords, extraTerms].filter(Boolean).join(' ').trim();
        return combined || null;
    }

    removeFile() {
        // Clear file data
        this.config.fileUpload.content = null;
        this.config.fileUpload.fileName = null;

        // Update UI using utility functions
        this.hideElement('fileInfo');
        this.setElementProperty('fileInput', 'value', '');
        this.showElement('addDataBtn');
        const fileSearchOption = document.getElementById('file-search-option');
        if (fileSearchOption) fileSearchOption.style.display = '';

        this.showToast(ChatPlayground.MESSAGES.SUCCESS.FILE_REMOVED);
        this.restartConversation('file-remove');
    }

    getEffectiveSystemMessage() {
        // Return system message without file upload content
        // File content will be appended to user messages instead
        return this.currentSystemMessage;
    }

    updateConversationHistoryWithCurrentSystemMessage() {
        // This method ensures that when the system message is changed in the UI,
        // all turns in the conversation history get the updated system message
        // when building the messages array for the API request.
        // Note: We don't modify the stored conversationHistory itself,
        // but rather rebuild the messages array with the current system message
        // at request time to avoid storing large amounts of duplicate system messages.
    }

    setupImageAnalysisToggle() {
        // Handle image analysis toggle
        const imageAnalysisToggle = document.getElementById('image-analysis-toggle');
        if (imageAnalysisToggle) {
            imageAnalysisToggle.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                this.visionSettings.imageAnalysis = isEnabled;
                this.updateAttachButtonState();

                // Download model when enabled for the first time
                if (isEnabled && !this.mobileNetModel && !this.isModelDownloading) {
                    this.updateSaveButtonState(); // Disable save button before download
                    await this.downloadMobileNetModel();
                    this.updateSaveButtonState(); // Re-enable save button after download
                }

                console.log('Image analysis:', isEnabled ? 'enabled' : 'disabled');
            });
            // Initialize state
            this.visionSettings.imageAnalysis = imageAnalysisToggle.checked;
            this.updateAttachButtonState();
        }
    }

    updateAttachButtonState() {
        if (this.attachBtn) {
            // Only enable attach button if image analysis is enabled AND model is downloaded (not downloading)
            this.attachBtn.disabled = !this.visionSettings.imageAnalysis || this.isModelDownloading || !this.mobileNetModel;
        }
    }

    updateSaveButtonState() {
        const saveBtn = document.getElementById('save-capabilities-btn');
        if (saveBtn) {
            const shouldDisable = this.isModelDownloading;
            saveBtn.disabled = shouldDisable;

            if (shouldDisable) {
                saveBtn.textContent = 'Downloading Model...';
                saveBtn.style.opacity = '0.6';
                saveBtn.style.cursor = 'not-allowed';
            } else {
                saveBtn.textContent = 'Save';
                saveBtn.style.opacity = '1';
                saveBtn.style.cursor = 'pointer';
            }
        }
    }

    openConfigFlyout() {
        const flyoutOverlay = document.getElementById('config-flyout-overlay');
        const voiceSelect = document.getElementById('config-voice-select');

        if (flyoutOverlay) {
            flyoutOverlay.style.display = 'block';
        }

        // Restore voice selection if we have one
        if (voiceSelect && this.speechSettings.voice) {
            voiceSelect.value = this.speechSettings.voice;
        }
    }

    closeConfigFlyout() {
        const flyoutOverlay = document.getElementById('config-flyout-overlay');
        if (flyoutOverlay) {
            flyoutOverlay.style.display = 'none';
        }
    }

    async toggleVoiceMode(isEnabled) {
        this.voiceMode = isEnabled;

        const chatPanel = document.querySelector('.chat-panel');
        const voiceControls = document.getElementById('voice-controls');
        const textInputWrapper = document.getElementById('text-input-wrapper');
        const textWelcome = document.getElementById('text-welcome');
        const voiceWelcome = document.getElementById('voice-welcome');
        const chatMessages = document.getElementById('chat-messages');
        const voiceSelect = document.getElementById('config-voice-select');
        const previewBtn = document.getElementById('preview-voice-btn');

        if (isEnabled) {

            if (textWelcome) {
                textWelcome.style.display = 'none';
            }
            if (voiceWelcome) {
                voiceWelcome.style.display = 'flex';
            }

            // Enable voice controls
            if (voiceSelect && this.voicesAvailable) {
                voiceSelect.disabled = false;
            }
            if (previewBtn && this.voicesAvailable) {
                previewBtn.disabled = false;
            }

            console.log('Voice mode enabled');
        } else {
            // Switch back to text mode UI
            if (chatPanel) {
                chatPanel.classList.remove('voice-mode');
            }
            this.closeVoiceInputErrorModal();
            if (voiceControls) {
                voiceControls.style.display = 'none';
            }
            if (textInputWrapper) {
                textInputWrapper.style.display = 'block';
            }
            if (textWelcome) {
                textWelcome.style.display = 'flex';
            }
            if (voiceWelcome) {
                voiceWelcome.style.display = 'none';
            }

            // Disable voice controls
            if (voiceSelect) {
                voiceSelect.disabled = true;
            }
            if (previewBtn) {
                previewBtn.disabled = true;
            }

            // Reset to Web Speech API for next time
            this.usingWebSpeech = true;

            // Clear any messages that might have been added
            if (chatMessages) {
                const messages = chatMessages.querySelectorAll('.message');
                messages.forEach(msg => msg.remove());
            }

            // Stop any ongoing speech
            if (speechSynthesis) {
                speechSynthesis.cancel();
            }

            console.log('Voice mode disabled');
        }
    }

    async downloadMobileNetModel() {
        if (this.mobileNetModel || this.isModelDownloading) {
            return;
        }

        this.isModelDownloading = true;
        this.updateSaveButtonState(); // Disable save button

        const progressContainer = document.getElementById('vision-progress-container');
        const progressFill = document.getElementById('vision-progress-fill');
        const progressText = document.getElementById('vision-progress-text');

        try {
            // Show progress container
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }

            // Update progress text
            if (progressText) {
                progressText.textContent = 'Initializing TensorFlow.js...';
            }

            // Wait for TensorFlow.js to be ready
            await tf.ready();

            // Update progress
            if (progressFill) progressFill.style.width = '30%';
            if (progressText) progressText.textContent = 'Loading MobileNet model...';

            // Load MobileNet model using the same approach as image-analyzer
            const mobileNetModel = await mobilenet.load({
                version: 2,
                alpha: 1.0,
                modelUrl: undefined,
                inputRange: [0, 1]
            });

            // Update progress
            if (progressFill) progressFill.style.width = '90%';
            if (progressText) progressText.textContent = 'Model ready!';

            this.mobileNetModel = mobileNetModel;

            // Complete progress
            if (progressFill) progressFill.style.width = '100%';
            if (progressText) progressText.textContent = 'Model ready!';

            // Hide progress after a short delay
            setTimeout(() => {
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            }, 2000);

            console.log('MobileNet model downloaded and ready');

        } catch (error) {
            console.error('Error downloading MobileNet model:', error);
            let errorMessage = 'Error downloading model: ';
            if (error.message) {
                errorMessage += error.message;
            } else {
                errorMessage += 'Unknown error occurred.';
            }

            if (progressText) {
                progressText.textContent = errorMessage;
            }

            // Hide progress after error
            setTimeout(() => {
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            }, 5000);
        } finally {
            this.isModelDownloading = false;
            this.updateSaveButtonState(); // Re-enable save button
            this.updateAttachButtonState(); // Update attach button state (enable if model loaded, disable if error)
        }
    }

    handleImageUpload() {
        // Check if image analysis is enabled (required for both modes)
        if (!this.visionSettings.imageAnalysis) {
            this.showToast('Please enable image analysis first');
            return;
        }

        // Check if model is ready
        if (!this.mobileNetModel) {
            if (this.isModelDownloading) {
                this.showToast('Model is downloading. Please wait...');
            } else {
                this.showToast('Model not ready. Please try enabling image analysis again.');
            }
            return;
        }

        // Create a file input element
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.jpg,.jpeg,.png';
        fileInput.style.display = 'none';

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.processImageFile(file);
            }
        });

        // Trigger file selection
        document.body.appendChild(fileInput);
        fileInput.click();
        document.body.removeChild(fileInput);
    }

    async processImageFile(file) {
        // Use centralized validation for image files
        const validation = this.validateFileType(
            file,
            this.config.visionSettings.allowedImageTypes,
            this.config.visionSettings.maxImageSize
        );

        if (!validation.valid) {
            this.showToast(validation.error);
            return;
        }

        try {
            // Create image element
            const img = new Image();
            const imageUrl = URL.createObjectURL(file);

            img.onload = async () => {
                // Store image data for next message
                this.pendingImage = {
                    img: img,
                    fileName: file.name,
                    imageUrl: imageUrl
                };

                // Display small thumbnail next to input
                this.displayInputThumbnail(img);

                this.showToast(ChatPlayground.MESSAGES.SUCCESS.IMAGE_READY);
            };

            img.onerror = () => {
                this.showToast(ChatPlayground.MESSAGES.ERRORS.IMAGE_LOAD);
                URL.revokeObjectURL(imageUrl);
            };

            img.src = imageUrl;

        } catch (error) {
            console.error('Error processing image:', error);
            this.showToast(ChatPlayground.MESSAGES.ERRORS.IMAGE_PROCESS);
        }
    }

    displayInputThumbnail(img) {
        // Get the input thumbnail container
        const thumbnailContainer = document.getElementById('input-thumbnail-container');
        const thumbnailImg = document.getElementById('input-thumbnail');
        const removeBtn = document.getElementById('remove-thumbnail-btn');

        // Set the thumbnail image
        thumbnailImg.src = img.src;

        // Show the thumbnail container
        thumbnailContainer.style.display = 'block';

        // Add event listener to remove button (remove old listener first)
        const newRemoveBtn = removeBtn.cloneNode(true);
        removeBtn.parentNode.replaceChild(newRemoveBtn, removeBtn);

        newRemoveBtn.addEventListener('click', () => {
            this.removePendingImage();
        });
    }

    removePendingImage() {
        // Clean up pending image data
        if (this.pendingImage && this.pendingImage.imageUrl) {
            URL.revokeObjectURL(this.pendingImage.imageUrl);
        }
        this.pendingImage = null;

        // Hide thumbnail container
        const thumbnailContainer = document.getElementById('input-thumbnail-container');
        thumbnailContainer.style.display = 'none';
    }

    async classifyImage(img) {
        try {
            // Get predictions from MobileNet
            const predictions = await this.mobileNetModel.classify(img);
            return predictions;
        } catch (error) {
            console.error('Error classifying image:', error);
            throw error;
        }
    }

    formatPredictions(predictions) {
        // Format top 3 predictions as text for the model
        const topPredictions = predictions.slice(0, 3);
        return topPredictions.map((prediction, index) => {
            const className = prediction.className.replace(/_/g, ' ');
            const confidence = Math.round(prediction.probability * 100);
            return `${index + 1}. ${className} (${confidence}% confidence)`;
        }).join('\n');
    }

    attachEventListeners() {
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        // Add keyboard support for collapsible buttons
        const collapsibleButtons = document.querySelectorAll('.collapsible-btn');
        collapsibleButtons.forEach(button => {
            button.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    button.click();
                }
            });
        });

        // Add keyboard support for icon buttons
        const iconButtons = document.querySelectorAll('.icon-btn');
        iconButtons.forEach(button => {
            button.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    button.click();
                }
            });
        });

        // Dynamic system message update
        this.systemMessage.addEventListener('input', () => {
            this.currentSystemMessage = this.systemMessage.value;
        });

        // Attach button (image upload)
        if (this.attachBtn) {
            this.attachBtn.addEventListener('click', () => this.handleImageUpload());
        }

        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
        });

        // Clear chat button (New Chat icon in header)
        const newChatBtn = document.querySelector('.chat-controls .icon-btn:not(.help-btn):not(.config-btn)');
        if (newChatBtn) {
            newChatBtn.addEventListener('click', async () => {
                await this.clearChat();
            });
        }

        // Help/About button
        const helpBtn = document.querySelector('.chat-controls .help-btn');
        if (helpBtn) {
            helpBtn.addEventListener('click', () => {
                window.openAboutModal();
            });
        }

        // Parameters button
        const parametersBtn = document.getElementById('parameters-btn');
        if (parametersBtn) {
            parametersBtn.addEventListener('click', () => {
                window.openParametersModal();
            });
        }

        // Configuration button
        const configBtn = document.querySelector('.config-btn');
        if (configBtn) {
            configBtn.addEventListener('click', () => {
                this.openConfigFlyout();
            });
        }

        // Voice mode toggle
        const voiceModeToggle = document.getElementById('voice-mode-toggle');
        if (voiceModeToggle) {
            voiceModeToggle.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                await this.toggleVoiceMode(isEnabled);
                if (isEnabled) {
                    this.openConfigFlyout();
                }
            });
        }

        // Voice Start button
        const voiceStartBtn = document.getElementById('voice-start-btn');
        if (voiceStartBtn) {
            voiceStartBtn.addEventListener('click', () => {
                this.startVoiceInput();
            });
        }

        // Voice CC (closed captions) button
        const voiceCcBtn = document.getElementById('voice-cc-btn');
        if (voiceCcBtn) {
            voiceCcBtn.addEventListener('click', () => {
                this.toggleCaptions();
            });
        }

        // Voice Cancel button
        const voiceCancelBtn = document.getElementById('voice-cancel-btn');
        if (voiceCancelBtn) {
            voiceCancelBtn.addEventListener('click', () => {
                this.cancelVoiceInteraction();
            });
        }

        // Voice select dropdown
        const voiceSelect = document.getElementById('config-voice-select');
        if (voiceSelect) {
            voiceSelect.addEventListener('change', async (e) => {
                await this.handleVoiceSelectionChange(e.target.value);
            });
        }

        // Preview voice button
        const previewVoiceBtn = document.getElementById('preview-voice-btn');
        if (previewVoiceBtn) {
            previewVoiceBtn.addEventListener('click', () => {
                this.previewVoice();
            });
        }

        // Avatar toggle
        const avatarToggle = document.getElementById('avatar-toggle');
        if (avatarToggle) {
            avatarToggle.addEventListener('change', (e) => {
                this.toggleAvatar(e.target.checked);
            });
        }

        // Configuration flyout close button
        const closeFlyoutBtn = document.getElementById('close-config-flyout');
        if (closeFlyoutBtn) {
            closeFlyoutBtn.addEventListener('click', () => {
                this.closeConfigFlyout();
            });
        }

        // Configuration flyout overlay click to close
        const flyoutOverlay = document.getElementById('config-flyout-overlay');
        if (flyoutOverlay) {
            flyoutOverlay.addEventListener('click', (e) => {
                // Only close if clicking the overlay itself, not the panel
                if (e.target === flyoutOverlay) {
                    this.closeConfigFlyout();
                }
            });
        }

        // Voice input fallback modal controls
        const voiceErrorModal = document.getElementById('voice-input-error-modal');
        const voiceErrorCloseBtn = document.getElementById('voice-input-error-close');
        const voiceErrorCancelBtn = document.getElementById('voice-input-error-cancel');
        const voiceErrorSubmitBtn = document.getElementById('voice-input-error-submit');
        const voiceFallbackInput = document.getElementById('voice-fallback-input');

        if (voiceErrorCloseBtn) {
            voiceErrorCloseBtn.addEventListener('click', () => this.closeVoiceInputErrorModal());
        }

        if (voiceErrorCancelBtn) {
            voiceErrorCancelBtn.addEventListener('click', () => this.closeVoiceInputErrorModal());
        }

        if (voiceErrorSubmitBtn) {
            voiceErrorSubmitBtn.addEventListener('click', () => this.submitTypedVoiceFallback());
        }

        if (voiceFallbackInput) {
            voiceFallbackInput.addEventListener('input', () => {
                const hasText = voiceFallbackInput.value.trim().length > 0;
                if (voiceErrorSubmitBtn) {
                    voiceErrorSubmitBtn.disabled = !hasText;
                }
            });

            voiceFallbackInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.submitTypedVoiceFallback();
                }
            });
        }

        if (voiceErrorModal) {
            voiceErrorModal.addEventListener('click', (e) => {
                if (e.target === voiceErrorModal) {
                    this.closeVoiceInputErrorModal();
                }
            });
        }

        // Speech Model Modal event listeners
        const speechModelModal = document.getElementById('speech-model-modal');
        const speechModelCancelBtn = document.getElementById('speech-model-cancel');
        const speechModelRetryBtn = document.getElementById('speech-model-retry');

        if (speechModelCancelBtn) {
            speechModelCancelBtn.addEventListener('click', () => this.cancelSpeechModelLoading());
        }

        if (speechModelRetryBtn) {
            speechModelRetryBtn.addEventListener('click', () => this.retrySpeechInput());
        }

        if (speechModelModal) {
            speechModelModal.addEventListener('click', (e) => {
                if (e.target === speechModelModal) {
                    // Don't close on overlay click - require explicit button click
                }
            });
        }

        // Model selection change
        this.modelSelect.addEventListener('change', () => this.handleModelChange());
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

    async initializeModel() {
        // Setup cancel link event listener first
        this.setupCancelLink();

        try {
            await this.initializeEngine();
        } catch (error) {
            console.error('Failed to initialize AI engine:', error);
        }
    }

    setupCancelLink() {
        const cancelLink = document.getElementById('cancel-model-link');
        if (cancelLink) {
            // Click handler
            cancelLink.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.cancelModelLoad();
            });

            // Keyboard handler for accessibility
            cancelLink.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    this.cancelModelLoad();
                }
            });
        }
    }

    cancelModelLoad() {
        // Set cancellation flag
        this.modelLoadingCancelled = true;

        // Abort any ongoing loading operations
        if (this.modelLoadingAbortController) {
            this.modelLoadingAbortController.abort();
        }

        // Clean up any partially loaded wllama instance
        if (this.wllama) {
            const _old = this.wllama;
            this.wllama = null;
            _old.exit().catch(() => { });
        }

        // Hide the cancel link
        const cancelLink = document.getElementById('cancel-model-link');
        if (cancelLink) {
            cancelLink.style.display = 'none';
        }

        // Switch to None mode (don't mark models as unavailable)
        this.usingWllama = false;
        this.usingWikipedia = true;
        this.currentMode = 'none';
        this.currentModelId = 'None (Wikipedia)';
        this.config.modelParameters = this.getModelDefaults();
        this.updateParameterUI();
        this.populateModelDropdown();

        this.updateProgress(100, 'Model loading cancelled - None (Wikipedia) mode ready');

        setTimeout(() => {
            this.enableUI();
            this.progressContainer.style.display = 'none';
        }, 1500);
    }

    async initializeEngine() {
        // Check hardware requirements before attempting to load model
        if (!this.checkHardwareRequirements()) {
            console.log('Hardware requirements not met, using Wikipedia mode only');
            this.wllamaFailed = true;
            this.wllamaLoaded = false;
            this.usingWllama = false;
            this.usingWikipedia = true;
            this.currentMode = 'none';
            this.currentModelId = 'None (Wikipedia)';
            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
            this.updateProgress(100, 'Wikipedia mode ready!<br><small style="font-size: 0.9em; color: #666;">Your device does not meet the minimum requirements (8GB RAM, 8 CPU cores) for the AI model.</small>', true);
            setTimeout(() => {
                this.enableUI();
            }, 2000);
            return;
        }

        this.modelLoadingAbortController = new AbortController();

        try {
            await this.initializeWllama();

            if (this.modelLoadingCancelled) {
                console.log('Model loading was cancelled by user');
                return;
            }

            console.log('Wllama initialized successfully');
            this.usingWllama = true;
            this.usingWikipedia = false;
            this.currentMode = 'phi-cpu';
            this.wllamaLoaded = true;
            this.wllamaFailed = false;
            this.currentModelId = 'Phi-3.5-mini-GGUF';

            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
        } catch (wllamaError) {
            if (this.modelLoadingCancelled || wllamaError.message.includes('cancelled by user')) {
                console.log('Model loading was cancelled by user');
                return;
            }

            console.error('Wllama initialization failed:', wllamaError);
            this.wllamaLoaded = false;
            this.wllamaFailed = true;
            this.usingWllama = false;
            this.usingWikipedia = true;
            this.currentMode = 'none';
            this.currentModelId = 'None (Wikipedia)';
            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
            this.updateProgress(100, 'Wikipedia mode ready! (No local model)');
            setTimeout(() => {
                this.enableUI();
            }, 2000);
        }
    }

    async initializeWllama(progressCallback) {
        console.log('Initializing wllama...');
        this.wllamaLoaded = false;

        const CONFIG_PATHS = {
            default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/wasm/wllama.wasm',
        };

        const useMultiThread = window.crossOriginIsolated === true;
        const availableThreads = navigator.hardwareConcurrency || 4;
        const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;
        console.log(`Cross-origin isolated: ${window.crossOriginIsolated}, available threads: ${availableThreads}, attempting ${preferredThreads} thread(s)`);

        const modelRef = {
            //repo: 'unsloth/Phi-4-mini-instruct-GGUF',
            repo: 'bartowski/Phi-3.5-mini-instruct-GGUF',
            quant: 'Q4_K_M'
        };

        const baseParams = {
            n_ctx: 712,
            progressCallback: ({ loaded, total }) => {
                if (this.modelLoadingCancelled) return;
                const percentage = Math.round((loaded / total) * 100);
                const adjusted = Math.round(20 + percentage * 0.8);
                this.updateProgress(adjusted, `Loading Phi 3.5-mini: ${percentage}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`, true);
                this.showCancelLink();
                if (progressCallback) progressCallback(loaded, total);
            }
        };

        // Detect GPU vendor; disable WebGPU for known-broken implementations or
        // if a previous GPU session crashed (this.gpuFailed=true).
        let GPU_ENABLED = !this.gpuFailed && !!navigator.gpu;
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

        // Helper: create a fresh Wllama instance and load the model.
        const attemptLoad = async (n_gpu_layers, n_threads) => {
            this.wllama = new Wllama(CONFIG_PATHS);
            await this.wllama.loadModelFromHF(modelRef, { ...baseParams, n_gpu_layers, n_threads });
        };

        if (this.modelLoadingCancelled) {
            throw new Error('Model loading cancelled by user');
        }

        this.updateProgress(10, 'Loading Phi 3.5-mini...');
        this.updateProgress(20, 'Downloading model...');

        const loadWithFallback = async () => {
            if (GPU_ENABLED) {
                try {
                    // Full GPU offload (all 32 layers). Full offload avoids the precision
                    // mismatch at CPU/GPU layer boundaries that caused garbled tokens.
                    await attemptLoad(32, preferredThreads);
                    this.wllamaUsedGPU = true;
                    console.log(`Wllama initialized with GPU (32 layers) + ${preferredThreads} thread(s)`);
                    return;
                } catch (gpuErr) {
                    if (this.modelLoadingCancelled) throw gpuErr;
                    console.warn(`GPU initialization failed (${gpuErr.message}), falling back to CPU`);
                    if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
                }
            } else {
                console.log('Skipping GPU: using CPU directly');
            }

            // CPU multi-threaded
            try {
                await attemptLoad(0, preferredThreads);
                this.wllamaUsedGPU = false;
                console.log(`Wllama initialized on CPU with ${preferredThreads} thread(s)`);
            } catch (cpuErr) {
                if (this.modelLoadingCancelled) throw cpuErr;
                if (preferredThreads > 1) {
                    console.warn(`Multi-thread CPU init failed (${cpuErr.message}), retrying with 1 thread`);
                    if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
                    // Final attempt: CPU single-threaded
                    await attemptLoad(0, 1);
                    this.wllamaUsedGPU = false;
                    console.log('Wllama initialized on CPU with 1 thread');
                } else {
                    throw cpuErr;
                }
            }
        };

        await loadWithFallback();

        if (this.modelLoadingCancelled) {
            if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
            throw new Error('Model loading cancelled by user');
        }

        this.wllamaLoaded = true;

        const cancelLink = document.getElementById('cancel-model-link');
        if (cancelLink) cancelLink.style.display = 'none';

        this.updateProgress(100, 'AI model ready!');
        setTimeout(() => {
            this.progressContainer.style.display = 'none';
            this.enableUI();
        }, 1000);

        console.log(`Wllama initialized successfully (GPU: ${this.wllamaUsedGPU})`);
    }



    showCancelLink() {
        const cancelLink = document.getElementById('cancel-model-link');
        if (cancelLink && !this.modelLoadingCancelled) {
            cancelLink.style.display = 'inline-block';
        }
    }

    updateProgress(percentage, text, useHTML = false) {
        this.progressFill.style.width = `${percentage}%`;
        if (useHTML) {
            this.progressText.innerHTML = text;
        } else {
            this.progressText.textContent = text;
        }
    }

    enableUI() {
        this.isModelLoaded = true;
        this.modelSelect.disabled = false;
        this.systemMessage.disabled = false;
        this.userInput.disabled = false;
        this.sendBtn.disabled = false;
        this.updateAttachButtonState(); // Update attach button based on vision settings

        // Enable voice mode start button
        const voiceStartBtn = document.getElementById('voice-start-btn');
        if (voiceStartBtn) {
            voiceStartBtn.disabled = false;
        }

        this.userInput.focus();

        // Populate model dropdown with available models
        this.populateModelDropdown();

        // Set parameter controls based on active mode
        this.setParameterControlsEnabled(this.currentMode !== 'none');
    }

    disableUI() {
        this.isModelLoaded = false;
        this.systemMessage.disabled = true;
        this.userInput.disabled = true;
        this.sendBtn.disabled = true;
        this.attachBtn.disabled = true;

        // Disable voice mode start button
        const voiceStartBtn = document.getElementById('voice-start-btn');
        if (voiceStartBtn) {
            voiceStartBtn.disabled = true;
        }
    }

    async handleModelChange() {
        const selectedValue = this.modelSelect.value;

        if (this.isGenerating) {
            console.log('Cannot switch models while generating');
            // Reset to current model
            this.populateModelDropdown();
            return;
        }

        // Reset cancellation flag when user manually switches models
        this.modelLoadingCancelled = false;

        // Determine if we're actually switching models
        const previousMode = this.currentMode;

        if (selectedValue === 'phi-cpu') {
            if (this.wllamaFailed) {
                alert('Phi 3.5-mini is not available because it previously failed to load.');
                this.populateModelDropdown();
                return;
            }

            if (!this.wllamaLoaded) {
                console.log('Loading wllama for the first time...');

                this.disableUI();
                this.progressContainer.style.display = 'block';

                try {
                    await this.initializeWllama();

                    this.usingWllama = true;
                    this.usingWikipedia = false;
                    this.currentMode = 'phi-cpu';
                    this.currentModelId = 'Phi-3.5-mini-GGUF';
                    this.wllamaFailed = false;

                    this.config.modelParameters = this.getModelDefaults();
                    this.updateParameterUI();
                    this.setParameterControlsEnabled(true);

                    await this.clearChat();
                    this.showToast('Switched to Phi 3.5-mini - Conversation restarted');

                    console.log('Switched to Phi 3.5-mini');
                } catch (error) {
                    console.error('Failed to load wllama:', error);

                    if (this.modelLoadingCancelled || error.message.includes('cancelled by user')) {
                        console.log('Model loading was cancelled by user - Wllama stays available');
                        return;
                    }

                    this.wllamaLoaded = false;
                    this.wllamaFailed = true;
                    this.usingWllama = false;
                    this.usingWikipedia = true;
                    this.currentMode = 'none';
                    this.currentModelId = 'None (Wikipedia)';
                    this.config.modelParameters = this.getModelDefaults();
                    this.updateParameterUI();
                    this.populateModelDropdown();
                    alert('Failed to load Phi 3.5-mini. Switching to None (Wikipedia mode).');
                    this.enableUI();
                }
            } else {
                this.usingWllama = true;
                this.usingWikipedia = false;
                this.currentMode = 'phi-cpu';
                this.currentModelId = 'Phi-3.5-mini-GGUF';

                this.config.modelParameters = this.getModelDefaults();
                this.updateParameterUI();
                this.setParameterControlsEnabled(true);

                if (previousMode !== this.currentMode) {
                    await this.clearChat();
                    this.showToast('Switched to Phi 3.5-mini - Conversation restarted');
                }

                console.log('Switched to Phi 3.5-mini');
            }
        } else if (selectedValue === 'none') {
            this.usingWllama = false;
            this.usingWikipedia = true;
            this.currentMode = 'none';
            this.currentModelId = 'None (Wikipedia)';
            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
            this.setParameterControlsEnabled(false);

            if (previousMode !== this.currentMode) {
                await this.clearChat();
                this.showToast('Switched to None (Wikipedia mode) - Conversation restarted');
            }

            console.log('Switched to None (Wikipedia mode)');
        }

        this.populateModelDropdown();
    }

    populateModelDropdown() {
        // Clear existing options
        this.modelSelect.innerHTML = '';

        // Add Phi 3.5-mini option
        const cpuOption = document.createElement('option');
        cpuOption.value = 'phi-cpu';
        cpuOption.textContent = 'Phi 3.5-mini';
        cpuOption.disabled = this.wllamaFailed;
        this.modelSelect.appendChild(cpuOption);

        // Add no-model Wikipedia option
        const noneOption = document.createElement('option');
        noneOption.value = 'none';
        noneOption.textContent = 'None';
        this.modelSelect.appendChild(noneOption);

        const canUseCpu = !this.wllamaFailed;

        if (this.currentMode === 'phi-cpu' && canUseCpu) {
            this.modelSelect.value = 'phi-cpu';
        } else {
            this.currentMode = 'none';
            this.usingWikipedia = true;
            this.usingWllama = false;
            this.currentModelId = 'None (Wikipedia)';
            this.modelSelect.value = 'none';
        }
    }



    setParameterControlsEnabled(enabled) {
        // Enable or disable all parameter sliders
        const parameterSliders = [
            'temperature-slider',
            'top-p-slider',
            'max-tokens-slider',
            'repetition-penalty-slider'
        ];

        parameterSliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.disabled = !enabled;
                // Update visual appearance
                slider.style.opacity = enabled ? '1' : '0.5';
                slider.style.cursor = enabled ? 'pointer' : 'not-allowed';
            }
        });
    }



    // Extract keywords from text (excluding common stopwords)
    extractKeywords(text) {
        const stopwords = new Set([
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

        // Extract words, convert to lowercase, filter stopwords and short words
        const words = text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopwords.has(word));

        // Return unique keywords
        return [...new Set(words)];
    }

    // Strip punctuation from text, keeping only alphanumeric and whitespace
    stripPunctuation(text) {
        return text.replace(/[^a-z0-9\s]/g, ' ');
    }

    // Extract relevant lines from file content based on keywords
    extractRelevantLines(fileContent, keywords) {
        if (!fileContent || !keywords || keywords.length === 0) {
            return '';
        }

        const lines = fileContent.split('\n');
        let bestLine = '';
        let bestCount = 0;

        for (const line of lines) {
            // Strip punctuation and lowercase for comparison
            const lineWords = this.stripPunctuation(line.toLowerCase()).split(/\s+/);
            const count = keywords.filter(keyword => lineWords.includes(keyword)).length;
            if (count > bestCount) {
                bestCount = count;
                bestLine = line.trim();
            }
        }

        return bestLine;
    }

    async handleSendMessage() {
        // If already generating, stop instead of sending
        if (this.isGenerating) {
            await this.stopGeneration();
            return;
        }

        if (!this.isModelLoaded) return;

        let userMessage = this.userInput.value.trim();
        if (!userMessage && !this.pendingImage) return;
        if (!userMessage) userMessage = ""; // Allow empty message if there's an image

        const currentSystemPrompt = (this.systemMessage?.value ?? this.currentSystemMessage).trim();
        this.currentSystemMessage = currentSystemPrompt;

        const hasProhibitedSystemPrompt = currentSystemPrompt && this.containsProhibitedContent(currentSystemPrompt);
        const hasProhibitedUserPrompt = userMessage && this.containsProhibitedContent(userMessage);

        if (hasProhibitedSystemPrompt || hasProhibitedUserPrompt) {
            await this.handleModerationFailure(hasProhibitedUserPrompt ? userMessage : '');
            return;
        }

        // Log the current system prompt to console
        console.log('Current system prompt:', this.currentSystemMessage);

        // Process pending image if exists
        let imageAnalysis = '';
        let imageElement = null;

        if (this.pendingImage) {
            try {
                // Get image analysis (requires MobileNet to be pre-loaded)
                const predictions = await this.classifyImage(this.pendingImage.img);
                imageAnalysis = predictions[0].className.replace(/_/g, ' ')

                // Create image element for message bubble
                imageElement = document.createElement('img');
                imageElement.src = this.pendingImage.img.src;
                imageElement.className = 'message-image';
                imageElement.alt = this.pendingImage.fileName;

            } catch (error) {
                console.error('Error analyzing image:', error);
                this.showToast('Error analyzing image. Sending message without analysis.');
            }
        }

        // Reset stop state and typing state
        this.stopRequested = false;
        if (this.typingState) {
            this.typingState.isTyping = false;
            this.typingState = null;
        }

        // Web search intercept
        if (this.isWebSearchActive() && /^(find|search)\b/i.test(userMessage)) {
            this.addMessage('user', userMessage, imageElement);
            this.userInput.value = '';
            this.userInput.style.height = 'auto';
            if (this.pendingImage) this.removePendingImage();
            const keywords = this.extractWebSearchKeywords(userMessage, imageAnalysis || null);
            this.isGenerating = true;
            this.updateUIForGeneration(true);
            const typingIndicator = this.addTypingIndicator();
            await new Promise(resolve => setTimeout(resolve, 600));
            typingIndicator.remove();
            const msgEl = this.addMessage('assistant', '');
            const contentEl = msgEl.querySelector('.message-content');
            if (keywords) {
                const url = `https://www.bing.com/search?q=${encodeURIComponent(keywords)}`;
                const plainText = `OK, I searched the web for you.\nHere's what I found.`;
                const finalHtml = `OK, I searched the web for you.<br><a href="${url}" target="_blank" rel="noopener noreferrer">Here's what I found.</a>`;
                await this.typeResponseWithFinalHtml(contentEl, plainText, finalHtml);
            } else {
                await this.typeResponse(contentEl, 'Please enter a more specific search query.');
            }
            return;
        }

        // Text summarization intercept (Basic mode only)
        const isBasicMode = this.currentMode === 'none' || this.usingWikipedia;
        if (isBasicMode) {
            const lines = userMessage.split('\n');
            const firstLine = lines[0].trim().toLowerCase();

            // Check for Summarization Command
            if (lines.length > 1 && firstLine.startsWith('summarize')) {
                const contentToAnalyze = lines.slice(1).join('\n');

                this.addMessage('user', userMessage, imageElement);
                this.userInput.value = '';
                this.userInput.style.height = 'auto';
                if (this.pendingImage) this.removePendingImage();

                this.isGenerating = true;
                this.updateUIForGeneration(true);

                // Show typing indicator
                const typingIndicator = this.addTypingIndicator();

                // Simulate "reading" delay
                await new Promise(resolve => setTimeout(resolve, 1000));

                if (this.stopRequested) {
                    typingIndicator.remove();
                    this.isGenerating = false;
                    this.updateUIForGeneration(false);
                    return;
                }

                const summary = this.summarizeText(contentToAnalyze);
                typingIndicator.remove();

                const assistantMessageEl = this.addMessage('assistant', '');
                const contentEl = assistantMessageEl.querySelector('.message-content');

                const summaryText = `<b>Summary:</b><br><br>${this.renderMarkdown(summary)}`;
                await this.typeResponse(contentEl, summary);

                // Add to conversation history
                this.conversationHistory.push({ role: 'user', content: this.getFirstSentence(userMessage) });
                this.conversationHistory.push({ role: 'assistant', content: this.getFirstSentence(summary) });

                return;
            }
        }

        // Add user message to chat (with image if available)
        this.addMessage('user', userMessage, imageElement);

        // Clean up input and pending image
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        // Clean up pending image
        if (this.pendingImage) {
            this.removePendingImage();
        }

        this.isGenerating = true;
        this.updateUIForGeneration(true);

        // Show typing indicator
        const typingIndicator = this.addTypingIndicator();

        try {
            // Update conversation history to reflect current system message
            this.updateConversationHistoryWithCurrentSystemMessage();

            // Prepare conversation history
            const systemContent = this.getEffectiveSystemMessage() + (imageAnalysis ? '\nKeep responses short and succinct.' : '');
            const messages = [
                { role: "system", content: systemContent }
            ];

            // Add only the last conversation pair (previous prompt + response)
            // History already stores first-sentence-only content, so no truncation needed here
            messages.push(...this.conversationHistory.slice(-2));

            // Add user message with image analysis and file context if available
            let finalUserMessage = userMessage;
            if (imageAnalysis) {
                finalUserMessage += '\n\n[Current image shows: ' + imageAnalysis + ']';
            }

            // If file is uploaded, extract the most relevant line and append to user message
            this.fileContentUsedInPrompt = false;
            if (this.config.fileUpload.content) {
                const keywords = this.extractKeywords(userMessage);
                console.log('Extracted keywords from user prompt:', keywords);

                const relevantLine = this.extractRelevantLines(this.config.fileUpload.content, keywords);

                if (relevantLine) {
                    console.log('Found most relevant line from file:', relevantLine);
                    finalUserMessage += '\nAnswer with a single, succinct sentence based on this information:\n' + relevantLine;
                    this.fileContentUsedInPrompt = true;
                } else {
                    console.log('No relevant lines found in file for the given keywords, treating as normal prompt');
                }
            }

            // If system prompt indicates short response is wanted and there's no image/file context, guide the model
            if (!imageAnalysis && !this.fileContentUsedInPrompt) {
                const promptFromUi = (this.systemMessage?.value || this.currentSystemMessage || '').toLowerCase();
                const wantsShortResponse = /\b(short|concise|brief|succinct)\b/i.test(promptFromUi);
                if (wantsShortResponse) {
                    finalUserMessage += '\nRespond with a single, short paragraph.';
                }
            }

            messages.push({ role: "user", content: finalUserMessage });

            // Log the complete prompt being sent to the model
            console.log('=== COMPLETE PROMPT BEING SENT TO MODEL ===');
            console.log('Current System Message (from UI):', this.currentSystemMessage);
            console.log('Effective System Message (with file data):', this.getEffectiveSystemMessage());
            console.log('Model:', this.currentModelId);
            console.log('Total messages:', messages.length);
            console.log('Messages:');
            messages.forEach((msg, index) => {
                console.log(`\n[${index}] Role: ${msg.role}`);
                console.log(`Content: ${msg.content}`);
            });
            console.log('\n=== END PROMPT ===\n');

            // Remove typing indicator
            typingIndicator.remove();

            // Add thinking indicator with animated dots
            const thinkingIndicator = this.addThinkingIndicator();

            // Route to the appropriate engine
            if (this.currentMode === 'none' || this.usingWikipedia) {
                await this.handleWikipediaMode(thinkingIndicator, userMessage, imageAnalysis);
            } else if (this.usingWllama) {
                await this.handleWllamaMode(messages, thinkingIndicator, userMessage, imageAnalysis);
            } else {
                throw new Error('No AI model available. Please wait for Phi 3.5-mini to load or switch to Wikipedia mode.');
            }

        } catch (error) {
            console.error('Error generating response:', error);
            if (typingIndicator.parentNode) {
                typingIndicator.remove();
            }
            // Remove thinking indicator if it exists
            const thinkingIndicator = this.chatMessages.querySelector('.thinking-indicator');
            if (thinkingIndicator) {
                thinkingIndicator.remove();
            }

            const errorMessage = 'Sorry, I encountered an error while generating a response. Please try restarting the conversation. If this happens repeatedly, try switching to a different model.';
            const assistantMessageEl = this.addMessage('assistant', '');
            const contentEl = assistantMessageEl.querySelector('.message-content');

            // Type out the error message
            await this.typeResponse(contentEl, errorMessage);
        } finally {
            this.isGenerating = false;
            this.updateUIForGeneration(false);
        }
    }

    renderMarkdown(text) {
        if (!text) return '';
        // Escape HTML first to prevent XSS
        const escaped = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        // Convert markdown patterns (bold before italic to avoid partial matches)
        return escaped
            .replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/gs, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }


    // Helper function to check for prohibited content
    containsProhibitedContent(text) {
        if (!text || typeof text !== 'string') return false;

        // Convert to lowercase for case-insensitive matching
        const lowerText = text.toLowerCase();

        for (const term of this.prohibitedTerms) {
            const regex = new RegExp(`\\b${this.escapeRegex(term)}\\b`, 'i');
            if (regex.test(lowerText)) {
                return true;
            }
        }

        return false;
    }

    async handleModerationFailure(userMessage = '') {
        if (userMessage) {
            this.addMessage('user', userMessage);
            this.userInput.value = '';
            this.userInput.style.height = 'auto';
        }

        const assistantMessageEl = this.addMessage('assistant', '');
        const contentEl = assistantMessageEl.querySelector('.message-content');
        await this.typeResponse(contentEl, ChatPlayground.MESSAGES.MODERATION.BLOCKED);
        this.userInput.focus();
    }

    // Helper function to extract first sentence from text
    getFirstSentence(text) {
        if (!text) return '';

        // Find first sentence-ending punctuation: . ! : ?
        // Require that it's followed by whitespace, newline, or end of string
        // This prevents breaking on email addresses like "expenses@contoso.com"
        const match = text.match(/^[^.!:?]+[.!:?](?=\s|$)/);
        if (match) {
            return match[0];
        }

        // No sentence-ending punctuation found, take first 60 characters
        return text.substring(0, 60);
    }

    extractLeadingSentences(text, maxSentences = 2) {
        const input = (text ?? '').trim();
        if (!input) return '';

        let sentenceCount = 0;
        let endIndex = -1;

        for (let i = 0; i < input.length; i++) {
            const char = input[i];
            if (char !== '.' && char !== '!' && char !== '?') continue;

            const prev = i > 0 ? input[i - 1] : '';
            const next = i < input.length - 1 ? input[i + 1] : '';

            // Skip if period is between digits (e.g., "3.14")
            if (char === '.' && /\d/.test(prev) && /\d/.test(next)) {
                continue;
            }

            // Only treat as sentence boundary if followed by space, newline, or end of string
            // This prevents breaking on email addresses like "expenses@contoso.com"
            if (next && !/\s/.test(next)) {
                continue;
            }

            sentenceCount += 1;
            if (sentenceCount >= maxSentences) {
                endIndex = i + 1;
                break;
            }
        }

        if (endIndex === -1) {
            return input;
        }

        return input.slice(0, endIndex).trim();
    }

    async generateWithWikipedia(query, imageClassName = '') {
        try {
            const firstLine = (query || '').split('\n')[0];
            const queryKeywords = this.extractKeywords(firstLine);
            const imageKeywords = imageClassName ? this.extractKeywords(imageClassName) : [];

            const combinedKeywords = [...new Set([...(queryKeywords || []), ...(imageKeywords || [])])];
            const keywords = combinedKeywords.join(' ');
            if (!keywords) return null;

            const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(keywords)}&format=json&origin=*&srlimit=1`;
            const searchResponse = await fetch(searchUrl);
            if (!searchResponse.ok) throw new Error('Wikipedia search request failed');

            const searchData = await searchResponse.json();
            const results = searchData?.query?.search;
            if (!results || results.length === 0) return null;

            const title = results[0].title;
            const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
            const summaryResponse = await fetch(summaryUrl);
            if (!summaryResponse.ok) throw new Error('Wikipedia summary request failed');

            const summaryData = await summaryResponse.json();
            const extract = summaryData?.extract;
            if (!extract || extract.length < 20) return null;

            const firstParagraph = extract.split('\n').find(p => p.trim().length > 0) || extract;
            const extractedSentences = this.extractLeadingSentences(firstParagraph, 2);
            const result = extractedSentences || firstParagraph.substring(0, 300).trim();

            return result.length >= 20 ? result : null;
        } catch (error) {
            console.error('Wikipedia lookup failed:', error);
            return null;
        }
    }

    async generateNoneModeResponse(query, imageClassName = '') {
        const firstLine = (query || '').split('\n')[0];

        if (this.config.fileUpload.content) {
            const keywords = this.extractKeywords(firstLine);
            const bestLine = this.extractRelevantLines(this.config.fileUpload.content, keywords);

            if (bestLine) {
                this.fileContentUsedInPrompt = true;
                return bestLine;
            }

            return "I couldn't find a relevant line in the uploaded file for your prompt keywords.";
        }

        const wikiResponse = await this.generateWithWikipedia(query, imageClassName);
        if (wikiResponse) {
            return `${wikiResponse}\n\n(Source: Wikipedia)`;
        }

        return "I'm sorry. I couldn't find a relevant Wikipedia article for that prompt.";
    }

    maybeMutateNoneModeResponse(text) {
        const isNoneMode = this.currentMode === 'none' || this.usingWikipedia;
        const temperature = Number(this.config?.modelParameters?.temperature ?? 0);

        if (!isNoneMode || !(temperature > 1.0) || !text) {
            return text;
        }

        const wordMatches = [...text.matchAll(/[A-Za-z0-9']+/g)];
        if (wordMatches.length < 2) {
            return text;
        }

        const maxWordsToReverse = Math.min(4, wordMatches.length);
        const wordsToReverseCount = 2 + Math.floor(Math.random() * (maxWordsToReverse - 1));
        const selectedIndexes = new Set();

        while (selectedIndexes.size < wordsToReverseCount) {
            selectedIndexes.add(Math.floor(Math.random() * wordMatches.length));
        }

        const replacements = Array.from(selectedIndexes)
            .map(index => wordMatches[index])
            .sort((a, b) => b.index - a.index);

        let mutatedText = text;
        for (const match of replacements) {
            const start = match.index;
            const end = start + match[0].length;
            mutatedText = mutatedText.slice(0, start)
                + this.reverseWord(match[0])
                + mutatedText.slice(end);
        }

        return mutatedText;
    }

    maybeShortenNoneModeResponse(text) {
        const isNoneMode = this.currentMode === 'none' || this.usingWikipedia;
        if (!isNoneMode || !text) {
            return text;
        }

        const promptFromUi = (this.systemMessage?.value || this.currentSystemMessage || '').toLowerCase();
        const wantsShortResponse = /\b(short|concise|brief|succinct)\b/i.test(promptFromUi);

        if (!wantsShortResponse) {
            return text;
        }

        return this.extractLeadingSentences(text, 1);
    }

    /**
     * Summarizes text using TextRank algorithm
     * @param {string} text - The text to summarize
     * @returns {string} Summary of top 3 sentences
     */
    summarizeText(text) {
        // 1. Split into sentences (simple approximation)
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

        if (sentences.length <= 3) return sentences.join(' ');

        // 2. Tokenize sentences
        const tokenizedSentences = sentences.map(s => {
            return s.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => !ChatPlayground.STOPWORDS.has(w) && w.length > 0);
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

    async handleWikipediaMode(thinkingIndicator, userMessage, imageAnalysis = '') {
        thinkingIndicator.remove();

        const assistantMessageEl = this.addMessage('assistant', 'Searching Wikipedia...');
        const contentEl = assistantMessageEl.querySelector('.message-content');

        // Reset file content usage tracking
        this.fileContentUsedInPrompt = false;

        let responseText = await this.generateNoneModeResponse(userMessage, imageAnalysis);
        responseText = this.maybeShortenNoneModeResponse(responseText);
        responseText = this.maybeMutateNoneModeResponse(responseText);

        if (!this.isGenerating) {
            return;
        }

        // Append file attribution if a file is uploaded and relevant content was used (for display only)
        let displayResponse = responseText;
        if (this.fileContentUsedInPrompt && this.config.fileUpload.fileName && responseText.trim()) {
            displayResponse = responseText + `\n(Ref: ${this.config.fileUpload.fileName})`;
        }

        await this.typeResponse(contentEl, displayResponse);

        // Add to conversation history (without file attribution, to prevent cumulative citations)
        // Store only the first sentence so cached tokens match what is sent on the next request
        this.conversationHistory.push({ role: 'user', content: this.getFirstSentence(userMessage) });
        this.conversationHistory.push({ role: 'assistant', content: this.getFirstSentence(responseText) });
    }

    // Helper function to remove a trailing incomplete sentence
    trimIncompleteFinalSentence(text) {
        if (!text) return '';

        const trimmedText = text.trimEnd();
        if (!trimmedText) return '';

        // Keep structured multi-line outputs (lists, bullets, etc.) untouched.
        // Sentence heuristics are unreliable for numbered entries like "1.".
        const lines = trimmedText.split(/\r?\n/).filter(line => line.trim().length > 0);
        const hasListLikeLine = lines.some(line => /^\s*(?:[-*]|\d+[.)])\s+/.test(line));
        if (hasListLikeLine) {
            return trimmedText;
        }

        // If the response already ends with sentence-final punctuation, keep it as-is
        if (/[.!?]["')\]]*$/.test(trimmedText)) {
            return trimmedText;
        }

        // Find the last complete sentence boundary and remove trailing partial sentence
        const match = trimmedText.match(/([.!?]["')\]]*)(?![\s\S]*[.!?]["')\]]*)/);
        if (!match) {
            // If no sentence boundary exists, preserve the response instead of dropping it.
            return trimmedText;
        }

        const sentenceEndIndex = match.index + match[0].length;
        return trimmedText.slice(0, sentenceEndIndex).trimEnd();
    }

    /**
     * HTML-escapes text then converts basic markdown (bold, italic, paragraphs)
     * to safe HTML for display in the response bubble.
     */
    formatResponse(text) {
        if (!text) return '';
        // Escape HTML first to prevent XSS
        let formatted = escapeHtml(text);
        // **bold** → <strong>
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // *italic* → <em>
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // Wrap double-newline-separated blocks as paragraphs, single newlines as <br>
        formatted = formatted.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
        return formatted;
    }

    async handleWllamaMode(messages, thinkingIndicator, userMessage, imageAnalysis = '', isGpuRetry = false, existingContentEl = null) {
        if (!this.wllama) {
            throw new Error('Wllama is not initialized. Please wait for AI mode to finish loading.');
        }

        if (thinkingIndicator) thinkingIndicator.remove();

        let contentEl;
        if (existingContentEl) {
            contentEl = existingContentEl;
        } else {
            const assistantMessageEl = this.addMessage('assistant', '');
            contentEl = assistantMessageEl.querySelector('.message-content');
        }

        contentEl.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><p style="font-size: 0.85em; color: #666; margin: 8px 0 0 0; font-style: italic;">(I\'m working on a response. Thanks for your patience!)</p>';
        contentEl.style.width = 'fit-content';
        contentEl.style.whiteSpace = 'normal';

        this.fileContentUsedInPrompt = false;
        if (this.config.fileUpload.content) {
            const keywords = this.extractKeywords(userMessage);
            const relevantLines = this.extractRelevantLines(this.config.fileUpload.content, keywords);
            if (relevantLines) {
                this.fileContentUsedInPrompt = true;
                const lastMsg = messages[messages.length - 1];
                if (lastMsg && lastMsg.role === 'user') {
                    lastMsg.content += '\nRespond based only on the following information:\n' + relevantLines;
                }
            }
        }

        const controller = new AbortController();
        this.currentAbortController = controller;

        let fullResponse = '';
        let completion = null;

        try {
            completion = await this.wllama.createChatCompletion({
                messages,
                max_tokens: this.config.modelParameters.max_tokens,
                temperature: this.config.modelParameters.temperature,
                top_k: 30,
                top_p: this.config.modelParameters.top_p,
                repeat_penalty: this.config.modelParameters.repetition_penalty,
                repeat_last_n: 64,
                cache_prompt: false, // Prevent KV cache accumulation to avoid memory buffer errors with small context window
                abortSignal: controller.signal,
                stream: true
            });

            this.currentStream = completion;

            let firstChunkReceived = false;

            for await (const chunk of completion) {
                if (this.stopRequested) {
                    console.log('Wllama generation stopped by user');
                    break;
                }

                const tokenText = chunk.choices?.[0]?.delta?.content ?? '';
                if (tokenText) {
                    if (!firstChunkReceived) {
                        firstChunkReceived = true;
                        contentEl.textContent = '';
                        contentEl.style.width = '';
                        contentEl.style.whiteSpace = '';
                    }
                    fullResponse += tokenText;
                    contentEl.textContent = fullResponse;
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }
            }

            this.currentAbortController = null;

            if (fullResponse.trim() && !this.stopRequested) {
                const cleanedResponse = this.trimIncompleteFinalSentence(fullResponse);

                let displayResponse = cleanedResponse;
                if (this.fileContentUsedInPrompt && this.config.fileUpload.fileName) {
                    displayResponse = cleanedResponse + `\n(Ref: ${this.config.fileUpload.fileName})`;
                }

                contentEl.innerHTML = this.formatResponse(displayResponse);

                this.conversationHistory.push({ role: 'user', content: this.getFirstSentence(userMessage) });
                this.conversationHistory.push({ role: 'assistant', content: this.getFirstSentence(cleanedResponse) });
            } else if (this.stopRequested && fullResponse.trim()) {
                let displayResponse = fullResponse;
                if (this.fileContentUsedInPrompt && this.config.fileUpload.fileName) {
                    displayResponse = fullResponse + `\n(Ref: ${this.config.fileUpload.fileName})`;
                }
                displayResponse += '\n\n[Response stopped by user - not saved to history]';
                contentEl.textContent = displayResponse;
                console.log('Stopped response not added to conversation history');
            } else if (!isGpuRetry && this.wllamaUsedGPU && !this.stopRequested) {
                await this.handleGpuFailoverAndRetry(messages, contentEl, userMessage, imageAnalysis);
                return;
            } else {
                contentEl.textContent = 'Sorry, I encountered an error while generating a response. Please try restarting the conversation. If this happens repeatedly, try switching to a different model.';
            }

        } catch (error) {
            if (this.stopRequested || error.name === 'AbortError' || error.message?.includes('abort')) {
                console.log('Generation aborted by user');
                if (fullResponse.trim()) {
                    let displayResponse = fullResponse + '\n\n[Response stopped by user - not saved to history]';
                    if (this.fileContentUsedInPrompt && this.config.fileUpload.fileName) {
                        displayResponse = fullResponse + `\n(Ref: ${this.config.fileUpload.fileName})\n\n[Response stopped - not saved to history]`;
                    }
                    contentEl.textContent = displayResponse;
                }
            } else if (!isGpuRetry && this.wllamaUsedGPU) {
                console.warn('GPU inference error, falling back to CPU');
                await this.handleGpuFailoverAndRetry(messages, contentEl, userMessage, imageAnalysis);
            } else {
                console.error('Error in wllama generation:', error);
                contentEl.textContent = 'Sorry, I encountered an error while generating a response. Please try restarting the conversation. If this happens repeatedly, try switching to a different model.';
            }
            this.currentAbortController = null;
        } finally {
            if (this.currentStream === completion) {
                this.currentStream = null;
            }
        }
    }



    async handleGpuFailoverAndRetry(messages, contentEl, userMessage, imageAnalysis) {
        console.warn('GPU inference failed; switching to CPU and retrying prompt...');
        this.gpuFailed = true;
        this.wllamaUsedGPU = false;
        this.wllamaLoaded = false;

        const deadWllama = this.wllama;
        this.wllama = null;
        deadWllama?.exit().catch(() => { });

        contentEl.style.width = 'fit-content';
        contentEl.style.whiteSpace = 'normal';
        contentEl.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>' +
            '<p style="font-size: 0.85em; color: #666; margin: 8px 0 0 0; font-style: italic;">I encountered a GPU issue. Switching to CPU mode and retrying — please wait...</p>';
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        try {
            await this.initializeWllama();
        } catch (reinitErr) {
            console.error('CPU re-initialisation failed after GPU crash:', reinitErr);
            contentEl.style.width = '';
            contentEl.style.whiteSpace = '';
            contentEl.textContent = 'Sorry, I encountered an error while generating a response. Please try restarting the conversation. If this happens repeatedly, try switching to a different model.';
            return;
        }

        if (!this.wllama || this.stopRequested) return;

        // Clear the failure notice and retry into the same bubble
        contentEl.textContent = '';
        contentEl.style.width = '';
        contentEl.style.whiteSpace = '';

        await this.handleWllamaMode(messages, null, userMessage, imageAnalysis, true, contentEl);
    }



    addThinkingIndicator() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message assistant-message thinking-indicator';
        thinkingDiv.innerHTML = `<div class="message-content" style="width: fit-content; white-space: normal"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><p style="font-size: 0.85em; color: #666; margin: 8px 0 0 0; font-style: italic;">(I'm working on a response. Thanks for your patience!)</p></div>`;
        this.chatMessages.appendChild(thinkingDiv);

        // Auto-scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        return thinkingDiv;
    }

    async typeResponseWithFinalHtml(contentEl, plainText, finalHtml) {
        let currentIndex = 0;
        const typingSpeed = 5;

        while (currentIndex < plainText.length && !this.stopRequested) {
            contentEl.textContent = plainText.substring(0, currentIndex + 1);
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            currentIndex++;
            await new Promise(resolve => setTimeout(resolve, typingSpeed));
        }

        // Set final HTML (with link) once typing is done
        contentEl.innerHTML = finalHtml;
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        this.isGenerating = false;
        this.updateUIForGeneration(false);
    }

    async typeResponse(contentEl, text) {
        let currentIndex = 0;
        const typingSpeed = 5; // milliseconds between characters

        // Continue typing as long as we haven't been stopped and there's more text
        while (currentIndex < text.length && !this.stopRequested) {
            contentEl.textContent = text.substring(0, currentIndex + 1);

            // Auto-scroll to bottom
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

            currentIndex++;
            await new Promise(resolve => setTimeout(resolve, typingSpeed));
        }

        // Ensure full text is displayed
        contentEl.textContent = text;

        // Mark typing as complete
        this.isGenerating = false;
        this.updateUIForGeneration(false);
    }

    startTypingAnimation(contentEl, initialText) {
        this.typingState = {
            contentEl: contentEl,
            fullText: initialText,
            currentIndex: 0,
            isTyping: true,
            typingSpeed: 5
        };

        this.continueTyping();
    }

    updateTypingContent(newText) {
        if (this.typingState) {
            this.typingState.fullText = newText;
        }
    }

    async continueTyping() {
        if (!this.typingState || !this.typingState.isTyping) return;

        const { contentEl, typingSpeed } = this.typingState;

        while (this.typingState.isTyping && !this.stopRequested) {
            // Use current fullText (which gets updated by streaming)
            const currentFullText = this.typingState.fullText;

            // Check if we've typed everything we currently have
            if (this.typingState.currentIndex >= currentFullText.length) {
                // Wait a bit for more content to arrive, but continue if we're not generating anymore
                if (!this.isGenerating) {
                    break; // No more content coming, we're done
                }
                await new Promise(resolve => setTimeout(resolve, 50)); // Wait for more content
                continue;
            }

            // Type the next character
            contentEl.textContent = currentFullText.substring(0, this.typingState.currentIndex + 1);

            // Auto-scroll to bottom
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

            this.typingState.currentIndex++;
            await new Promise(resolve => setTimeout(resolve, typingSpeed));
        }

        // Ensure full text is displayed
        if (this.typingState && this.typingState.contentEl) {
            this.typingState.contentEl.textContent = this.typingState.fullText;
        }

        // Mark typing as complete but don't update UI if still speaking
        if (this.typingState) {
            this.typingState.isTyping = false;
        }

        // Update UI
        this.updateUIForGeneration(false);
    }

    async waitForTypingComplete() {
        while (this.typingState && this.typingState.isTyping) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }
    }

    addMessage(role, content, imageElement = null) {
        // Hide welcome message only if NOT in voice mode
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage && !this.voiceMode) {
            welcomeMessage.style.display = 'none';
        }

        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}-message`;

        messageEl.innerHTML = `
            <div class="message-content">${escapeHtml(content)}</div>
        `;

        // Add image if provided
        if (imageElement && role === 'user') {
            const messageContent = messageEl.querySelector('.message-content');
            messageContent.insertBefore(imageElement, messageContent.firstChild);
        }

        this.chatMessages.appendChild(messageEl);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        return messageEl;
    }

    addTypingIndicator() {
        const typingEl = document.createElement('div');
        typingEl.className = 'message assistant-message';
        typingEl.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        this.chatMessages.appendChild(typingEl);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        return typingEl;
    }

    updateUIForGeneration(isGenerating) {
        this.userInput.disabled = isGenerating;

        if (isGenerating) {
            this.sendBtn.textContent = '■'; // Purple/black square
            this.sendBtn.style.color = '#6c3fa5'; // Purple color
            this.sendBtn.disabled = false;
            this.announceToScreenReader('AI is generating a response. Press the submit button to stop generation.');
        } else {
            this.sendBtn.textContent = '➤'; // Arrow
            this.sendBtn.style.color = '#6c3fa5';
            this.sendBtn.disabled = false;
            this.announceToScreenReader('Response generation completed.');
            // Return focus to input after response is complete
            this.userInput.focus();
        }
    }

    announceToScreenReader(message) {
        const announcer = document.getElementById('aria-announcer');
        if (announcer) {
            announcer.textContent = message;
            // Clear the message after a delay to allow screen reader to announce it
            setTimeout(() => {
                announcer.textContent = '';
            }, 1000);
        }
    }

    async stopGeneration() {
        this.stopRequested = true;

        // Stop typing animation
        if (this.typingState) {
            this.typingState.isTyping = false;
        }

        // Abort the generation using AbortController
        if (this.currentAbortController) {
            console.log('Aborting generation via AbortController');
            this.currentAbortController.abort();
            this.currentAbortController = null;
        }

        this.currentStream = null;

        this.updateUIForGeneration(false);
    }


    async restartConversation(reason = 'user-action') {
        // Clear the conversation history and reset the chat UI
        await this.clearChat();

        // Show a message to the user about the restart
        const restartMessage = 'Conversation restarted.';
        const systemMessageEl = this.addMessage('system', restartMessage);
        systemMessageEl.classList.add('system-restart-message');
    }

    async clearChat() {
        this.conversationHistory = [];

        // Show appropriate welcome message based on mode
        if (this.voiceMode) {
            this.chatMessages.innerHTML = `
                <div class="welcome-message" id="voice-welcome" style="display: flex;">
                    <div class="voice-chat-icon" aria-hidden="true">
                        <img class="avatar-image" id="voice-avatar-image" style="display: none;" alt="Avatar">
                    </div>
                    <h3>Let's talk</h3>
                    <p>Talk like you would to a person. The agent listens and responds.</p>
                </div>
            `;
            // Update avatar display after creating the HTML
            setTimeout(() => this.updateAvatarDisplay(), 0);
        } else {
            this.chatMessages.innerHTML = `
                <div class="welcome-message" id="text-welcome">
                    <div class="chat-icon">💬</div>
                    <h3>What do you want to chat about?</h3>
                </div>
            `;
        }
    }

    // Removed updateTokenCount function - disclaimer is now static

    // ========== Speech and Voice Functions ==========

    loadVoiceHealthCache() {
        try {
            const rawCache = localStorage.getItem(this.voiceHealthCacheKey);
            this.voiceHealthCache = rawCache ? JSON.parse(rawCache) : {};
        } catch (error) {
            console.warn('Failed to load voice health cache:', error);
            this.voiceHealthCache = {};
        }
    }

    saveVoiceHealthCache() {
        try {
            localStorage.setItem(this.voiceHealthCacheKey, JSON.stringify(this.voiceHealthCache));
        } catch (error) {
            console.warn('Failed to save voice health cache:', error);
        }
    }

    getVoiceHealthKey(voice) {
        return `${voice.name}::${voice.lang}`;
    }

    isVoiceHealthEntryFresh(entry) {
        if (!entry || !entry.checkedAt) return false;
        return (Date.now() - entry.checkedAt) <= this.voiceHealthCacheTtlMs;
    }

    updateVoiceHealthStatus(voice, ok) {
        if (!voice || !voice.name) return;
        const key = this.getVoiceHealthKey(voice);
        this.voiceHealthCache[key] = {
            ok: !!ok,
            checkedAt: Date.now()
        };
        this.saveVoiceHealthCache();
    }

    shouldMarkVoiceAsFailed(errorCode = '') {
        const code = String(errorCode || '').toLowerCase();
        return !['interrupted', 'canceled', 'cancelled'].includes(code);
    }

    getDisplayableVoices(voices) {
        return voices.filter((voice) => {
            const cacheEntry = this.voiceHealthCache[this.getVoiceHealthKey(voice)];
            if (!cacheEntry || !this.isVoiceHealthEntryFresh(cacheEntry)) {
                return true;
            }
            return cacheEntry.ok;
        });
    }

    async probeVoiceAvailability(voice, timeoutMs = 4000, maxAttempts = 2) {
        if (!voice || !('speechSynthesis' in window)) {
            return 'inconclusive';
        }

        let sawInconclusiveError = false;

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const result = await new Promise((resolve) => {
                let settled = false;
                let timedOut = false;
                const utterance = new SpeechSynthesisUtterance('Voice check test');
                utterance.voice = voice;
                utterance.volume = 0;
                utterance.rate = 1;
                utterance.pitch = 1;

                const finish = (ok, errorCode = '') => {
                    if (settled) return;
                    settled = true;
                    clearTimeout(timeoutHandle);
                    utterance.onend = null;
                    utterance.onerror = null;
                    resolve({ ok, errorCode, timedOut });
                };

                utterance.onend = () => finish(true, '');
                utterance.onerror = (event) => {
                    const code = event && event.error ? event.error : 'unknown';
                    finish(false, code);
                };

                const timeoutHandle = setTimeout(() => {
                    timedOut = true;
                    try {
                        speechSynthesis.cancel();
                    } catch (error) {
                        console.warn('Voice probe cancel failed:', error);
                    }
                    finish(false, 'timeout');
                }, timeoutMs);

                try {
                    speechSynthesis.cancel();
                    speechSynthesis.speak(utterance);
                } catch (error) {
                    console.warn('Voice probe failed to start:', error);
                    finish(false, 'start-failed');
                }
            });

            if (result.ok) {
                return 'ok';
            }

            if (!this.shouldMarkVoiceAsFailed(result.errorCode) && !result.timedOut) {
                sawInconclusiveError = true;
                continue;
            }

            if (attempt < maxAttempts) {
                continue;
            }

            return 'failed';
        }

        return sawInconclusiveError ? 'inconclusive' : 'failed';
    }

    runVoiceHealthChecks(allEnglishVoices) {
        if (!Array.isArray(allEnglishVoices) || allEnglishVoices.length === 0) {
            return;
        }

        if (this.voiceHealthCheckPromise) {
            return;
        }

        if (this.isListening || this.isSpeaking || this.isGenerating) {
            return;
        }

        const voicesToProbe = allEnglishVoices.filter((voice) => {
            const cacheEntry = this.voiceHealthCache[this.getVoiceHealthKey(voice)];
            return !cacheEntry || !this.isVoiceHealthEntryFresh(cacheEntry);
        });

        if (voicesToProbe.length === 0) {
            return;
        }

        this.voiceHealthCheckPromise = (async () => {
            for (const voice of voicesToProbe) {
                const isHealthy = await this.probeVoiceAvailability(voice);
                this.updateVoiceHealthStatus(voice, isHealthy);
            }
        })().finally(() => {
            this.voiceHealthCheckPromise = null;
            this.populateVoices();
        });
    }

    async handleVoiceSelectionChange(selectedVoiceName) {
        this.speechSettings.voice = selectedVoiceName;
        console.log('Voice selected:', selectedVoiceName);

        if (!selectedVoiceName || selectedVoiceName === 'none') {
            return;
        }

        const voices = speechSynthesis.getVoices();
        const selectedVoice = voices.find((voice) => voice.name === selectedVoiceName);
        if (!selectedVoice) {
            this.showToast('Selected voice is no longer available.');
            this.populateVoices();
            return;
        }

        const currentNonce = ++this.voiceSelectionProbeNonce;
        const probePromise = this.probeVoiceAvailability(selectedVoice);
        this.voiceSelectionProbePromise = probePromise;
        const probeResult = await probePromise;
        if (this.voiceSelectionProbePromise === probePromise) {
            this.voiceSelectionProbePromise = null;
        }

        if (currentNonce !== this.voiceSelectionProbeNonce) {
            return;
        }

        if (probeResult === 'ok') {
            this.updateVoiceHealthStatus(selectedVoice, true);
            this.populateVoices();
            return;
        }

        if (probeResult === 'failed') {
            this.updateVoiceHealthStatus(selectedVoice, false);
            this.populateVoices();
            this.showToast('That voice is not available right now. Switched to a working voice.');
            return;
        }

        this.showToast('Voice check was inconclusive. Keeping selected voice.');
        this.populateVoices();
    }

    async waitForSpeechIdle(timeoutMs = 2000) {
        const start = Date.now();
        while (Date.now() - start < timeoutMs) {
            if (!speechSynthesis.speaking && !speechSynthesis.pending) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        return !speechSynthesis.speaking && !speechSynthesis.pending;
    }

    async playPreviewAttempt(selectedVoice) {
        return await new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance('This is my voice.');
            utterance.voice = selectedVoice;
            utterance.rate = 1;

            utterance.onerror = (event) => {
                resolve({ ok: false, errorCode: event?.error || 'unknown' });
            };

            utterance.onend = () => {
                resolve({ ok: true, errorCode: '' });
            };

            try {
                speechSynthesis.speak(utterance);
            } catch (error) {
                resolve({ ok: false, errorCode: 'start-failed' });
            }
        });
    }

    populateVoices() {
        const voiceSelect = document.getElementById('config-voice-select');
        if (!voiceSelect) return;

        const loadVoices = () => {
            const voices = speechSynthesis.getVoices();
            // Get all English voices
            const englishVoices = voices.filter(voice => voice && voice.lang && voice.lang.startsWith('en'));
            const displayVoices = this.getDisplayableVoices(englishVoices);

            // Preserve currently selected voice
            const currentlySelectedVoice = this.speechSettings.voice || voiceSelect.value;

            voiceSelect.innerHTML = '';

            if (displayVoices.length > 0) {
                this.voicesAvailable = true;
                displayVoices.forEach((voice) => {
                    if (!voice || !voice.name) return;
                    const option = document.createElement('option');
                    option.value = voice.name;
                    const localLabel = voice.localService ? ' (Local)' : '';
                    option.textContent = `${voice.name} (${voice.lang})${localLabel}`;
                    voiceSelect.appendChild(option);
                });

                // Restore previously selected voice or select the first one
                if (currentlySelectedVoice && displayVoices.find(v => v.name === currentlySelectedVoice)) {
                    voiceSelect.value = currentlySelectedVoice;
                    this.speechSettings.voice = currentlySelectedVoice;
                } else if (displayVoices.length > 0) {
                    voiceSelect.value = displayVoices[0].name;
                    this.speechSettings.voice = displayVoices[0].name;
                }

                // Enable voice select when voice mode is on
                if (this.voiceMode) {
                    voiceSelect.disabled = false;
                }
            } else {
                this.voicesAvailable = false;
                const option = document.createElement('option');
                option.value = 'none';
                option.textContent = 'No working voices available';
                voiceSelect.appendChild(option);
                voiceSelect.disabled = true;
                this.speechSettings.voice = null;
            }

            this.voicesLoaded = true;
            // Do not run passive background probing here: some engines require direct
            // user interaction and can falsely fail cloud voices when tested passively.
        };

        if (speechSynthesis.getVoices().length > 0) {
            loadVoices();
        } else {
            if (!this._voicesChangedListenerAdded) {
                this._voicesChangedListenerAdded = true;
                speechSynthesis.addEventListener('voiceschanged', () => {
                    this._voicesChangedListenerAdded = false;
                    loadVoices();
                }, { once: true });
            }
            setTimeout(loadVoices, 100);
        }
    }

    initializeSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        this.recognition.maxAlternatives = 1;

        this.recognition.onstart = () => {
            console.log('Web Speech Recognition started');
            this.isListening = true;
            // Clear the starting flag once we've successfully started
            this.isStartingRecognition = false;
        };

        this.recognition.onresult = (event) => {
            const result = event.results[0];
            if (result.isFinal) {
                const transcript = result[0].transcript.trim();
                if (transcript) {
                    this.handleSpokenInput(transcript);
                }
            }
        };

        this.recognition.onend = () => {
            console.log('Web Speech Recognition ended');
            this.isListening = false;
            // Ensure starting flag is cleared
            this.isStartingRecognition = false;
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;

            // Ignore 'aborted' errors when we're starting recognition - cleanup abort is expected
            if (event.error === 'aborted' && this.isStartingRecognition) {
                console.log('Ignoring abort error during startup cleanup');
                return;
            }

            // Always clear the starting flag on any error
            this.isStartingRecognition = false;

            // Ignore 'aborted' errors - these are expected when stopping recognition
            if (event.error === 'aborted') {
                this.resetVoiceUI();
                return;
            }

            // If user explicitly cancelled voice interaction, don't do anything
            if (this.voiceInteractionCancelled) {
                this.resetVoiceUI();
                return;
            }

            // Ignore 'no-speech' errors - these are normal when user doesn't speak
            if (event.error === 'no-speech') {
                console.log('No speech detected, stopping voice recognition');
                this.resetVoiceUI();
                return;
            }

            // In voice mode, try Vosk failover only for network/service errors
            // Don't failover for audio-capture, language-not-supported, or other benign errors
            if (this.voiceMode && (event.error === 'network' || event.error === 'service-not-allowed')) {
                this.handleSpeechRecognitionFailure(event.error);
                return;
            }

            this.resetVoiceUI();

            // Show appropriate error message
            if (this.voiceMode && event.error !== 'not-allowed') {
                this.openVoiceInputErrorModal(event.error);
            } else if (event.error !== 'not-allowed') {
                this.showToast(ChatPlayground.MESSAGES.ERRORS.SPEECH_ERROR);
            }
        };
    }

    async handleSpeechRecognitionFailure(errorCode) {
        // Permission denied - show error modal, don't fallback
        if (errorCode === 'not-allowed' || errorCode === 'service-not-allowed') {
            this.resetVoiceUI();
            this.openVoiceInputErrorModal(errorCode);
            return;
        }

        // Try Vosk failover - STOP current interaction and switch to Vosk mode
        console.log('Web Speech API failed, switching to Vosk mode...');
        this.usingWebSpeech = false;

        // Load Vosk model if not already loaded
        if (!this.voskLoaded && !this.voskLoadingFailed) {
            await this.loadVoskModel();
            // Modal is now showing with Cancel/Retry buttons
            // User will decide whether to retry or cancel
        } else if (this.voskLoaded) {
            // Vosk already loaded from a previous failover
            // Show a simple message and let them retry manually
            this.resetVoiceUI();
            this.showToast('Switched to offline speech recognition. Click Start Session to continue.');
        } else {
            // Vosk failed to load previously
            this.resetVoiceUI();
            this.openVoiceInputErrorModal(errorCode);
        }
    }

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

            // Show modal
            this.openSpeechModelModal();

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

                    // Append the recognized text to a temporary buffer
                    if (!this.voskTranscript) {
                        this.voskTranscript = '';
                    }
                    this.voskTranscript += (this.voskTranscript ? " " : "") + result.text;
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

                    // Mark that we have speech so auto-stop can work
                    this.hasSpeech = true;
                    // Store the partial transcript as fallback
                    this.voskPartialTranscript = result.partial;
                    this.lastSpeechTime = Date.now();
                    this.resetSilenceTimer();
                }
            });

            this.voskLoaded = true;
            console.log('Vosk speech model loaded successfully');

            // Update modal to show ready state
            this.updateSpeechModelModal('Offline speech model ready!', true);

            return true;
        } catch (error) {
            console.error('Failed to load Vosk model:', error);
            this.voskLoadingFailed = true;
            this.updateSpeechModelModal('Failed to load offline speech model. Voice input is unavailable.', false);
            return false;
        }
    }

    openSpeechModelModal() {
        const modal = document.getElementById('speech-model-modal');
        const status = document.getElementById('speech-model-status');
        const progress = document.getElementById('speech-model-progress');
        const retryBtn = document.getElementById('speech-model-retry');

        if (!modal) return;

        if (status) {
            status.textContent = 'Loading offline speech model... This may take a moment.';
        }

        if (progress) {
            progress.style.width = '50%'; // Indeterminate progress
        }

        if (retryBtn) {
            retryBtn.disabled = true;
        }

        modal.style.display = 'flex';
    }

    updateSpeechModelModal(message, enableRetry) {
        const status = document.getElementById('speech-model-status');
        const progress = document.getElementById('speech-model-progress');
        const retryBtn = document.getElementById('speech-model-retry');

        if (status) {
            status.textContent = message;
        }

        if (progress) {
            progress.style.width = enableRetry ? '100%' : '0%';
        }

        if (retryBtn) {
            retryBtn.disabled = !enableRetry;
        }
    }

    closeSpeechModelModal() {
        const modal = document.getElementById('speech-model-modal');
        if (!modal) return;
        modal.style.display = 'none';
    }

    cancelSpeechModelLoading() {
        this.closeSpeechModelModal();
        // Return to ready state - user can click Start Session when ready
        this.resetVoiceUI();
    }

    async retrySpeechInput() {
        this.closeSpeechModelModal();



        openVoiceInputErrorModal(errorCode = '') {
            const modal = document.getElementById('voice-input-error-modal');
            const input = document.getElementById('voice-fallback-input');
            const submitBtn = document.getElementById('voice-input-error-submit');
            const description = modal ? modal.querySelector('.voice-fallback-description') : null;

            if (!modal || !input) {
                this.showToast(ChatPlayground.MESSAGES.ERRORS.SPEECH_ERROR);
                return;
            }

            if (description) {
                const details = errorCode ? ` (${errorCode})` : '';
                description.textContent = `A speech recognition error${details} prevented voice input for this turn. You can type your message below and continue the conversation.`;
            }

            input.value = '';
            if (submitBtn) {
                submitBtn.disabled = true;
            }

            modal.style.display = 'flex';

            setTimeout(() => {
                input.focus();
            }, 50);
        }

        closeVoiceInputErrorModal() {
            const modal = document.getElementById('voice-input-error-modal');
            if (!modal) return;

            modal.style.display = 'none';
        }

        submitTypedVoiceFallback() {
            const input = document.getElementById('voice-fallback-input');
            if (!input) return;

            const typedPrompt = input.value.trim();
            if (!typedPrompt) {
                return;
            }

            this.closeVoiceInputErrorModal();
            this.handleSpokenInput(typedPrompt);
        }

    async startVoiceInput() {
            // If already listening, stop
            if (this.isListening || this.isRecording) {
                this.stopSpeechRecognition(true);
                return;
            }

            // Clear conversation history when starting a new session
            await this.clearChat();

            // Reset cancelled flag
            this.voiceInteractionCancelled = false;

        } else {
            await this.startVoskRecognition();
        }
    }

    setupListeningUI() {
        const startBtn = document.getElementById('voice-start-btn');
        const cancelBtn = document.getElementById('voice-cancel-btn');
        const ccBtn = document.getElementById('voice-cc-btn');
        const chatIcon = document.querySelector('.voice-chat-icon');
        const voiceWelcome = document.getElementById('voice-welcome');

        if (startBtn) {
            startBtn.style.display = 'none';
        }
        if (cancelBtn) {
            cancelBtn.style.display = 'inline-block';
        }
        if (ccBtn) {
            ccBtn.style.display = 'inline-block'; // Show CC button throughout the session
        }
        if (chatIcon) {
            chatIcon.style.animation = 'pulse 1s infinite'; // Pulse while listening
        }

        // Keep captions hidden unless user explicitly enables them
        this.updateMessageVisibility();

        // Update welcome message
        if (voiceWelcome) {
            voiceWelcome.querySelector('h3').textContent = 'Listening...';
            voiceWelcome.querySelector('p').textContent = 'Speak now...';
        }
    }

    async startWebSpeechRecognition() {
        if (!this.recognition) {
            this.showToast(ChatPlayground.MESSAGES.ERRORS.SPEECH_NOT_AVAILABLE);
            this.resetVoiceUI();
            return;
        }

        try {
            // Set flag to ignore abort errors during startup
            this.isStartingRecognition = true;
            console.log('Starting Web Speech Recognition...');

            try {
                this.recognition.abort();
            } catch (e) {
                console.warn('Error calling recognition.abort():', e);
            }

            setTimeout(() => {
                try {
                    console.log('Calling recognition.start()');
                    this.recognition.start();
                    // Set a timeout to clear the flag if onstart doesn't fire
                    setTimeout(() => {
                        if (this.isStartingRecognition) {
                            console.warn('Recognition start timeout - clearing flag');
                            this.isStartingRecognition = false;
                            // If we're still not listening after timeout, something went wrong
                            if (!this.isListening) {
                                console.error('Recognition failed to start');
                                this.resetVoiceUI();
                                this.showToast(ChatPlayground.MESSAGES.ERRORS.VOICE_INPUT_FAILED);
                            }
                        }
                    }, 2000);
                } catch (error) {
                    this.isStartingRecognition = false;
                    console.error('Error calling recognition.start():', error);
                    this.isListening = false;
                    this.showToast(ChatPlayground.MESSAGES.ERRORS.VOICE_INPUT_FAILED);
                    this.resetVoiceUI();
                }
            }, 100);
        } catch (error) {
            this.isStartingRecognition = false;
            console.error('Error starting speech recognition:', error);
            this.isListening = false;
            this.showToast(ChatPlayground.MESSAGES.ERRORS.VOICE_INPUT_FAILED);
            this.resetVoiceUI();
        }
    }

    async startVoskRecognition() {
        // Ensure Vosk is loaded
        if (!this.voskLoaded) {
            if (!this.voskLoadingFailed) {
                const loaded = await this.loadVoskModel();
                if (!loaded) {
                    this.resetVoiceUI();
                    this.showToast('Voice input is unavailable.');
                    return;
                }
            } else {
                this.resetVoiceUI();
                this.showToast('Voice input is unavailable.');
                return;
            }
        }

        if (!this.voskRecognizer) {
            this.showToast('Speech input is not available.');
            this.resetVoiceUI();
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
            this.hasSpeech = false;
            this.voskTranscript = '';
            this.voskPartialTranscript = '';

            // Start no-speech timeout
            this.noSpeechTimer = setTimeout(() => {
                if (this.isRecording && !this.hasSpeech) {
                    console.log('No speech detected in 5 seconds, cancelling...');
                    this.stopSpeechRecognition(true);
                    this.showToast('No speech detected. Please try again.');
                }
            }, this.noSpeechTimeoutDuration);

            // Start silence timer
            this.resetSilenceTimer();

        } catch (error) {
            console.error('Error starting Vosk recording:', error);
            this.isRecording = false;
            this.resetVoiceUI();

            if (error.name === 'NotAllowedError') {
                this.showToast('Microphone access was denied.');
            } else {
                this.showToast('Error accessing microphone.');
            }
        }
    }

    stopSpeechRecognition(cancelled = false) {
        // Stop Web Speech API if active
        if (this.isListening && this.recognition) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.error('Error stopping Web Speech recognition:', error);
            }
            this.isListening = false;
        }

        // Stop Vosk if active
        if (this.isRecording) {
            this.stopVoskRecording(cancelled);
        }

        // If cancelled and no transcript processing will happen, reset UI
        if (cancelled && !this.isRecording) {
            this.resetVoiceUI();
        }
    }

    stopVoskRecording(cancelled = false) {
        this.isRecording = false;

        console.log('stopVoskRecording called:', { cancelled, hasSpeech: this.hasSpeech, transcript: this.voskTranscript, partial: this.voskPartialTranscript });

        // Clear timers
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        if (this.noSpeechTimer) {
            clearTimeout(this.noSpeechTimer);
            this.noSpeechTimer = null;
        }

        // Disconnect audio nodes first
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Process transcript if we have speech and not cancelled
        if (!cancelled && this.hasSpeech) {
            // Use final transcript if available, otherwise use partial as fallback
            let transcript = this.voskTranscript.trim();
            if (!transcript && this.voskPartialTranscript) {
                transcript = this.voskPartialTranscript.trim();
            }

            this.voskTranscript = '';
            this.voskPartialTranscript = '';

            if (transcript) {
                console.log('Processing transcript:', transcript);

                // Close audio context AFTER we have the transcript, 
                // with a small delay to ensure audio cleanup doesn't interfere
                setTimeout(() => {
                    if (this.audioContext) {
                        this.audioContext.close();
                        this.audioContext = null;
                    }
                }, 100);

                this.handleSpokenInput(transcript);
                return; // Don't reset UI yet, handleSpokenInput will handle it
            }
        }

        // Close audio context if we're not processing transcript
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        // Reset UI if cancelled or no speech
        this.resetVoiceUI();
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
                    this.stopSpeechRecognition(false);
                }
            }, this.silenceTimeout);
        }
    }

    handleSpokenInput(transcript) {
        console.log('handleSpokenInput called:', transcript);

        // Validate and sanitize transcript
        if (!transcript || typeof transcript !== 'string') {
            console.error('Invalid transcript received');
            this.resetVoiceUI();
            return;
        }

        let sanitizedTranscript = transcript.trim();
        if (sanitizedTranscript.length > 1000) {
            sanitizedTranscript = sanitizedTranscript.substring(0, 1000);
        }

        if (sanitizedTranscript.length === 0) {
            console.error('Transcript empty after sanitization');
            this.resetVoiceUI();
            return;
        }

        // Check for prohibited content
        if (this.containsProhibitedContent(sanitizedTranscript)) {
            // Add user message to chat (respecting current CC visibility)
            const userMessage = this.addMessage('user', sanitizedTranscript);
            if (userMessage && !this.showCaptions) {
                userMessage.classList.add('hidden');
            }

            // Add canned response
            const assistantMessage = this.addMessage('assistant', ChatPlayground.MESSAGES.MODERATION.BLOCKED);
            if (assistantMessage && !this.showCaptions) {
                assistantMessage.classList.add('hidden');
            }

            // Update UI for speaking
            const voiceWelcome = document.getElementById('voice-welcome');
            const chatIcon = document.querySelector('.voice-chat-icon');
            const cancelBtn = document.getElementById('voice-cancel-btn');
            const ccBtn = document.getElementById('voice-cc-btn');

            if (cancelBtn) {
                cancelBtn.style.display = 'inline-block';
            }
            if (ccBtn) {
                ccBtn.style.display = 'inline-block';
            }
            if (voiceWelcome) {
                voiceWelcome.querySelector('h3').textContent = 'Speaking';
                voiceWelcome.querySelector('p').textContent = 'Adjust volume as necessary.';
            }
            if (chatIcon) {
                chatIcon.style.animation = 'pulse 1s infinite';
            }

            // Speak the canned response
            this.speakResponse(ChatPlayground.MESSAGES.MODERATION.BLOCKED);

            return;
        }

        // Update UI - show processing state
        const startBtn = document.getElementById('voice-start-btn');
        const cancelBtn = document.getElementById('voice-cancel-btn');
        const ccBtn = document.getElementById('voice-cc-btn');
        const voiceWelcome = document.getElementById('voice-welcome');

        if (startBtn) {
            startBtn.style.display = 'none';
        }
        if (cancelBtn) {
            cancelBtn.style.display = 'inline-block';
        }
        if (ccBtn) {
            ccBtn.style.display = 'inline-block'; // Show CC button now
        }
        // Keep pulse animation running during processing - don't stop it
        if (voiceWelcome) {
            voiceWelcome.querySelector('h3').textContent = 'Processing...';
            voiceWelcome.querySelector('p').textContent = 'This can take some time...';
        }

        // Add user message to chat (respecting current CC visibility)
        const userMessage = this.addMessage('user', sanitizedTranscript);
        if (userMessage && !this.showCaptions) {
            userMessage.classList.add('hidden');
        }

        // Web search intercept (voice mode)
        if (this.isWebSearchActive() && /^(find|search)\b/i.test(sanitizedTranscript)) {
            const keywords = this.extractWebSearchKeywords(sanitizedTranscript);
            const plainText = keywords
                ? "OK, I searched the web for you. Here's what I found."
                : 'Please enter a more specific search query.';
            const assistantMsgEl = this.addMessage('assistant', '');
            if (keywords) {
                const url = `https://www.bing.com/search?q=${encodeURIComponent(keywords)}`;
                assistantMsgEl.querySelector('.message-content').innerHTML =
                    `OK, I searched the web for you. <a href="${url}" target="_blank" rel="noopener noreferrer">Here's what I found.</a>`;
            } else {
                assistantMsgEl.querySelector('.message-content').textContent = plainText;
            }
            if (assistantMsgEl && !this.showCaptions) assistantMsgEl.classList.add('hidden');
            const voiceWelcome = document.getElementById('voice-welcome');
            const chatIcon = document.querySelector('.voice-chat-icon');
            if (voiceWelcome) {
                voiceWelcome.querySelector('h3').textContent = 'Speaking';
                voiceWelcome.querySelector('p').textContent = 'Adjust volume as necessary.';
            }
            if (chatIcon) chatIcon.style.animation = 'pulse 1s infinite';
            this.conversationHistory.push(
                { role: 'user', content: this.getFirstSentence(sanitizedTranscript) },
                { role: 'assistant', content: this.getFirstSentence(plainText) }
            );
            this.speakResponse(plainText);
            return;
        }

        // Vocalize acknowledgment while processing model response (skip for Wikipedia mode - too fast)
        if (this.speechSettings.textToSpeech && this.voicesAvailable && 'speechSynthesis' in window &&
            !this.usingWikipedia && this.currentMode !== 'none') {
            const acknowledgment = new SpeechSynthesisUtterance("OK, let me think about that...");

            // Use the selected voice if available
            if (this.speechSettings.voice && this.speechSettings.voice !== 'default') {
                const voices = speechSynthesis.getVoices();
                const selectedVoice = voices.find(voice => voice.name === this.speechSettings.voice);
                if (selectedVoice) {
                    acknowledgment.voice = selectedVoice;
                }
            }

            // Speak the acknowledgment without blocking - it will play while model processes
            speechSynthesis.speak(acknowledgment);
        }

        // Generate response
        this.generateVoiceResponse(sanitizedTranscript);
    }

    async generateVoiceResponse(userMessage) {
        console.log('generateVoiceResponse started for:', userMessage);
        this.isGenerating = true;

        // Append instruction for concise response in voice mode
        const voiceModeUserMessage = userMessage + '\nRespond with a single, concise paragraph.';

        try {
            let responseText = '';

            // Route to the appropriate engine based on current mode (same as text mode)
            if (this.currentMode === 'none' || this.usingWikipedia) {
                // Use Wikipedia/None mode
                console.log('Using Wikipedia/None mode for response generation');
                this.fileContentUsedInPrompt = false;
                responseText = await this.generateNoneModeResponse(userMessage);
                responseText = this.maybeShortenNoneModeResponse(responseText);
                responseText = this.maybeMutateNoneModeResponse(responseText);

                // Append file attribution if file content was used
                if (this.fileContentUsedInPrompt && this.config.fileUpload.fileName && responseText.trim()) {
                    responseText = responseText + `\n(Ref: ${this.config.fileUpload.fileName})`;
                }
            } else if (this.usingWllama && this.wllama) {
                // Use wllama (Phi 3.5-mini)
                console.log('Using Wllama for voice response generation');
                const voiceMessages = [
                    { role: 'system', content: this.currentSystemMessage + '\nKeep responses short and succinct.' },
                    ...this.conversationHistory.slice(-2),
                    { role: 'user', content: voiceModeUserMessage }
                ];

                const voiceCompletionParams = {
                    messages: voiceMessages,
                    max_tokens: Math.min(this.config.modelParameters.max_tokens, 250),
                    temperature: this.config.modelParameters.temperature,
                    top_k: 30,
                    top_p: this.config.modelParameters.top_p,
                    repeat_penalty: this.config.modelParameters.repetition_penalty,
                    repeat_last_n: 64,
                    stream: false
                };

                this.currentAbortController = new AbortController();
                const completion = await this.wllama.createChatCompletion({
                    ...voiceCompletionParams,
                    abortSignal: this.currentAbortController.signal
                });
                this.currentAbortController = null;
                responseText = completion.choices?.[0]?.message?.content?.trim() ?? '';
                console.log('Wllama voice completion finished, response length:', responseText.length);

                // GPU failure recovery: if GPU was used and the response is empty, tear
                // down and retry once on CPU.
                if (!responseText && this.wllamaUsedGPU && !this.stopRequested) {
                    console.warn('GPU voice inference produced no output; switching to CPU and retrying...');
                    this.gpuFailed = true;
                    this.wllamaUsedGPU = false;
                    this.wllamaLoaded = false;
                    const deadWllama = this.wllama;
                    this.wllama = null;
                    deadWllama?.exit().catch(() => { });
                    try {
                        await this.initializeWllama();
                        if (this.wllama && !this.stopRequested) {
                            this.currentAbortController = new AbortController();
                            const retryCompletion = await this.wllama.createChatCompletion({
                                ...voiceCompletionParams,
                                abortSignal: this.currentAbortController.signal
                            });
                            this.currentAbortController = null;
                            responseText = retryCompletion.choices?.[0]?.message?.content?.trim() ?? '';
                            console.log('CPU voice retry finished, response length:', responseText.length);
                        }
                    } catch (reinitErr) {
                        console.error('CPU re-init failed after GPU voice crash:', reinitErr);
                        this.currentAbortController = null;
                    }
                }
            } else {
                responseText = "No AI model is currently available. Please wait for the model to load.";
            }

            console.log('Response generation complete, length:', responseText.length);

            // Add assistant message to chat (respecting current CC visibility)
            const assistantMessage = this.addMessage('assistant', responseText);
            if (assistantMessage && !this.showCaptions) {
                assistantMessage.classList.add('hidden');
            }

            console.log('Updating UI to Speaking state...');

            // Update UI to "Speaking" state
            const voiceWelcome = document.getElementById('voice-welcome');
            const chatIcon = document.querySelector('.voice-chat-icon');

            if (voiceWelcome) {
                voiceWelcome.querySelector('h3').textContent = 'Speaking';
                voiceWelcome.querySelector('p').textContent = 'Adjust volume as necessary.';
                console.log('Updated voiceWelcome to Speaking');
            }

            // Start pulsing animation while speaking
            if (chatIcon) {
                chatIcon.style.animation = 'pulse 1s infinite';
            }

            console.log('About to call speakResponse...');

            // Speak the response
            this.speakResponse(responseText);

            // Add to conversation history
            this.conversationHistory.push(
                { role: 'user', content: this.getFirstSentence(userMessage) },
                { role: 'assistant', content: this.getFirstSentence(responseText) }
            );

            console.log('generateVoiceResponse completed successfully');
        } catch (error) {
            console.error('Error generating response:', error);
            console.error('Error stack:', error.stack);
            this.showToast('Error generating response. Please try again or switch models.');
            this.resetVoiceUI();
        } finally {
            this.isGenerating = false;
            console.log('generateVoiceResponse finally block, isGenerating set to false');
        }
    }

    speakResponse(text) {
        console.log('speakResponse called:', { text: text.substring(0, 100) + '...', textToSpeech: this.speechSettings.textToSpeech, voicesAvailable: this.voicesAvailable });

        if (!this.speechSettings.textToSpeech || !this.voicesAvailable) {
            console.log('TTS disabled or voices unavailable, skipping speech');
            this.onSpeechComplete();
            return;
        }

        if (!('speechSynthesis' in window)) {
            console.log('speechSynthesis not available');
            this.onSpeechComplete();
            return;
        }

        // Cancel any ongoing speech and wait a bit for cleanup
        speechSynthesis.cancel();

        // Small delay to let speechSynthesis cleanup complete
        setTimeout(() => {
            const utterance = new SpeechSynthesisUtterance(text);
            let selectedVoice = null;

            if (this.speechSettings.voice && this.speechSettings.voice !== 'default') {
                const voices = speechSynthesis.getVoices();
                selectedVoice = voices.find(voice => voice.name === this.speechSettings.voice);
                if (selectedVoice) {
                    utterance.voice = selectedVoice;
                }
            }

            utterance.rate = 1;
            utterance.pitch = 1;
            utterance.volume = 1;

            this.isSpeaking = true;

            // Safety timeout in case onend never fires (estimate based on text length)
            const estimatedDuration = (text.length / 15) * 1000 + 5000; // ~15 chars per second + 5 sec buffer
            const safetyTimeout = setTimeout(() => {
                console.warn('Speech synthesis timeout - forcing completion');
                if (this.isSpeaking) {
                    speechSynthesis.cancel();
                    this.isSpeaking = false;
                    this.onSpeechComplete();
                }
            }, estimatedDuration);

            utterance.onstart = () => {
                console.log('TTS utterance started');
            };

            utterance.onend = () => {
                console.log('TTS utterance ended normally');
                clearTimeout(safetyTimeout);
                this.isSpeaking = false;
                this.onSpeechComplete();
            };

            utterance.onerror = (event) => {
                console.error('TTS utterance error:', event);
                clearTimeout(safetyTimeout);
                if (selectedVoice && this.shouldMarkVoiceAsFailed(event?.error)) {
                    this.updateVoiceHealthStatus(selectedVoice, false);
                    this.populateVoices();
                }
                this.isSpeaking = false;
                this.onSpeechComplete();
            };

            console.log('Speaking utterance...');
            speechSynthesis.speak(utterance);
        }, 50);
    }

    onSpeechComplete() {
        // If user cancelled, reset to default state
        if (this.voiceInteractionCancelled) {
            this.resetVoiceUI();
            return;
        }


        const startBtn = document.getElementById('voice-start-btn');
        const cancelBtn = document.getElementById('voice-cancel-btn');
        const ccBtn = document.getElementById('voice-cc-btn');
        const voiceWelcome = document.getElementById('voice-welcome');

        if (chatIcon) {
            chatIcon.style.animation = 'none';
        }
        if (startBtn) {
            startBtn.style.display = 'inline-block';
            startBtn.disabled = false;
        }
        if (cancelBtn) {
            cancelBtn.style.display = 'none';
        }
        if (ccBtn) {
            ccBtn.style.display = 'none';
        }

        // Reset captions state to off for next conversation
        this.showCaptions = false;
        this.updateCCButton();

        if (voiceWelcome) {
            voiceWelcome.querySelector('h3').textContent = "Let's talk";
            voiceWelcome.querySelector('p').textContent = 'Talk like you would to a person. The agent listens and responds.';
        }
    }

    toggleCaptions() {
        this.showCaptions = !this.showCaptions;
        this.updateCCButton();
        this.updateMessageVisibility();
    }

    updateCCButton() {
        const ccBtn = document.getElementById('voice-cc-btn');
        if (!ccBtn) return;

        if (this.showCaptions) {
            ccBtn.innerHTML = '[<s>cc</s>]';
        } else {
            ccBtn.innerHTML = '[cc]';
        }
    }

    updateMessageVisibility() {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        const messages = chatMessages.querySelectorAll('.message');
        messages.forEach(msg => {
            if (this.showCaptions) {
                msg.classList.remove('hidden');
            } else {
                msg.classList.add('hidden');
            }
        });
    }

    cancelVoiceInteraction() {
        // Mark interaction as cancelled
        this.voiceInteractionCancelled = true;

        // Stop speech recognition (unified method handles both engines)
        this.stopSpeechRecognition(true);

        // Close speech model modal if open
        this.closeSpeechModelModal();

        // Stop speech synthesis
        if (speechSynthesis) {
            speechSynthesis.cancel();
        }

        // Stop generation if in progress
        if (this.isGenerating) {
            this.stopGeneration().catch((error) => {
                console.warn('Failed to stop generation during voice cancel:', error);
            });
        }

        // Show all messages when user cancels
        this.showCaptions = true;
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            const messages = chatMessages.querySelectorAll('.message');
            messages.forEach(msg => {
                msg.classList.remove('hidden');
            });
        }

        // Reset UI
        this.isListening = false;
        this.isSpeaking = false;
        this.isGenerating = false;
        this.resetVoiceUI();
    }

    async previewVoice() {
        const voices = speechSynthesis.getVoices();
        const voiceSelect = document.getElementById('config-voice-select');
        const previewBtn = document.getElementById('preview-voice-btn');
        const selectedVoiceName = voiceSelect ? voiceSelect.value : null;

        if (!selectedVoiceName) {
            this.showToast('Please select a voice first');
            return;
        }

        const selectedVoice = voices.find(voice => voice.name === selectedVoiceName);
        if (!selectedVoice) {
            this.showToast('Voice not found');
            return;
        }

        // Cancel any ongoing speech
        speechSynthesis.cancel();

        // Show testing state on button
        if (previewBtn) {
            previewBtn.disabled = true;
            previewBtn.textContent = '...';
        }

        const resetButton = () => {
            if (previewBtn) {
                previewBtn.disabled = false;
                previewBtn.textContent = '▶';
            }
        };

        try {
            // If a selection-triggered probe is in-flight, let it settle first
            if (this.voiceSelectionProbePromise) {
                await Promise.race([
                    this.voiceSelectionProbePromise,
                    new Promise(resolve => setTimeout(resolve, 2200))
                ]);
            }

            await this.waitForSpeechIdle(1200);
            speechSynthesis.cancel();

            let result = await this.playPreviewAttempt(selectedVoice);

            // Retry once on transient startup issues
            if (!result.ok && !this.shouldMarkVoiceAsFailed(result.errorCode)) {
                await new Promise(resolve => setTimeout(resolve, 250));
                await this.waitForSpeechIdle(1200);
                speechSynthesis.cancel();
                result = await this.playPreviewAttempt(selectedVoice);
            }

            if (!result.ok) {
                console.error('Voice preview error:', result.errorCode);
                if (this.shouldMarkVoiceAsFailed(result.errorCode)) {
                    this.updateVoiceHealthStatus(selectedVoice, false);
                    this.populateVoices();
                }
                this.showToast('Voice preview failed. Please try another voice.');
            }
        } catch (error) {
            console.error('Error speaking:', error);
            this.showToast('Error playing voice preview');
        } finally {
            resetButton();
        }
    }

    // ========== End Speech and Voice Functions ==========

    // ========== Avatar Functions ==========

    initializeAvatars() {
        const avatarGrid = document.getElementById('avatar-grid');
        if (!avatarGrid) return;

        // Clear existing avatars
        avatarGrid.innerHTML = '';

        // Load saved preferences
        const savedAvatarEnabled = localStorage.getItem('avatarEnabled') === 'true';
        const savedAvatar = localStorage.getItem('selectedAvatar') || this.availableAvatars[0];

        this.avatarEnabled = savedAvatarEnabled;
        this.selectedAvatar = savedAvatar;

        // Set toggle state
        const avatarToggle = document.getElementById('avatar-toggle');
        if (avatarToggle) {
            avatarToggle.checked = savedAvatarEnabled;
        }

        // Show/hide avatar selection
        const avatarSelection = document.getElementById('avatar-selection');
        if (avatarSelection) {
            avatarSelection.style.display = savedAvatarEnabled ? 'block' : 'none';
        }

        // Create avatar items
        this.availableAvatars.forEach((avatar) => {
            const avatarItem = document.createElement('div');
            avatarItem.className = 'avatar-item';
            if (avatar === this.selectedAvatar) {
                avatarItem.classList.add('selected');
            }

            const img = document.createElement('img');
            img.src = `avatars/${avatar}`;
            img.alt = avatar.replace('.svg', '');

            const name = document.createElement('div');
            name.className = 'avatar-name';
            name.textContent = avatar.replace('.svg', '');

            avatarItem.appendChild(img);
            avatarItem.appendChild(name);

            avatarItem.addEventListener('click', () => {
                this.selectAvatar(avatar);
            });

            avatarGrid.appendChild(avatarItem);
        });

        // Update display if avatar is enabled
        if (this.avatarEnabled) {
            setTimeout(() => this.updateAvatarDisplay(), 0);
        }
    }

    toggleAvatar(enabled) {
        this.avatarEnabled = enabled;
        localStorage.setItem('avatarEnabled', enabled);

        const avatarSelection = document.getElementById('avatar-selection');

        if (enabled) {
            if (avatarSelection) {
                avatarSelection.style.display = 'block';
            }
            this.updateAvatarDisplay();
        } else {
            if (avatarSelection) {
                avatarSelection.style.display = 'none';
            }
            // Hide avatar image, show purple circle
            const avatarImage = document.getElementById('voice-avatar-image');
            if (avatarImage) {
                avatarImage.style.display = 'none';
            }
        }
    }

    selectAvatar(avatarName) {
        this.selectedAvatar = avatarName;
        localStorage.setItem('selectedAvatar', avatarName);

        // Update selection UI
        const avatarItems = document.querySelectorAll('.avatar-item');
        avatarItems.forEach(item => {
            const img = item.querySelector('img');
            if (img && img.src.endsWith(avatarName)) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });

        // Update avatar display if enabled
        if (this.avatarEnabled) {
            this.updateAvatarDisplay();
        }
    }

    updateAvatarDisplay() {
        const avatarImage = document.getElementById('voice-avatar-image');
        if (!avatarImage || !this.selectedAvatar) return;

        if (this.avatarEnabled) {
            avatarImage.src = `avatars/${this.selectedAvatar}`;
            avatarImage.style.display = 'block';
        } else {
            avatarImage.style.display = 'none';
        }
    }

    // ========== End Avatar Functions ==========

    showToast(message) {
        // Announce to screen readers
        this.announceToScreenReader(message);

        // Simple toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #6c3fa5;
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
        `;
        toast.textContent = message;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }
}

// Global functions for UI interactions
window.toggleSection = function (sectionId) {
    const content = document.getElementById(sectionId);
    const button = content.previousElementSibling;

    const isExpanded = content.style.display === 'block';

    if (isExpanded) {
        content.style.display = 'none';
        button.textContent = button.textContent.replace('▼', '▶');
        button.setAttribute('aria-expanded', 'false');
    } else {
        content.style.display = 'block';
        button.textContent = button.textContent.replace('▶', '▼');
        button.setAttribute('aria-expanded', 'true');
    }
};

window.resetParameters = function () {
    // Get the app instance (we'll need to store it globally)
    if (window.chatPlaygroundApp) {
        // Get model-specific defaults
        const defaults = window.chatPlaygroundApp.getModelDefaults();

        // Update app parameters
        window.chatPlaygroundApp.modelParameters = { ...defaults };

        // Update sliders and displays
        const updates = [
            { slider: 'temperature-slider', value: 'temperature-value', param: 'temperature' },
            { slider: 'top-p-slider', value: 'top-p-value', param: 'top_p' },
            { slider: 'max-tokens-slider', value: 'max-tokens-value', param: 'max_tokens' },
            { slider: 'repetition-penalty-slider', value: 'repetition-penalty-value', param: 'repetition_penalty' }
        ];

        updates.forEach(({ slider, value, param }) => {
            const sliderEl = document.getElementById(slider);
            const valueEl = document.getElementById(value);
            if (sliderEl && valueEl) {
                sliderEl.value = defaults[param];
                valueEl.textContent = defaults[param];
                // Update aria-valuetext for screen readers
                sliderEl.setAttribute('aria-valuetext', defaults[param].toString());
            }
        });

        window.chatPlaygroundApp.showToast('Parameters reset to defaults');
    }
};

window.triggerFileUpload = function () {
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.click();
    }
};

window.toggleAddDropdown = function (event) {
    event.stopPropagation();
    const btn = document.getElementById('add-tool-btn');
    const menu = document.getElementById('add-dropdown-menu');
    const isOpen = menu.style.display !== 'none';
    menu.style.display = isOpen ? 'none' : 'block';
    btn.setAttribute('aria-expanded', String(!isOpen));
};

window.selectTool = function (tool) {
    const menu = document.getElementById('add-dropdown-menu');
    const btn = document.getElementById('add-tool-btn');
    menu.style.display = 'none';
    btn.setAttribute('aria-expanded', 'false');
    if (tool === 'file-search') {
        window.triggerFileUpload();
    } else if (tool === 'web-search') {
        const info = document.getElementById('web-search-info');
        const option = document.getElementById('web-search-option');
        if (info) info.style.display = 'flex';
        if (option) option.style.display = 'none';
    }
};

window.removeWebSearch = function () {
    const info = document.getElementById('web-search-info');
    const option = document.getElementById('web-search-option');
    if (info) info.style.display = 'none';
    if (option) option.style.display = '';
};

document.addEventListener('click', function (e) {
    const menu = document.getElementById('add-dropdown-menu');
    const btn = document.getElementById('add-tool-btn');
    if (menu && !menu.contains(e.target) && e.target !== btn) {
        menu.style.display = 'none';
        if (btn) btn.setAttribute('aria-expanded', 'false');
    }
});

window.removeFile = function () {
    if (window.chatPlaygroundApp) {
        window.chatPlaygroundApp.removeFile();
    }
};

window.openAboutModal = function () {
    const modal = document.getElementById('about-modal');
    if (modal) {
        // Store the currently focused element to restore later
        window.lastFocusedElement = document.activeElement;

        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling

        // Focus the modal for screen readers
        setTimeout(() => {
            const modalTitle = document.getElementById('about-modal-title');
            if (modalTitle) {
                modalTitle.focus();
            }
        }, 100);

        // Add keyboard trap for accessibility
        window.trapFocus(modal);
    }
};

window.closeAboutModal = function () {
    const modal = document.getElementById('about-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling

        // Restore focus to the element that opened the modal
        if (window.lastFocusedElement) {
            window.lastFocusedElement.focus();
            window.lastFocusedElement = null;
        }

        // Remove keyboard trap
        window.removeFocusTrap();
    }
};

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Parameters Modal Functions
window.openParametersModal = function () {
    const modal = document.getElementById('parameters-modal');
    if (modal) {
        modal.style.display = 'flex';
        // Sync modal sliders with current values from left pane
        syncParametersToModal();
        // Add click-outside-to-close
        modal.addEventListener('click', handleParametersModalClick);
    }
};

window.closeParametersModal = function () {
    const modal = document.getElementById('parameters-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.removeEventListener('click', handleParametersModalClick);
    }
};

function handleParametersModalClick(e) {
    const modal = document.getElementById('parameters-modal');
    if (e.target === modal) {
        window.closeParametersModal();
    }
}

function updateModalSliderFromSource(sourceId, sourceValueId, value) {
    // Map source IDs to modal IDs
    const modalId = sourceId.replace('-slider', '') === sourceId.replace('-slider', '') ? 'modal-' + sourceId : 'modal-' + sourceId;
    const modalValueId = 'modal-' + sourceValueId;

    const modalSlider = document.getElementById(modalId);
    const modalValue = document.getElementById(modalValueId);

    if (modalSlider && modalValue) {
        modalSlider.value = value;
        modalValue.textContent = value;
        modalSlider.setAttribute('aria-valuetext', value.toString());
    }
}

function syncParametersToModal() {
    // Sync modal sliders from the app's config (the authoritative source of truth).
    // Reading from DOM source elements is avoided because those sidebar sliders
    // don't exist in the HTML — only the modal-prefixed ones do.
    const params = window.chatPlaygroundApp ? window.chatPlaygroundApp.config.modelParameters : null;

    if (params) {
        const updates = [
            { target: 'modal-temperature-slider', value: 'modal-temperature-value', param: 'temperature' },
            { target: 'modal-top-p-slider', value: 'modal-top-p-value', param: 'top_p' },
            { target: 'modal-max-tokens-slider', value: 'modal-max-tokens-value', param: 'max_tokens' },
            { target: 'modal-repetition-penalty-slider', value: 'modal-repetition-penalty-value', param: 'repetition_penalty' }
        ];

        updates.forEach(({ target, value, param }) => {
            const targetEl = document.getElementById(target);
            const valueEl = document.getElementById(value);
            if (targetEl && valueEl) {
                targetEl.value = params[param];
                valueEl.textContent = params[param];
                targetEl.setAttribute('aria-valuetext', params[param].toString());
            }
        });
    }

    // Attach event listeners only once per slider element to avoid accumulation
    // across multiple modal openings.
    ['modal-temperature-slider', 'modal-top-p-slider', 'modal-max-tokens-slider', 'modal-repetition-penalty-slider'].forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        if (slider && !slider.dataset.listenerAttached) {
            slider.addEventListener('input', handleModalParameterChange);
            slider.dataset.listenerAttached = 'true';
        }
    });
}

function handleModalParameterChange(e) {
    const slideId = e.target.id;
    const value = e.target.value;
    const valueId = slideId.replace('-slider', '-value');
    const valueEl = document.getElementById(valueId);

    if (valueEl) {
        valueEl.textContent = value;
        e.target.setAttribute('aria-valuetext', value);
    }

    // Also update the left pane slider
    const sourceId = slideId.replace('modal-', '');
    const sourceEl = document.getElementById(sourceId);
    if (sourceEl) {
        sourceEl.value = value;
        const sourceValueId = sourceId.replace('-slider', '-value');
        const sourceValueEl = document.getElementById(sourceValueId);
        if (sourceValueEl) {
            sourceValueEl.textContent = value;
            sourceEl.setAttribute('aria-valuetext', value);
        }
    }

    // Update app config
    if (window.chatPlaygroundApp) {
        const paramName = slideId.replace('modal-', '').replace('-slider', '');
        const paramKey = paramName === 'top-p' ? 'top_p' :
            paramName === 'max-tokens' ? 'max_tokens' :
                paramName === 'repetition-penalty' ? 'repetition_penalty' : paramName;
        window.chatPlaygroundApp.config.modelParameters[paramKey] = parseFloat(value);
    }
}

window.resetParametersFromModal = function () {
    // Get model-specific defaults
    const defaults = window.chatPlaygroundApp ? window.chatPlaygroundApp.getModelDefaults() : {
        temperature: 0.5,
        top_p: 0.9,
        max_tokens: 768,
        repetition_penalty: 1.1
    };

    // Update modal sliders
    const updates = [
        { slider: 'modal-temperature-slider', value: 'modal-temperature-value', param: 'temperature' },
        { slider: 'modal-top-p-slider', value: 'modal-top-p-value', param: 'top_p' },
        { slider: 'modal-max-tokens-slider', value: 'modal-max-tokens-value', param: 'max_tokens' },
        { slider: 'modal-repetition-penalty-slider', value: 'modal-repetition-penalty-value', param: 'repetition_penalty' }
    ];

    updates.forEach(({ slider, value, param }) => {
        const sliderEl = document.getElementById(slider);
        const valueEl = document.getElementById(value);
        const defaultVal = defaults[param];

        if (sliderEl && valueEl) {
            sliderEl.value = defaultVal;
            valueEl.textContent = defaultVal;
            sliderEl.setAttribute('aria-valuetext', defaultVal.toString());
        }

        // Also update left pane
        const sourceId = slider.replace('modal-', '');
        const sourceEl = document.getElementById(sourceId);
        const sourceValueId = sourceId.replace('-slider', '-value');
        const sourceValueEl = document.getElementById(sourceValueId);

        if (sourceEl && sourceValueEl) {
            sourceEl.value = defaultVal;
            sourceValueEl.textContent = defaultVal;
            sourceEl.setAttribute('aria-valuetext', defaultVal.toString());
        }
    });

    // Update app config
    if (window.chatPlaygroundApp) {
        window.chatPlaygroundApp.config.modelParameters = { ...defaults };
    }

    if (window.chatPlaygroundApp && window.chatPlaygroundApp.showToast) {
        window.chatPlaygroundApp.showToast('Parameters reset to defaults');
    }
};

// Focus trap functionality for modal accessibility
window.trapFocus = function (modal) {
    const focusableElements = modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    window.modalKeydownHandler = function (e) {
        if (e.key === 'Tab') {
            if (e.shiftKey) {
                if (document.activeElement === firstFocusable) {
                    lastFocusable.focus();
                    e.preventDefault();
                }
            } else {
                if (document.activeElement === lastFocusable) {
                    firstFocusable.focus();
                    e.preventDefault();
                }
            }
        } else if (e.key === 'Escape') {
            window.closeChatCapabilitiesModal(); // Modal removed
        }
    };

    document.addEventListener('keydown', window.modalKeydownHandler);
};

window.removeFocusTrap = function () {
    if (window.modalKeydownHandler) {
        document.removeEventListener('keydown', window.modalKeydownHandler);
        window.modalKeydownHandler = null;
    }
};

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatPlaygroundApp = new ChatPlayground();

    // Dark mode toggle handler
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggle.checked = true;
        }

        // Toggle dark mode
        themeToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
            }
        });
    }

    // Add parameters modal click-outside-to-close functionality
    const parametersModal = document.getElementById('parameters-modal');
    if (parametersModal) {
        parametersModal.addEventListener('click', (e) => {
            if (e.target === parametersModal) {
                window.closeParametersModal();
            }
        });
    }

    // Close modals with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const voiceInputErrorModal = document.getElementById('voice-input-error-modal');
            if (voiceInputErrorModal && voiceInputErrorModal.style.display !== 'none') {
                window.chatPlaygroundApp?.closeVoiceInputErrorModal();
                return;
            }

            const parametersModal = document.getElementById('parameters-modal');
            if (parametersModal && parametersModal.style.display !== 'none') {
                window.closeParametersModal();
            }
        }
    });
});
