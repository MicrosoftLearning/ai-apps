import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js';

class AskAnton {
    constructor() {
        this.engine = null;      // WebLLM engine
        this.wllama = null;      // Wllama engine (fallback)
        this.conversationHistory = [];
        this.isGenerating = false;
        this.indexData = null;
        this.stopRequested = false;
        this.currentStream = null;
        this.currentAbortController = null;
        this.webGPUAvailable = false;
        this.usingWllama = false;
        this.isLoadingModel = false;
        this.currentModal = null;
        this.lastFocusedElement = null;
        this.modalFocusTrapHandler = null;
        this.usedVoiceInput = false;

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
            chatContainer: document.getElementById('chat-container'),
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            micBtn: document.getElementById('mic-btn'),
            restartBtn: document.getElementById('restart-btn'),
            searchStatus: document.getElementById('search-status'),
            modeToggle: document.getElementById('mode-toggle'),
            modeToggleText: document.getElementById('mode-toggle-text'),
            aboutBtn: document.getElementById('about-btn'),
            aboutModal: document.getElementById('about-modal'),
            aboutModalClose: document.getElementById('about-modal-close'),
            aboutModalOk: document.getElementById('about-modal-ok'),
            aiModeModal: document.getElementById('ai-mode-modal'),
            modalClose: document.getElementById('modal-close'),
            modalOk: document.getElementById('modal-ok')
        };

        this.systemPrompt = `You are Anton, a knowledgeable and friendly AI learning assistant who helps students understand AI concepts.

IMPORTANT: Follow these guidelines when responding:
- Do not engage in conversation on topics other than artificial intelligence and computing.
- Explain concepts clearly and concisely in a single paragraph based only on the provided context.
- Keep responses short and focused on the question, with no headings.
- Use examples and analogies when helpful.
- Use simple language suitable for learners in a conversational, friendly tone.
- Provide a general descriptions and overviews, but do NOT provide explicit steps or instructions for developing AI solutions.
- If the context includes "Sorry, I couldn't find any specific information on that topic. Please try rephrasing your question or explore other AI concepts.", use that exact phrasing and no additional information.
- Do not start responses with "A:" or "Q:".
- Keep your responses concise and to the point, in ONE paragraph.
- Do NOT provide links for more information (these will be added automatically later).`;

        // Prohibited words for content moderation (whole words only)
        this.prohibitedWords = [];

        this.initialize();
    }

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    async initialize() {
        try {
            // Load prohibited words used by content moderation
            await this.loadProhibitedWords();

            // Load the index (no longer loading Vosk upfront)
            await this.loadIndex();

            // Try to initialize WebLLM first, fall back to wllama if needed
            await this.initializeEngine();

            // Setup event listeners
            this.setupEventListeners();

        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize. Please refresh the page.');
        }
    }

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================

    reverseWord(text) {
        return text.split('').reverse().join('');
    }

    shiftWord(text, amount) {
        return text
            .split('')
            .map(char => String.fromCharCode(char.charCodeAt(0) + amount))
            .join('');
    }

    async loadProhibitedWords() {
        try {
            const response = await fetch('moderation/mod.txt');
            if (!response.ok) throw new Error('Failed to load prohibited words');

            const encodedWordsText = await response.text();
            this.prohibitedWords = encodedWordsText
                .split(/\r?\n/)
                .map(word => word.trim())
                .filter(word => word.length > 0)
                .map(word => this.shiftWord(this.reverseWord(word.toLowerCase()), 1));

            console.log('Loaded prohibited words:', this.prohibitedWords.length);
        } catch (error) {
            console.error('Error loading prohibited words:', error);
            throw error;
        }
    }

    async loadIndex() {
        try {
            this.updateProgress(5, 'Loading knowledge base...');
            const response = await fetch('index.json');
            if (!response.ok) throw new Error('Failed to load index');
            this.indexData = await response.json();
            console.log('Loaded index with', this.indexData.length, 'categories');

            // Build a flat lookup map: keyword -> {document, category, link}
            this.keywordMap = new Map();
            this.indexData.forEach(category => {
                category.documents.forEach(doc => {
                    doc.keywords.forEach(keyword => {
                        const normalizedKeyword = keyword.toLowerCase().trim();
                        if (normalizedKeyword) {
                            this.keywordMap.set(normalizedKeyword, {
                                document: doc,
                                category: category.category,
                                link: category.link
                            });
                        }
                    });
                });
            });
            console.log('Built keyword map with', this.keywordMap.size, 'keywords');
        } catch (error) {
            console.error('Error loading index:', error);
            throw error;
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

            this.enableInput();
            this.elements.micBtn.disabled = false;
            return true;
        } catch (error) {
            console.error('Error loading Vosk model:', error);
            this.voskLoadingFailed = true;
            this.addSystemMessage('Failed to load offline speech model. Voice input is unavailable.');
            this.enableInput();
            this.elements.micBtn.disabled = false;
            return false;
        }
    }

    // ============================================================================
    // LLM ENGINE INITIALIZATION (WebLLM & Wllama)
    // ============================================================================

    checkWebGPUSupport() {
        // Check if WebGPU is available in the browser
        if (!navigator.gpu) {
            console.log('WebGPU not supported in this browser');
            return false;
        }
        return true;
    }

    async initializeEngine() {
        // Check for WebGPU support before attempting to load WebLLM
        const hasWebGPU = this.checkWebGPUSupport();

        if (!hasWebGPU) {
            console.log('WebGPU not available, using wllama (CPU mode)');
            await this.initializeWllama();
            return;
        }

        // Try WebLLM first (faster with GPU)
        try {
            await this.initializeWebLLM();
        } catch (error) {
            console.log('WebLLM initialization failed, falling back to wllama');
            await this.initializeWllama();
        }
    }

    async initializeWebLLM() {
        try {
            this.updateProgress(15, 'Loading AI model (WebGPU)...');

            const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';

            this.engine = await webllm.CreateMLCEngine(
                targetModelId,
                {
                    initProgressCallback: (progress) => {
                        const percentage = Math.max(15, Math.round(progress.progress * 85) + 15);
                        this.updateProgress(
                            percentage,
                            `Loading model: ${Math.round(progress.progress * 100)}%`
                        );
                    }
                }
            );

            this.updateProgress(100, 'Ready to chat!');
            console.log('WebLLM engine initialized successfully');
            this.webGPUAvailable = true;
            this.usingWllama = false;

            setTimeout(() => {
                this.showChatInterface();
            }, 500);

        } catch (error) {
            console.error('Failed to initialize WebLLM:', error);
            throw error; // Re-throw to trigger fallback
        }
    }

    async initializeWllama(progressCallback = null) {
        try {
            // Check if already initialized
            if (this.wllama) {
                console.log('Wllama already initialized');
                return;
            }

            const isLazyLoad = this.webGPUAvailable; // If WebGPU is available, this is a lazy load

            if (!isLazyLoad) {
                this.updateProgress(15, 'Loading AI model (CPU mode)...');
            }

            // Configure WASM paths for CDN
            const CONFIG_PATHS = {
                'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm',
                'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm',
            };

            // Try multithreaded (4 threads) first if cross-origin isolated, fall back to single-threaded
            const useMultiThread = window.crossOriginIsolated === true;
            const preferredThreads = useMultiThread ? 4 : 1;
            console.log(`Cross-origin isolated: ${window.crossOriginIsolated}, attempting ${preferredThreads} thread(s)`);

            const modelConfig = {
                n_ctx: 512,      // Smaller context for faster processing
                n_threads: preferredThreads,
                progressCallback: ({ loaded, total }) => {
                    const percentage = Math.max(15, Math.round((loaded / total) * 85) + 15);
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

            try {
                // Initialize wllama with CDN-hosted WASM files
                this.wllama = new Wllama(CONFIG_PATHS);

                // Load model from HuggingFace with optimized settings
                await this.wllama.loadModelFromHF(
                    'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                    'smollm2-360m-instruct-q8_0.gguf',
                    modelConfig
                );
                console.log(`Wllama initialized successfully with ${preferredThreads} thread(s)`);
            } catch (multiErr) {
                if (preferredThreads > 1) {
                    console.warn(`Multi-threaded init failed (${multiErr.message}), falling back to single thread`);

                    // Retry with single thread
                    this.wllama = new Wllama(CONFIG_PATHS);
                    await this.wllama.loadModelFromHF(
                        'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                        'smollm2-360m-instruct-q8_0.gguf',
                        {
                            ...modelConfig,
                            n_threads: 1
                        }
                    );
                    console.log('Wllama initialized successfully with 1 thread (fallback)');
                } else {
                    throw multiErr;
                }
            }

            if (!isLazyLoad) {
                this.updateProgress(100, 'Ready to chat! (CPU mode)');
            }
            console.log('Wllama initialized successfully with SmolLM2-360M-Instruct');

            if (!isLazyLoad) {
                this.webGPUAvailable = false;
                this.usingWllama = true;

                setTimeout(() => {
                    this.showChatInterface();
                }, 500);
            }

        } catch (error) {
            console.error('Failed to initialize wllama:', error);
            if (!this.webGPUAvailable) {
                this.showError('Failed to load AI model. Please refresh the page.');
            }
            throw error;
        }
    }

    // ============================================================================
    // UI STATE MANAGEMENT
    // ============================================================================

    updateProgress(percentage, text) {
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = text;

        // Update progress bar ARIA attributes
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.setAttribute('aria-label', text);
        }
    }

    showChatInterface() {
        this.elements.progressSection.style.display = 'none';
        this.elements.chatContainer.style.display = 'flex';
        this.updateModeToggle();
        this.elements.userInput.focus();
    }

    showError(message) {
        this.elements.progressText.textContent = message;
        this.elements.progressFill.style.backgroundColor = '#dc3545';
    }

    disableInput() {
        this.elements.userInput.disabled = true;
        this.elements.sendBtn.disabled = true;
        this.elements.micBtn.disabled = true;
        this.elements.userInput.placeholder = 'Loading model...';
    }

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

    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => {
            if (this.isGenerating) {
                this.stopGeneration();
            } else {
                this.sendMessage();
            }
        });

        // Enter key to send (Shift+Enter for new line)
        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !this.isGenerating) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.elements.userInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });

        // Keyboard navigation
        this.elements.userInput.addEventListener('keydown', (e) => {
            // Enter to send (without Shift)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.isGenerating) {
                    this.sendMessage();
                }
            }
            // Escape to stop generation
            if (e.key === 'Escape' && this.isGenerating) {
                this.stopGeneration();
            }
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

        // Mode toggle button
        this.elements.modeToggle.addEventListener('change', () => {
            this.toggleMode();
        });

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

        // Dynamic AI mode link keyboard handling (for links added to messages)
        this.elements.chatMessages.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.classList.contains('ai-mode-link')) {
                e.preventDefault();
                e.target.click();
            }
        });
    }

    // ============================================================================
    // CONTENT MODERATION & TEXT PROCESSING
    // ============================================================================

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    containsProhibitedWords(text) {
        // Convert to lowercase for case-insensitive matching
        const lowerText = text.toLowerCase();

        // Create word boundaries regex pattern for whole word matching
        for (const word of this.prohibitedWords) {
            // Use word boundary to match whole words only
            const regex = new RegExp(`\\b${word}\\b`, 'i');
            if (regex.test(lowerText)) {
                console.log(`Content moderation: blocked word "${word}" detected`);
                return true;
            }
        }

        return false;
    }

    normalizeSearchText(text) {
        return text.toLowerCase().trim().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    }

    getSearchIntentQuery(text) {
        const trimmedText = text.trim();
        const lowerText = trimmedText.toLowerCase();

        if (lowerText.startsWith('search ')) {
            return trimmedText.slice(7).trim();
        }

        if (lowerText.startsWith('find ')) {
            return trimmedText.slice(5).trim();
        }

        return null;
    }

    extractBingSearchKeywords(text) {
        const normalizedText = this.normalizeSearchText(text);
        const words = normalizedText.split(' ').filter(Boolean);
        const stopWords = new Set([
            'a', 'an', 'and', 'anton', 'for', 'from', 'i', 'in', 'me', 'of', 'on',
            'or', 'please', 'show', 'tell', 'the', 'to', 'up', 'use', 'using', 'with'
        ]);
        const uniqueWords = [];
        const seenWords = new Set();

        words.forEach(word => {
            if (word.length < 2 || stopWords.has(word) || seenWords.has(word)) {
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
        const stopWords = ['what', 'is', 'are', 'the', 'a', 'an', 'how', 'does', 'do', 'can', 'about', 'tell', 'me', 'explain', 'describe', 'show', 'give', 'anton', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'why', 'which', 'whom', 'whose', 'why', 'all', 'any', 'this', 'that', 'these', 'those'];
        words.forEach(word => {
            if (word.length >= 2 && !stopWords.includes(word)) {
                nGrams.push({
                    text: word,
                    length: 1
                });
            }
        });

        console.log('Extracted n-grams:', nGrams.map(ng => `"${ng.text}" (${ng.length})`));

        // Match n-grams to keywords in the index
        const matchedKeywords = new Set();
        const documentMatches = new Map(); // doc id -> {doc, category, link, matchedKeywords[]}

        nGrams.forEach(ngram => {
            const match = this.keywordMap.get(ngram.text);
            if (match) {
                matchedKeywords.add(ngram.text);

                const docId = match.document.id;
                if (!documentMatches.has(docId)) {
                    documentMatches.set(docId, {
                        document: match.document,
                        category: match.category,
                        link: match.link,
                        matchedKeywords: []
                    });
                }
                documentMatches.get(docId).matchedKeywords.push(ngram.text);
            }
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

    searchContext(userQuestion) {
        const { matches, matchedKeywords } = this.performSearch(userQuestion);

        // If no matches, fall back to AI Concepts category
        if (matches.length === 0) {
            this.elements.searchStatus.textContent = '🔍 No specific context found';
            const aiConceptsCategory = this.indexData.find(cat => cat.category === 'AI Concepts');
            if (aiConceptsCategory && aiConceptsCategory.documents.length > 0) {
                const fallbackDoc = aiConceptsCategory.documents[0];
                return {
                    context: fallbackDoc.content,
                    categories: [aiConceptsCategory.category],
                    links: [aiConceptsCategory.link],
                    documents: [fallbackDoc]
                };
            }
            return { context: null, categories: [], links: [], documents: [] };
        }

        // Build context from all matched documents - use full content, no summarization
        const contextParts = matches.map(match => {
            return match.document.content;
        });

        const categories = [...new Set(matches.map(m => m.category))];
        const links = [...new Set(matches.map(m => m.link))];
        const documents = matches.map(m => m.document);

        this.elements.searchStatus.textContent = `🔍 Found context in: ${categories.join(', ')}`;

        return {
            context: contextParts.join('\n\n'),
            categories: categories,
            links: links,
            documents: documents
        };
    }

    // ============================================================================
    // MESSAGE HANDLING & RESPONSE GENERATION
    // ============================================================================

    async sendMessage() {
        const userMessage = this.elements.userInput.value.trim();

        // Validate input
        if (!userMessage || this.isGenerating || this.isLoadingModel) return;

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
            return;
        }

        // Check if wllama is still loading when in CPU mode
        if (this.usingWllama && !this.wllama) {
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

        // Generate response
        await this.generateResponse(userMessage, searchResult, usedVoice);
    }

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

    stopGeneration() {
        this.isGenerating = false;
        this.stopRequested = true;
        this.currentStream = null;

        // Abort the generation properly using AbortController
        if (this.currentAbortController) {
            console.log('Aborting generation via AbortController');
            this.currentAbortController.abort();
            this.currentAbortController = null;
        }

        this.updateSendButton(false);
        console.log('Stop requested');
    }

    toggleMode() {
        if (!this.webGPUAvailable) {
            this.lastFocusedElement = document.activeElement;
            this.showAiModeModal();
            return;
        }

        this.usingWllama = !this.usingWllama;
        this.updateModeToggle();

        // If switching to wllama and it's not loaded yet, show loading message
        if (this.usingWllama && !this.wllama) {
            this.isLoadingModel = true;
            this.disableInput();
            const loadingMsg = this.addSystemMessage('Switching to CPU mode - loading model... 0%');
            const loadingMsgElement = loadingMsg.querySelector('p');

            // Lazy load wllama
            this.initializeWllama((progress) => {
                // Update the loading message with progress
                if (loadingMsgElement) {
                    const percentage = Math.round(progress * 100);
                    loadingMsgElement.textContent = `Switching to CPU mode - loading model... ${percentage}%`;
                }
            }).then(() => {
                const mode = this.usingWllama ? 'CPU' : 'GPU';
                if (loadingMsgElement) {
                    loadingMsgElement.textContent = `Switched to ${mode} mode`;
                }
                this.isLoadingModel = false;
                this.enableInput();
            }).catch(error => {
                console.error('Failed to load wllama:', error);
                if (loadingMsgElement) {
                    loadingMsgElement.textContent = 'Failed to load CPU mode. Reverting to GPU mode.';
                }
                this.usingWllama = false;
                this.updateModeToggle();
                this.isLoadingModel = false;
                this.enableInput();
            });
        } else {
            const mode = this.usingWllama ? 'CPU' : 'GPU';
            console.log(`Switched to ${mode} mode`);
            this.addSystemMessage(`Switched to ${mode} mode`);
        }
    }

    updateModeToggle() {
        const isGpuMode = !this.usingWllama;

        // Update checkbox state
        this.elements.modeToggle.checked = this.usingWllama;
        this.elements.modeToggle.setAttribute('aria-checked', this.usingWllama ? 'true' : 'false');

        // Update text label
        this.elements.modeToggleText.textContent = isGpuMode ? 'GPU' : 'CPU';

        // Update title and aria-label
        const modeTitle = isGpuMode ?
            'Currently using GPU (WebLLM). Toggle to switch to CPU mode.' :
            'Currently using CPU (wllama). Toggle to switch to GPU mode.';
        const ariaLabel = isGpuMode ?
            'Toggle engine mode. Currently in GPU mode. Toggle to switch to CPU mode.' :
            'Toggle engine mode. Currently in CPU mode. Toggle to switch to GPU mode.';

        this.elements.modeToggle.parentElement.parentElement.title = modeTitle;
        this.elements.modeToggle.setAttribute('aria-label', ariaLabel);

        // Disable toggle if WebGPU not available
        if (!this.webGPUAvailable) {
            this.elements.modeToggle.disabled = true;
            this.elements.modeToggle.checked = true; // CPU mode
            this.elements.modeToggle.setAttribute('aria-checked', 'true');
            this.elements.modeToggleText.textContent = 'CPU';
            this.elements.modeToggle.parentElement.parentElement.title = 'WebGPU not available - CPU mode only. Click for more info.';
            this.elements.modeToggle.setAttribute('aria-label', 'Engine mode set to CPU only. WebGPU not available. Press for more information.');

            // Make the label clickable to show info modal
            const label = this.elements.modeToggle.parentElement;
            label.style.cursor = 'pointer';
            label.tabIndex = 0; // Make focusable for keyboard navigation

            label.onclick = (e) => {
                if (this.elements.modeToggle.disabled) {
                    e.preventDefault();
                    this.lastFocusedElement = document.activeElement;
                    this.showAiModeModal();
                }
            };

            // Add keyboard support for info modal when disabled
            label.onkeydown = (e) => {
                if (this.elements.modeToggle.disabled && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    this.lastFocusedElement = document.activeElement;
                    this.showAiModeModal();
                }
            };
        }
    }

    // ============================================================================
    // MESSAGE UI RENDERING
    // ============================================================================

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

    renderAssistantMessage(messageTextDiv, assistantMessage, categories = [], links = [], placeholders = {}) {
        let displayMessage = assistantMessage;

        if (links && links.length > 0 && categories && categories.length > 0) {
            displayMessage += '\n\n---\n\n**Learn more:** [[LEARN_MORE_LINKS]]';
        }

        let formattedMessage = this.formatResponse(displayMessage);

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
    }

    async respondWithSearchLink(userMessage, searchQuery, usedVoiceInput = false) {
        const searchResult = this.searchContext(searchQuery);
        const bingKeywords = this.extractBingSearchKeywords(searchQuery) || this.normalizeSearchText(searchQuery);
        const encodedKeywords = encodeURIComponent(bingKeywords).replace(/%20/g, '+');
        const bingUrl = `https://www.bing.com/search?q=site%3Alearn.microsoft.com+${encodedKeywords}`;
        const historyAssistantMessage = `OK, I searched for "${bingKeywords}".\nHere's what I found.`;
        const assistantMessage = historyAssistantMessage.replace("Here's what I found.", '[[SEARCH_RESULT_LINK]]');

        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);

        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');
        const searchLinkHtml = `<a href="${bingUrl}" target="_blank" rel="noopener noreferrer">Here's what I found.</a>`;

        if (this.usingWllama) {
            messageTextDiv.innerHTML = '<span class="typing-indicator" aria-label="Anton is typing">●●●</span><p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow in CPU mode. Thanks for your patience!)</p>';
        } else {
            messageTextDiv.innerHTML = '<span class="typing-indicator">●●●</span>';
        }

        try {
            await new Promise(resolve => setTimeout(resolve, 250));

            if (usedVoiceInput) {
                this.playRandomResponseAudio();
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

            setTimeout(() => {
                this.elements.searchStatus.textContent = '';
            }, 2000);
        }
    }

    async generateResponse(userMessage, searchResult, usedVoiceInput = false) {
        const { context, categories, links } = searchResult;

        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);

        // Add empty message that we'll stream into
        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');

        // Show thinking indicator with CPU mode notice if applicable
        if (this.usingWllama) {
            messageTextDiv.innerHTML = '<span class="typing-indicator" aria-label="Anton is typing">●●●</span><p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow in CPU mode. Thanks for your patience!)</p>';
        } else {
            messageTextDiv.innerHTML = '<span class="typing-indicator">●●●</span>';
        }

        try {
            // Route to the appropriate engine
            let assistantMessage = '';

            if (this.usingWllama) {
                assistantMessage = await this.generateWithWllama(userMessage, context, messageTextDiv, usedVoiceInput);
            } else {
                assistantMessage = await this.generateWithWebLLM(userMessage, context, messageTextDiv, usedVoiceInput);
            }

            // Add learn more links
            if (links && links.length > 0 && categories && categories.length > 0) {
                this.renderAssistantMessage(messageTextDiv, assistantMessage, categories, links);
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
            if (!this.usingWllama) {
                this.addMessage('assistant', 'Sorry, I encountered an error in GPU mode. Try switching to CPU mode using the toggle at the top, then ask your question again.');
            } else {
                this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        } finally {
            this.isGenerating = false;
            this.stopRequested = false;
            this.currentStream = null;
            this.updateSendButton(false);

            // Clear search status after response is complete
            setTimeout(() => {
                this.elements.searchStatus.textContent = '';
            }, 2000);
        }
    }

    // ============================================================================
    // LLM RESPONSE GENERATION (WebLLM & Wllama)
    // ============================================================================

    async generateWithWebLLM(userMessage, context, messageTextDiv, usedVoiceInput = false) {
        let userPrompt = userMessage;
        if (context) {
            userPrompt = `${context}\n\nQ: ${userMessage}`;
        }

        const recentHistory = this.conversationHistory.slice(-6);
        recentHistory.push({
            role: 'user',
            content: userPrompt
        });

        const messages = [
            { role: 'system', content: this.systemPrompt },
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

    async generateWithWllama(userMessage, context, messageTextDiv, usedVoiceInput = false) {
        // Ensure wllama is loaded
        if (!this.wllama) {
            throw new Error('Wllama is not initialized. Please wait for CPU mode to finish loading.');
        }

        // Build ChatML formatted prompt
        let chatMLPrompt = '<|im_start|>system\n';
        chatMLPrompt += 'You are Anton, a teacher of AI and computing concepts. You always follow these rules.\n\n';
        chatMLPrompt += 'Rules:\n';
        chatMLPrompt += '- Discuss AI and computing topics only\n';
        chatMLPrompt += '- Do not provide specific steps or instructions\n\n';
        chatMLPrompt += '- Provide factual and accurate information\n\n';
        chatMLPrompt += '<|im_end|>\n\n';

        // Add truncated previous prompt and response if available
        if (this.conversationHistory.length >= 2) {
            // Get the last user message and assistant response
            const prevUser = this.conversationHistory[this.conversationHistory.length - 2];
            const prevAssistant = this.conversationHistory[this.conversationHistory.length - 1];

            if (prevUser.role === 'user' && prevAssistant.role === 'assistant') {
                const prevUserSentence = this.extractFirstSentence(prevUser.content);
                const prevAssistantSentence = this.extractFirstSentence(prevAssistant.content);

                chatMLPrompt += '<|im_start|>user\n';
                chatMLPrompt += prevUserSentence + '\n';
                chatMLPrompt += '<|im_end|>\n\n';
                chatMLPrompt += '<|im_start|>assistant\n';
                chatMLPrompt += prevAssistantSentence + '\n';
                chatMLPrompt += '<|im_end|>\n\n';
            }
        }

        // Add current user message
        chatMLPrompt += '<|im_start|>user\n';
        // Add context from index.json if available (truncate to prevent context overflow)
        if (context) {
            const maxContextLength = 400;
            const truncatedContext = context.length > maxContextLength
                ? context.substring(0, maxContextLength) + '...'
                : context;
            chatMLPrompt += 'Respond by summarizing the following information:\n---\n' + truncatedContext + '\n';
        }
        chatMLPrompt += userMessage + '\n';
        chatMLPrompt += '<|im_end|>\n\n';
        chatMLPrompt += '<|im_start|>assistant\n';

        console.log('Sending prompt to wllama (length:', chatMLPrompt.length, 'chars)');

        let assistantMessage = '';
        let audioPlayed = false;

        // Create AbortController for this generation
        const controller = new AbortController();
        this.currentAbortController = controller;

        // Clear KV cache before generation to ensure clean state
        try {
            await this.wllama.kvClear();
            console.log('KV cache cleared before generation');
        } catch (error) {
            console.log('KV cache clear failed:', error.message);
        }

        // Use streaming with proper abort support
        try {
            const completion = await this.wllama.createCompletion(chatMLPrompt, {
                nPredict: 150,
                sampling: {
                    temp: 0.7,
                    top_k: 40,
                    top_p: 0.9,
                    penalty_repeat: 1.1
                },
                stopTokens: ['<|im_end|>', '<|im_start|>'],
                abortSignal: controller.signal,
                stream: true
            });

            this.currentStream = completion;

            for await (const chunk of completion) {
                if (chunk.currentText) {
                    // Play audio on first chunk if voice input was used
                    if (!audioPlayed && usedVoiceInput) {
                        this.playRandomResponseAudio();
                        audioPlayed = true;
                    }

                    assistantMessage = chunk.currentText;
                    messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
                    this.scrollToBottom();
                }
            }

            // Clear abort controller on successful completion
            this.currentAbortController = null;

            // Clear KV cache after successful generation
            console.log('Clearing KV cache after generation');
            await this.wllama.kvClear();
            console.log('KV cache cleared successfully');

        } catch (error) {
            // Check if this was an abort (expected when user clicks stop)
            if (error.name === 'AbortError' || error.message?.includes('abort')) {
                console.log('Generation aborted by user');
                // Clear the partial/corrupted state
                await this.wllama.kvClear();
                console.log('KV cache cleared after abort');
            } else {
                console.log('Wllama generation error:', error.message || 'unknown error');
                // Clear cache on error too
                try {
                    await this.wllama.kvClear();
                } catch (e) {
                    console.log('Failed to clear cache after error:', e.message);
                }
            }
            this.currentAbortController = null;
        }

        console.log('Wllama response complete, length:', assistantMessage.length);

        return assistantMessage;
    }

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

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

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

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    // ============================================================================
    // SPEECH RECOGNITION HELPER METHODS
    // ============================================================================

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

    playRandomResponseAudio() {
        // Randomly select one of the 7 audio files
        const audioNumber = Math.floor(Math.random() * 7) + 1;
        const audioPath = `audio/response_${audioNumber}.wav`;

        const audio = new Audio(audioPath);
        audio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }

    playModerationAudio() {
        const audio = new Audio('moderation/sorry.wav');
        audio.play().catch(error => {
            console.error('Error playing moderation audio:', error);
        });
    }

    // ============================================================================
    // SPEECH RECOGNITION - WEB SPEECH API & VOSK
    // ============================================================================

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

    restartConversation() {
        if (confirm('Are you sure you want to start a new conversation? This will clear the chat history.')) {
            // Clear conversation history
            this.conversationHistory = [];

            // Clear chat messages (keep welcome message)
            const messages = this.elements.chatMessages.querySelectorAll('.message:not(.welcome-message)');
            messages.forEach(msg => msg.remove());

            // Clear search status
            this.elements.searchStatus.textContent = '';

            console.log('Conversation restarted');
        }
    }

    showAiModeModal() {
        this.elements.aiModeModal.style.display = 'flex';
        this.currentModal = this.elements.aiModeModal;
        this.elements.aiModeModal.setAttribute('aria-hidden', 'false');
        setTimeout(() => {
            this.elements.modalClose.focus();
            this.setupModalFocusTrap(this.elements.aiModeModal);
        }, 100);
    }

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
