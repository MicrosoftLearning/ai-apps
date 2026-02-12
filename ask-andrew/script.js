import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js';

class AskAndrew {
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
        this.currentModal = null;
        this.lastFocusedElement = null;
        this.modalFocusTrapHandler = null;
        this.usedVoiceInput = false;
        
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
        
        this.systemPrompt = `You are Andrew, a knowledgeable and friendly AI learning assistant who helps students understand AI concepts.

IMPORTANT: Follow these guidelines when responding:
- Do not engage in conversation on topics other than artificial intelligence and computing.
- Explain concepts clearly and concisely in a single paragraph based only on the provided context.
- Keep responses short and focused on the question, with no headings.
- Use examples and analogies when helpful.
- Use simple language suitable for learners in a conversational, friendly tone.
- Provide a general descriptions and overviews, but do NOT provide explicit steps or instructions for developing AI solutions.
- If the context includes "Sorry, I couldn't find any specific information on that topic. Please try rephrasing your question or explore other AI concepts.", use that exact phrasing and no additional information.
- Do not start responses with "A:" or "Q:".
- Keep your responses concise and to the point.
- Do NOT provide links for more information (these will be added automatically later).`;

        this.initialize();
    }

    async initialize() {
        try {
            // Load the index
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

    async initializeEngine() {
        // Try WebLLM first (faster with GPU)
        try {
            await this.initializeWebLLM();
        } catch (error) {
            console.log('WebLLM not available, falling back to wllama');
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
            
            // Initialize wllama with CDN-hosted WASM files
            this.wllama = new Wllama(CONFIG_PATHS);
            
            // Load model from HuggingFace with optimized settings
            await this.wllama.loadModelFromHF(
                'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                'smollm2-360m-instruct-q8_0.gguf',
                {
                    n_ctx: 512,      // Smaller context for faster processing
                    n_threads: 1,     // Single thread can be more stable
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
                }
            );
            
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

    autoResizeTextarea() {
        const textarea = this.elements.userInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    performSearch(userQuestion) {
        const lowerQuestion = userQuestion.toLowerCase().trim();
        
        // Normalize the question: remove punctuation, extra spaces
        const normalizedQuestion = lowerQuestion.replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
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
        const stopWords = ['what', 'is', 'are', 'the', 'a', 'an', 'how', 'does', 'do', 'can', 'about', 'tell', 'me', 'explain', 'describe', 'show', 'give'];
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
                    context: `[${aiConceptsCategory.category}]\n${fallbackDoc.content}`,
                    categories: [aiConceptsCategory.category],
                    links: [aiConceptsCategory.link],
                    documents: [fallbackDoc]
                };
            }
            return { context: null, categories: [], links: [], documents: [] };
        }
        
        // Build context from all matched documents - use full content, no summarization
        const contextParts = matches.map(match => {
            return `[${match.category} - ${match.document.title}]\n${match.document.content}`;
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

    async sendMessage() {
        const userMessage = this.elements.userInput.value.trim();
        
        // Validate input
        if (!userMessage || this.isGenerating) return;
        
        // Limit message length to prevent abuse
        const MAX_MESSAGE_LENGTH = 1000;
        if (userMessage.length > MAX_MESSAGE_LENGTH) {
            this.addSystemMessage(`Message too long. Please keep it under ${MAX_MESSAGE_LENGTH} characters.`);
            return;
        }
        
        // Check if wllama is still loading when in CPU mode
        if (this.usingWllama && !this.wllama) {
            this.addSystemMessage('CPU mode is still loading. Please wait...');
            return;
        }
        
        // Store voice input flag before clearing
        const usedVoice = this.usedVoiceInput;
        this.usedVoiceInput = false;
        
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
                const greetingResponse = "Hello, I'm Andrew. I'm here to help you learn about AI concepts. What would you like to know?";
                this.addMessage('assistant', greetingResponse);
                return;
            }
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
            }).catch(error => {
                console.error('Failed to load wllama:', error);
                if (loadingMsgElement) {
                    loadingMsgElement.textContent = 'Failed to load CPU mode. Reverting to GPU mode.';
                }
                this.usingWllama = false;
                this.updateModeToggle();
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
        messageDiv.setAttribute('aria-label', `Message from ${role === 'assistant' ? 'Andrew' : 'You'}`);
        
        if (role === 'assistant') {
            messageDiv.innerHTML = `
                <div class="avatar andrew-avatar" aria-hidden="true">
                    <img src="images/andrew-icon.png" alt="Andrew the AI assistant avatar" class="avatar-image">
                </div>
                <div class="message-content">
                    <p class="message-author" aria-label="From Andrew">Andrew</p>
                    <div class="message-text" ${isTyping ? 'aria-live="polite" aria-busy="true"' : ''}>
                        ${isTyping 
                            ? '<span class="typing-indicator" aria-label="Andrew is typing">●●●</span>' 
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
            messageTextDiv.innerHTML = '<span class="typing-indicator">●●●</span><p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow in CPU mode. Thanks for your patience!)</p>';
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
                // Store original message without learn more for conversation history
                const originalMessage = assistantMessage;
                
                // Add placeholder for learn more section for display only
                assistantMessage += '\n\n---\n\n**Learn more:** [[LEARN_MORE_LINKS]]';
                
                // Format the message
                let formattedMessage = this.formatResponse(assistantMessage);
                
                // Build HTML links with category names
                const linkHtml = links.map((link, index) => {
                    const categoryName = categories[Math.min(index, categories.length - 1)];
                    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
                }).join(' • ');
                formattedMessage = formattedMessage.replace(/\[\[LEARN_MORE_LINKS\]\]/g, linkHtml);
                
                messageTextDiv.innerHTML = formattedMessage;
                
                // Reset assistantMessage to original for conversation history
                assistantMessage = originalMessage;
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
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
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

    async generateWithWllama(userMessage, context, messageTextDiv, usedVoiceInput = false) {
        // Ensure wllama is loaded
        if (!this.wllama) {
            throw new Error('Wllama is not initialized. Please wait for CPU mode to finish loading.');
        }
        
        // Build ChatML formatted prompt
        let chatMLPrompt = '<|im_start|>system\n';
        chatMLPrompt += 'You are Andrew, an AI learning assistant. Answer questions using ONLY the information below.\n\n';
        chatMLPrompt += 'Rules:\n';
        chatMLPrompt += '- AI and computing topics only\n';
        chatMLPrompt += '- One clear paragraph, simple language\n';
        chatMLPrompt += '- No development steps or instructions\n\n';
        chatMLPrompt += 'Information:\n';
        
        // Add context from index.json if available (truncate to prevent context overflow)
        if (context) {
            const maxContextLength = 400;
            const truncatedContext = context.length > maxContextLength 
                ? context.substring(0, maxContextLength) + '...' 
                : context;
            chatMLPrompt += truncatedContext + '\n';
        } else {
            chatMLPrompt += 'No specific information available.\n';
        }
        
        chatMLPrompt += '<|im_end|>\n\n';
        
        // Don't include previous conversation history to keep prompt minimal and fast
        
        // Add current user message
        chatMLPrompt += '<|im_start|>user\n';
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

    async animateTyping(element, htmlContent, speed = 5) {
        // Parse HTML to extract text while preserving structure
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;
        
        // For simple animation, just show the content progressively
        element.innerHTML = '';
        const words = htmlContent.split(' ');
        
        for (let i = 0; i < words.length; i++) {
            if (this.stopRequested) break;
            
            element.innerHTML = words.slice(0, i + 1).join(' ');
            this.scrollToBottom();
            
            // Small delay between words
            await new Promise(resolve => setTimeout(resolve, speed));
        }
        
        // Ensure final content is complete
        element.innerHTML = htmlContent;
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }
    
    playRandomResponseAudio() {
        // Randomly select one of the 7 audio files
        const audioNumber = Math.floor(Math.random() * 7) + 1;
        const audioPath = `audio/response_${audioNumber}.wav`;
        
        const audio = new Audio(audioPath);
        audio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }

    handleMicClick() {
        // Check if Speech Recognition is available
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            this.addMessage('assistant', 'Speech input is not available in this browser.');
            return;
        }
        
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        // Visual feedback - button appears active while listening
        this.elements.micBtn.style.opacity = '0.6';
        this.elements.micBtn.title = 'Listening...';
        this.elements.micBtn.setAttribute('aria-label', 'Listening to your voice input');
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.elements.userInput.value = transcript;
            this.autoResizeTextarea();
            this.usedVoiceInput = true;
            this.sendMessage();
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.addMessage('assistant', 'Speech input is not available.');
            this.elements.micBtn.style.opacity = '1';
            this.elements.micBtn.title = 'Voice input';
            this.elements.micBtn.setAttribute('aria-label', 'Voice input');
        };
        
        recognition.onend = () => {
            this.elements.micBtn.style.opacity = '1';
            this.elements.micBtn.title = 'Voice input';
            this.elements.micBtn.setAttribute('aria-label', 'Voice input');
        };
        
        try {
            recognition.start();
            console.log('Speech recognition started');
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            this.addMessage('assistant', 'Speech input is not available.');
            this.elements.micBtn.style.opacity = '1';
            this.elements.micBtn.title = 'Voice input';
            this.elements.micBtn.setAttribute('aria-label', 'Voice input');
        }
    }
    
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
window.askAndrew = null;

// Initialize the app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.askAndrew = new AskAndrew();
    });
} else {
    window.askAndrew = new AskAndrew();
}
