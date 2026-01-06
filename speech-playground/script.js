import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class ChatPlayground {
    constructor() {
        // Core state
        this.engine = null;
        this.isModelLoaded = false;
        this.webllmAvailable = false;
        this.conversationHistory = [];
        this.isGenerating = false;
        this.isSpeaking = false;
        this.isListening = false;
        this.currentSystemMessage = "You are a helpful AI assistant that answers spoken questions with vocalized responses. IMPORTANT: Make your responses brief and to the point.";
        this.currentModelId = null;
        this.wikipediaRequestCount = 0;
        
        // Track pending messages to display after speaking
        this.pendingUserMessage = null;
        this.pendingAssistantMessage = null;
        
        // Track applied vs pending settings
        this.appliedModelId = null;
        this.appliedVoice = null;
        this.appliedSystemMessage = null;
        this.pendingModelId = null;
        this.pendingVoice = null;
        this.pendingSystemMessage = null;
        this.hasUnappliedChanges = false;

        // Configuration objects
        this.config = {
            modelParameters: {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 1000,
                repetition_penalty: 1.1
            }
        };

        // Initialize speech settings
        this.speechSettings = {
            speechToText: true,
            textToSpeech: true,
            voice: '',
            speed: '1x'
        };

        // Speech and vision state
        this.recognition = null;
        this.voicesAvailable = false;
        this.voicesLoaded = false;

        // Initialize DOM element registry
        this.elements = {};
        this.eventListeners = [];

        // Initialize app
        this.initialize();
    }

    // Constants for messages
    static MESSAGES = {
        ERRORS: {
            SPEECH_NOT_AVAILABLE: 'Speech recognition not available'
        },
        TOAST: {
            CHAT_CLEARED: 'Chat cleared',
            MODEL_CHANGED: 'Model changed to Wikipedia fallback',
            VOICE_APPLIED: 'Voice setting applied',
            INSTRUCTIONS_UPDATED: 'Instructions updated',
            SETTINGS_RESET: 'Settings reset to defaults',
            SPEECH_UNAVAILABLE: 'Speech recognition not available.',
            VOICE_INPUT_FAILED: 'Could not start voice input.',
            RESPONSE_ERROR: 'Error generating response. Please try again.',
            MODEL_LOAD_ERROR: 'Error loading models. Using Wikipedia mode.',
            LOADING_MODEL: (modelId) => `Loading ${modelId}...`,
            MODEL_LOADED: 'Model loaded successfully!',
            MODEL_LOAD_FALLBACK: 'Failed to load model. Using Wikipedia fallback.'
        }
    };

    // Centralized initialization
    initialize() {
        this.initializeElements();
        this.disallowInteraction();
        this.attachEventListeners();
        this.populateVoices();
        this.initializeSpeechRecognition();
        this.initializeModel();
    }

    initializeElements() {
        const elementSelectors = {
            progressContainer: 'progress-container',
            progressFill: 'progress-fill',
            progressText: 'progress-text',
            modelSelect: 'model-select',
            systemMessage: 'system-message',
            chatMessages: 'chat-messages',
            voiceSelect: 'voice-select',
            startBtn: 'start-btn',
            cancelBtn: 'cancel-btn',
            applySettingsBtn: 'apply-settings-btn',
            resetSettingsBtn: 'reset-settings-btn'
        };

        Object.entries(elementSelectors).forEach(([key, id]) => {
            this.elements[key] = document.getElementById(id);
        });

        // Legacy references
        this.progressContainer = this.elements.progressContainer;
        this.progressFill = this.elements.progressFill;
        this.progressText = this.elements.progressText;
        this.modelSelect = this.elements.modelSelect;
        this.systemMessage = this.elements.systemMessage;
        this.chatMessages = this.elements.chatMessages;
        this.voiceSelect = this.elements.voiceSelect;
        this.startBtn = this.elements.startBtn;
        this.cancelBtn = this.elements.cancelBtn;
        this.applySettingsBtn = this.elements.applySettingsBtn;
        this.resetSettingsBtn = this.elements.resetSettingsBtn;
    }

    attachEventListeners() {
        // Handle system message changes (both change and input events)
        if (this.systemMessage) {
            const updateSystemMessage = (e) => {
                this.pendingSystemMessage = e.target.value;
                if (this.pendingSystemMessage !== this.appliedSystemMessage) {
                    this.hasUnappliedChanges = true;
                    this.updateApplyButtonState();
                }
            };
            this.addEventListenerTracked(this.systemMessage, 'input', updateSystemMessage);
        }

        // Handle clear chat button
        const clearBtn = document.querySelector('.chat-header .icon-btn');
        if (clearBtn) {
            this.addEventListenerTracked(clearBtn, 'click', () => {
                this.conversationHistory = [];
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = `
                        <div class="welcome-message">
                            <div class="chat-icon" aria-hidden="true"></div>
                            <h3>Let's talk</h3>
                            <p>Talk like you would to a person. The agent listens and responds.</p>
                        </div>
                    `;
                }
                this.showToast(ChatPlayground.MESSAGES.TOAST.CHAT_CLEARED);
            });
        }

        // Handle Apply Changes button
        const applySettingsBtn = this.elements.applySettingsBtn;
        if (applySettingsBtn) {
            this.addEventListenerTracked(applySettingsBtn, 'click', () => {
                this.applySettings();
            });
        }

        // Handle Reset button
        const resetSettingsBtn = this.elements.resetSettingsBtn;
        if (resetSettingsBtn) {
            this.addEventListenerTracked(resetSettingsBtn, 'click', () => {
                this.resetSettings();
            });
        }

        // Handle model selection changes (to pending state)
        const modelSelect = this.elements.modelSelect;
        if (modelSelect) {
            this.addEventListenerTracked(modelSelect, 'change', (e) => {
                this.pendingModelId = e.target.value;
                // Enable Apply button if selection differs from applied model
                if (this.pendingModelId !== this.appliedModelId) {
                    this.hasUnappliedChanges = true;
                    this.updateApplyButtonState();
                }
            });
        }

        // Handle voice selection changes (to pending state)
        const voiceSelect = this.elements.voiceSelect;
        if (voiceSelect) {
            this.addEventListenerTracked(voiceSelect, 'change', (e) => {
                this.pendingVoice = e.target.value;
                // Enable Apply button if selection differs from applied voice
                if (this.pendingVoice !== this.appliedVoice) {
                    this.hasUnappliedChanges = true;
                    this.updateApplyButtonState();
                }
            });
        }

        // Handle preview voice button
        const previewBtn = document.getElementById('preview-voice-btn');
        if (previewBtn) {
            this.addEventListenerTracked(previewBtn, 'click', () => {
                this.previewVoice();
            });
        }
    }

    applySettings() {
        // Apply model change if pending
        if (this.pendingModelId !== null && this.pendingModelId !== this.appliedModelId) {
            if (this.pendingModelId === 'none') {
                this.webllmAvailable = false;
                this.engine = null;
                this.appliedModelId = 'none';
                this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_CHANGED);
            } else {
                this.appliedModelId = this.pendingModelId;
                this.loadModel(this.pendingModelId);
            }
        }

        // Apply voice change if pending
        if (this.pendingVoice !== null && this.pendingVoice !== this.appliedVoice) {
            this.speechSettings.voice = this.pendingVoice;
            this.appliedVoice = this.pendingVoice;
            this.showToast(ChatPlayground.MESSAGES.TOAST.VOICE_APPLIED);
        }

        // Apply system message change if pending
        if (this.pendingSystemMessage !== null && this.pendingSystemMessage !== this.appliedSystemMessage) {
            this.appliedSystemMessage = this.pendingSystemMessage;
            this.currentSystemMessage = this.pendingSystemMessage + ' IMPORTANT: Make your responses brief and to the point.';
            this.showToast(ChatPlayground.MESSAGES.TOAST.INSTRUCTIONS_UPDATED);
        }

        // Clear unapplied changes flag and disable Apply button
        this.hasUnappliedChanges = false;
        this.updateApplyButtonState();
    }

    updateApplyButtonState() {
        const applyBtn = this.elements.applySettingsBtn;
        if (applyBtn) {
            applyBtn.disabled = !this.hasUnappliedChanges;
        }
    }

    updateWelcomeState(heading, subheading, chatIconAnimation = null) {
        const welcomeHeading = document.querySelector('.welcome-message h3');
        const welcomeParagraph = document.querySelector('.welcome-message p');
        const chatIcon = document.querySelector('.chat-icon');

        if (welcomeHeading) welcomeHeading.textContent = heading;
        if (welcomeParagraph) welcomeParagraph.textContent = subheading;
        if (chatIconAnimation && chatIcon) chatIcon.style.animation = chatIconAnimation;
    }

    resetSettings() {
        // Reset model dropdown to default (Phi-3)
        const defaultModel = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
        if (this.elements.modelSelect) {
            this.elements.modelSelect.value = defaultModel;
            this.pendingModelId = defaultModel;
        }

        // Reset voice dropdown to first available voice
        const voiceSelect = this.elements.voiceSelect;
        if (voiceSelect && voiceSelect.options.length > 0) {
            voiceSelect.value = voiceSelect.options[0].value;
            this.pendingVoice = voiceSelect.options[0].value;
        }

        // Reset system message to original default value
        const defaultSystemMessage = 'You are a helpful AI assistant that answers spoken questions with vocalized responses.';
        if (this.systemMessage) {
            this.systemMessage.value = defaultSystemMessage;
            this.pendingSystemMessage = defaultSystemMessage;
        }

        this.showToast(ChatPlayground.MESSAGES.TOAST.SETTINGS_RESET);
    }

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

    addEventListenerTracked(element, event, handler, options = false) {
        if (typeof element === 'string') {
            element = this.getElement(element);
        }
        if (element) {
            element.addEventListener(event, handler, options);
            this.eventListeners.push({ element, event, handler, options });
        }
    }

    updateProgress(containerId, fillId, textId, percentage, text) {
        this.showElement(containerId);
        this.setElementStyle(fillId, 'width', `${percentage}%`);
        this.setElementText(textId, text);
    }

    populateVoices() {
        const voiceSelect = document.getElementById('voice-select');
        if (!voiceSelect) return;

        const loadVoices = () => {
            const voices = speechSynthesis.getVoices();
            const englishVoices = voices.filter(voice => voice && voice.lang && voice.lang.startsWith('en'));

            // If voices have already been loaded, preserve the current dropdown selection
            if (this.voicesLoaded) {
                return; // Already loaded, don't rebuild
            }
            
            this.voicesLoaded = true;
            voiceSelect.innerHTML = '';

            if (englishVoices.length > 0) {
                this.voicesAvailable = true;
                englishVoices.forEach((voice, index) => {
                    if (!voice || !voice.name) return;
                    const option = document.createElement('option');
                    option.value = voice.name;
                    option.textContent = `${voice.name} (${voice.lang})`;
                    voiceSelect.appendChild(option);
                });
                
                // Select applied voice if set, otherwise select first voice
                if (this.appliedVoice) {
                    voiceSelect.value = this.appliedVoice;
                    this.speechSettings.voice = this.appliedVoice;
                    this.pendingVoice = this.appliedVoice;
                } else {
                    voiceSelect.value = englishVoices[0].name;
                    this.speechSettings.voice = englishVoices[0].name;
                    this.pendingVoice = englishVoices[0].name;
                }
            } else {
                this.voicesAvailable = false;
                const option = document.createElement('option');
                option.value = 'none';
                option.textContent = 'No voices available';
                option.selected = true;
                voiceSelect.appendChild(option);
                voiceSelect.disabled = true;
                this.speechSettings.voice = null;
                this.pendingVoice = null;
            }
        };

        if (speechSynthesis.getVoices().length > 0) {
            loadVoices();
        } else {
            speechSynthesis.addEventListener('voiceschanged', loadVoices);
            setTimeout(loadVoices, 100);
        }
    }

    disallowInteraction() {
        // Disable all UI elements until model loads
        if (this.elements.modelSelect) this.elements.modelSelect.disabled = true;
        if (this.elements.systemMessage) this.elements.systemMessage.disabled = true;
        if (this.elements.voiceSelect) this.elements.voiceSelect.disabled = true;
        if (this.elements.startBtn) this.elements.startBtn.disabled = true;
        if (this.elements.applySettingsBtn) this.elements.applySettingsBtn.disabled = true;
        if (this.elements.resetSettingsBtn) this.elements.resetSettingsBtn.disabled = true;
        
        // Disable preview button
        const previewBtn = document.getElementById('preview-voice-btn');
        if (previewBtn) previewBtn.disabled = true;
    }

    allowInteraction() {
        // Enable UI elements when model is loaded or fallback selected
        if (this.elements.modelSelect) this.elements.modelSelect.disabled = false;
        if (this.elements.systemMessage) this.elements.systemMessage.disabled = false;
        if (this.elements.voiceSelect) this.elements.voiceSelect.disabled = false;
        if (this.elements.startBtn) this.elements.startBtn.disabled = false;
        if (this.elements.resetSettingsBtn) this.elements.resetSettingsBtn.disabled = false;
        
        // Enable preview button
        const previewBtn = document.getElementById('preview-voice-btn');
        if (previewBtn && this.voicesAvailable) previewBtn.disabled = false;
        
        // Apply button is only enabled if there are unapplied changes
        this.updateApplyButtonState();
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
            this.isListening = true;
            // Cancel button is shown in startVoiceInput(), keep it visible
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
            this.isListening = false;
            // Don't hide cancel button here - keep it visible during processing and speaking
        };

        this.recognition.onerror = (event) => {
            this.isListening = false;
            // Don't update button here - keep cancel visible during error handling
            
            // Show error modal for any speech recognition error
            this.showSpeechErrorModal();
            
            // Also log the error type for debugging
            console.error('Speech recognition error:', event.error);
        };
    }

    showSpeechErrorModal() {
        const modal = document.getElementById('speech-error-modal');
        if (modal) {
            modal.style.display = 'flex';
        }
        // Reset UI to welcome state
        this.resetToWelcomeState();
    }

    startVoiceInput() {
        if (!this.recognition) {
            this.showToast(ChatPlayground.MESSAGES.TOAST.SPEECH_UNAVAILABLE);
            return;
        }

        if (this.isListening) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.error('Error stopping speech recognition:', error);
            }
            return;
        }

        // Update UI to show listening state
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');
        const chatIcon = document.querySelector('.chat-icon');

        if (startBtn) startBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'inline-block';
        if (chatIcon) chatIcon.style.animation = 'pulse 1s infinite';
        
        this.updateWelcomeState('Listening...', 'Speak now..');

        try {
            try {
                this.recognition.abort();
            } catch (e) {
                // Ignore
            }

            setTimeout(() => {
                this.recognition.start();
            }, 100);
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            this.isListening = false;
            this.updateStartButton();
            this.showToast(ChatPlayground.MESSAGES.TOAST.VOICE_INPUT_FAILED);
        }
    }

    handleSpokenInput(transcript) {
        // Update UI - show processing state
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');

        if (startBtn) startBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'inline-block';
        
        this.updateWelcomeState('Processing...', 'This can take some time...');

        // Store the messages to display later after speaking
        this.pendingUserMessage = transcript;

        // Send to model
        this.generateResponse(transcript);
    }

    async generateResponse(userMessage) {
        this.isGenerating = true;

        try {
            let responseText = '';

            if (this.webllmAvailable && this.engine) {
                // Use WebLLM
                const messages = this.buildMessages(userMessage);
                
                // Use the chat completions API
                const completion = await this.engine.chat.completions.create({
                    messages: messages,
                    temperature: this.config.modelParameters.temperature,
                    top_p: this.config.modelParameters.top_p,
                    max_tokens: this.config.modelParameters.max_tokens,
                    repetition_penalty: this.config.modelParameters.repetition_penalty,
                    stream: true
                });
                
                // Collect the streamed response
                for await (const chunk of completion) {
                    if (!this.isGenerating) break;
                    
                    const content = chunk.choices[0]?.delta?.content || '';
                    if (content) {
                        responseText += content;
                    }
                }
            } else {
                // Use Wikipedia fallback
                responseText = await this.queryWikipedia(userMessage);
            }

            // Store the assistant message to display later
            this.pendingAssistantMessage = responseText;

            // Update UI to "Speaking..." state only if voices are available
            if (this.voicesAvailable) {
                this.updateWelcomeState('Speaking...', 'Adjust volume as necessary.');
            }

            // Speak the response (or skip to displaying if no voices)
            this.speakResponse(responseText);

            // Add to conversation history
            this.conversationHistory.push(
                { role: 'user', content: userMessage },
                { role: 'assistant', content: responseText }
            );
        } catch (error) {
            console.error('Error generating response:', error);
            this.showToast(ChatPlayground.MESSAGES.TOAST.RESPONSE_ERROR);
            this.resetToWelcomeState();
        } finally {
            this.isGenerating = false;
        }
    }

    buildMessages(userMessage) {
        const messages = [
            { role: 'system', content: this.currentSystemMessage },
            ...this.conversationHistory,
            { role: 'user', content: userMessage }
        ];
        return messages;
    }

    addMessageToChat(text, role) {
        if (!this.chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;

        messageDiv.appendChild(contentDiv);
        this.chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    async extractKeywords(text) {
        console.log('Original prompt:', text);
        
        // Remove punctuation from the text
        const textWithoutPunctuation = text.replace(/[.,!?;:'"()[\]{}]/g, ' ');
        console.log('Text without punctuation:', textWithoutPunctuation);
        
        // Tokenize and extract important words
        const tokens = textWithoutPunctuation.toLowerCase().split(/\s+/);
        console.log('Tokens:', tokens);
        
        // Remove common stop words only
        const stopWords = new Set(["a", "about", "above", "after", "again", "against", "all", "am",
        "an", "and", "any", "are", "aren't", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by",
        "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each",
        "few", "for", "from", "further",
        "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's",
        "her", "here", "here's", "hers", "herself",
        "him", "himself", "his",
        "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've",
        "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's",
        "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own",
        "same", "shan't", "she", "she'd", "she'll", "she's",
        "should", "shouldn't",
        "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too",
        "under", "until", "up",
        "very",
        "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
        "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
        "which", "while", "who", "who's", "whom", "why", "why's",
        "with", "won't", "would", "wouldn't",
        "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
        ]);


        // Keep all words that aren't stop words and are longer than 1 character
        const keywords = tokens.filter(word => 
            word.length > 1 && !stopWords.has(word)
        );
        
        console.log('Filtered keywords array:', keywords);

        // Return all keywords joined together
        const keywordString = keywords.join(' ') || text;
        console.log('Final keyword string for search:', keywordString);
        
        return keywordString;
    }

    async searchWikipedia(keywords) {
        try {
            // Search Wikipedia API
            const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(keywords)}&format=json&origin=*`;
            const searchResponse = await fetch(searchUrl);
            const searchData = await searchResponse.json();

            if (!searchData.query || !searchData.query.search || searchData.query.search.length === 0) {
                return "I couldn't find any relevant information on Wikipedia for your query.";
            }

            // Get the first result's page ID
            const firstResult = searchData.query.search[0];
            const pageId = firstResult.pageid;

            // Fetch the full article content
            const contentUrl = `https://en.wikipedia.org/w/api.php?action=query&pageids=${pageId}&prop=extracts&exintro=true&explaintext=true&format=json&origin=*`;
            const contentResponse = await fetch(contentUrl);
            const contentData = await contentResponse.json();

            const pageContent = contentData.query.pages[pageId].extract;

            console.log('Wikipedia page content received:', pageContent.substring(0, 500));
            console.log('Total content length:', pageContent.length);

            // Get intro section including any lists
            // Split by double newlines but keep content until we hit a new section
            const paragraphs = pageContent.split('\n');
            let introContent = '';
            let lineCount = 0;
            const maxLines = 15; // Get more lines to capture lists
            
            for (let i = 0; i < paragraphs.length && lineCount < maxLines; i++) {
                const line = paragraphs[i].trim();
                if (line.length > 0) {
                    introContent += (introContent ? '\n' : '') + line;
                    lineCount++;
                }
                // Stop if we hit a section header (usually === or ==)
                if (line.includes('==') && i > 0) {
                    break;
                }
            }
            
            console.log('Intro content extracted:', introContent.substring(0, 500));
            
            return introContent;

        } catch (error) {
            console.error('Wikipedia search error:', error);
            return "I encountered an error while searching Wikipedia. Please try again.";
        }
    }

    async queryWikipedia(userMessage) {
        try {
            this.wikipediaRequestCount++;
            
            // Extract keywords from user input
            let keywords = await this.extractKeywords(userMessage);
            console.log('Extracted keywords from message:', keywords);

            // Search Wikipedia with keywords
            console.log('Searching Wikipedia with:', keywords);
            const articleText = await this.searchWikipedia(keywords);
            
            // Summarize the text to keep it concise
            const summary = await this.summarizeText(articleText);

            return summary;

        } catch (error) {
            console.error('Wikipedia fallback error:', error);
            return 'Sorry, I encountered an error while processing your request. Please try again.';
        }
    }

    async summarizeText(text) {
        console.log('Summarizing text, length:', text.length);
        console.log('Text to summarize:', text.substring(0, 300));
        
        // Since we're already limiting content in searchWikipedia,
        // just return the text
        if (text.length < 800) {
            return text;
        }

        // For longer content, check if it has list-like structure
        const lines = text.split('\n');
        const hasShortLines = lines.filter(l => l.length > 0 && l.length < 100).length > 3;
        
        if (hasShortLines) {
            // Looks like a list - return first ~600 chars
            let summary = '';
            for (const line of lines) {
                if (summary.length + line.length < 600) {
                    summary += (summary ? '\n' : '') + line;
                } else {
                    break;
                }
            }
            return summary;
        }

        // For regular narrative text, return first 2-3 sentences
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        if (sentences.length <= 2) {
            return text;
        }

        const summaryLength = Math.min(3, sentences.length);
        return sentences.slice(0, summaryLength).join(' ').trim();
    }

    speakResponse(text) {
        if (!this.speechSettings.textToSpeech || !this.voicesAvailable) {
            this.onSpeechComplete();
            return;
        }

        if (!('speechSynthesis' in window)) {
            this.onSpeechComplete();
            return;
        }

        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        if (this.speechSettings.voice && this.speechSettings.voice !== 'default') {
            const voices = speechSynthesis.getVoices();
            const selectedVoice = voices.find(voice => voice.name === this.speechSettings.voice);
            if (selectedVoice) {
                utterance.voice = selectedVoice;
            }
        }

        const speedMap = { '0.5x': 0.5, '1x': 1, '1.5x': 1.5, '2x': 2 };
        utterance.rate = speedMap[this.speechSettings.speed] || 1;
        utterance.pitch = 1;
        utterance.volume = 1;

        this.isSpeaking = true;

        utterance.onend = () => {
            this.isSpeaking = false;
            this.onSpeechComplete();
        };

        utterance.onerror = () => {
            this.isSpeaking = false;
            this.onSpeechComplete();
        };

        speechSynthesis.speak(utterance);
    }

    previewVoice() {
        const voices = speechSynthesis.getVoices();
        const voiceSelect = this.elements.voiceSelect;
        const selectedVoiceName = voiceSelect ? voiceSelect.value : this.pendingVoice;

        if (!selectedVoiceName) {
            this.showToast('Please select a voice first');
            return;
        }

        // Cancel any ongoing speech
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance('This is my voice.');
        utterance.text = 'This is my voice.';
        
        const selectedVoice = voices.find(voice => voice.name === selectedVoiceName);
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
        
        utterance.rate = 1;
        speechSynthesis.speak(utterance);
    }

    onSpeechComplete() {
        // Display the pending messages now that speaking is complete
        if (this.pendingUserMessage) {
            this.addMessageToChat(this.pendingUserMessage, 'user');
            this.pendingUserMessage = null;
        }
        
        if (this.pendingAssistantMessage) {
            this.addMessageToChat(this.pendingAssistantMessage, 'assistant');
            this.pendingAssistantMessage = null;
        }

        // Reset UI to welcome state
        this.resetToWelcomeState();
    }

    resetToWelcomeState() {
        const chatIcon = document.querySelector('.chat-icon');
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');

        if (chatIcon) chatIcon.style.animation = 'none';
        if (startBtn) startBtn.style.display = 'inline-block';
        if (cancelBtn) cancelBtn.style.display = 'none';
        
        this.updateWelcomeState("Let's talk", 'Talk like you would to a person. The agent listens and responds.');
    }

    updateStartButton() {
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');

        if (this.isListening) {
            if (startBtn) startBtn.textContent = '🎙️';
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
        } else {
            if (startBtn) startBtn.textContent = 'Start';
            if (cancelBtn) cancelBtn.style.display = 'none';
        }
    }

    async initializeModel() {
        try {
            console.log('initializeModel called');
            
            // Check if WebLLM is available
            if (!webllm || !webllm.CreateMLCEngine || !webllm.prebuiltAppConfig) {
                console.error('WebLLM not properly loaded');
                throw new Error('WebLLM not properly loaded');
            }

            // Update model select with specific models
            const modelSelect = this.elements.modelSelect;
            
            if (modelSelect) {
                modelSelect.innerHTML = '';
                
                // Add "None" option for Wikipedia fallback
                const noneOption = document.createElement('option');
                noneOption.value = 'none';
                noneOption.textContent = 'None (Wikipedia)';
                modelSelect.appendChild(noneOption);

                // Add only the Phi-3 mini model
                const phiOption = document.createElement('option');
                phiOption.value = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
                phiOption.textContent = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
                modelSelect.appendChild(phiOption);
            }

            // Load default model (Phi-3-mini-4k-instruct)
            const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
            if (modelSelect) {
                modelSelect.value = targetModelId;
            }
            this.pendingModelId = targetModelId;
            
            // Initialize system message settings
            const initialSystemMessage = 'You are a helpful AI assistant that answers spoken questions with vocalized responses.';
            if (this.systemMessage) {
                this.systemMessage.value = initialSystemMessage;
            }
            this.appliedSystemMessage = initialSystemMessage;
            this.pendingSystemMessage = initialSystemMessage;
            this.currentSystemMessage = initialSystemMessage + ' IMPORTANT: Make your responses brief and to the point.';
            
            await this.loadModel(targetModelId);
            
        } catch (error) {
            console.error('Error initializing models:', error);
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOAD_ERROR);
            this.webllmAvailable = false;
        }
    }

    async loadModel(modelId) {
        if (!modelId || modelId === 'none') return;

        try {
            console.log(`Trying to load model: ${modelId}`);
this.showToast(ChatPlayground.MESSAGES.TOAST.LOADING_MODEL(modelId));

            this.engine = await webllm.CreateMLCEngine(
                modelId,
                {
                    initProgressCallback: (progress) => {
                        console.log('Model loading progress:', progress);
                        const percentage = Math.round(progress.progress * 100);
                        this.updateProgress(
                            'progressContainer',
                            'progressFill',
                            'progressText',
                            percentage,
                            `Downloading model: ${percentage}%`
                        );
                    }
                }
            );

            console.log(`Successfully loaded model: ${modelId}`);
            this.currentModelId = modelId;
            this.appliedModelId = modelId;
            this.webllmAvailable = true;
            this.hideElement('progressContainer');
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOADED);
            this.allowInteraction();
        } catch (error) {
            console.error(`Error loading model ${modelId}:`, error);
            this.webllmAvailable = false;
            this.engine = null;
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOAD_FALLBACK);
            // Allow interaction with fallback mode
            this.appliedModelId = 'none';
            this.allowInteraction();
        }
    }

    showToast(message) {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            z-index: 1001;
            max-width: 300px;
        `;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

}

// Global functions for UI interactions
window.openAboutModal = function() {
    const modal = document.getElementById('about-modal');
    if (modal) {
        modal.style.display = 'flex';
        // Set focus to modal for accessibility
        const closeBtn = modal.querySelector('.modal-close-btn');
        if (closeBtn) setTimeout(() => closeBtn.focus(), 100);
        // Trap focus within modal
        window.currentModal = modal;
    }
};

window.closeAboutModal = function() {
    const modal = document.getElementById('about-modal');
    if (modal) {
        modal.style.display = 'none';
        window.currentModal = null;
        // Restore focus to trigger button
        const helpBtn = document.querySelector('.chat-controls .help-btn');
        if (helpBtn) helpBtn.focus();
    }
};

window.closeSpeechErrorModal = function() {
    const modal = document.getElementById('speech-error-modal');
    if (modal) {
        modal.style.display = 'none';
        window.currentModal = null;
        // Restore focus to start button
        const startBtn = document.getElementById('start-btn');
        if (startBtn) startBtn.focus();
    }
};

window.sendManualInput = function() {
    const input = document.getElementById('manual-input');
    if (input && input.value.trim()) {
        const text = input.value.trim();
        window.chatPlaygroundApp.handleSpokenInput(text);
        input.value = '';
        window.closeSpeechErrorModal();
    }
};

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    window.chatPlaygroundApp = new ChatPlayground();

    // Attach Start button listener
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            window.chatPlaygroundApp.startVoiceInput();
        });
    }

    // Attach Cancel button listener
    const cancelBtn = document.getElementById('cancel-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            // Stop speech synthesis if playing
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            
            // Stop listening if recording
            if (window.chatPlaygroundApp.recognition) {
                window.chatPlaygroundApp.recognition.stop();
            }
            
            // Display pending messages before clearing them
            if (window.chatPlaygroundApp.pendingUserMessage) {
                window.chatPlaygroundApp.addMessageToChat(window.chatPlaygroundApp.pendingUserMessage, 'user');
            }
            if (window.chatPlaygroundApp.pendingAssistantMessage) {
                window.chatPlaygroundApp.addMessageToChat(window.chatPlaygroundApp.pendingAssistantMessage, 'assistant');
            }
            
            // Clear pending messages
            window.chatPlaygroundApp.pendingUserMessage = null;
            window.chatPlaygroundApp.pendingAssistantMessage = null;
            
            // Reset state
            window.chatPlaygroundApp.isListening = false;
            window.chatPlaygroundApp.isSpeaking = false;
            window.chatPlaygroundApp.isGenerating = false;
            
            // Reset UI
            window.chatPlaygroundApp.resetToWelcomeState();
        });
    }

    // Add about modal click-outside-to-close
    const aboutModal = document.getElementById('about-modal');
    if (aboutModal) {
        aboutModal.addEventListener('click', (e) => {
            if (e.target === aboutModal) {
                window.closeAboutModal();
            }
        });
    }

    // Add speech error modal click-outside-to-close
    const errorModal = document.getElementById('speech-error-modal');
    if (errorModal) {
        errorModal.addEventListener('click', (e) => {
            if (e.target === errorModal) {
                window.closeSpeechErrorModal();
            }
        });
    }

    // Close modals with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            window.closeAboutModal();
            window.closeSpeechErrorModal();
        }
        
        // Trap focus within modal when one is open
        if (window.currentModal && window.currentModal.style.display !== 'none' && e.key === 'Tab') {
            const focusableElements = window.currentModal.querySelectorAll(
                'button, a, input, textarea, select, [tabindex]:not([tabindex=\"-1\"])'
            );
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            if (e.shiftKey) {
                if (document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                }
            } else {
                if (document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        }
    });
});
