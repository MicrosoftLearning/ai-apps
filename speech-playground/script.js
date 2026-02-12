import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/index.js';

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
        this.wllama = null; // wllama engine for CPU mode
        this.usingWllama = false; // Track which engine is active
        this.wllamaLoaded = false; // Track if wllama is initialized
        this.isModelLoaded = false;
        this.webllmAvailable = false;
        this.conversationHistory = [];
        this.isGenerating = false;
        this.isSpeaking = false;
        this.isListening = false;
        this.currentSystemMessage = "You are a helpful AI assistant that answers spoken questions with vocalized responses. IMPORTANT: Make your responses brief and to the point.";
        this.currentModelId = null;
        this.currentAbortController = null; // Track abort controller for wllama
        
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
        this.showCaptions = false; // Track whether to show conversation text during input

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
            MODEL_CHANGED: 'Model changed to SmolLM2 (CPU) fallback',
            VOICE_APPLIED: 'Voice setting applied',
            INSTRUCTIONS_UPDATED: 'Instructions updated',
            SETTINGS_RESET: 'Settings reset to defaults',
            SPEECH_UNAVAILABLE: 'Speech recognition not available.',
            VOICE_INPUT_FAILED: 'Could not start voice input.',
            RESPONSE_ERROR: 'Error generating response. Please try again.',
            MODEL_LOAD_ERROR: 'Error loading models. Using SmolLM2 (CPU) mode.',
            LOADING_MODEL: (modelId) => `Loading ${modelId}...`,
            MODEL_LOADED: 'Model loaded successfully!',
            MODEL_LOAD_FALLBACK: 'Failed to load model. Using SmolLM2 (CPU) fallback.'
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
                let value = e.target.value;
                // Enforce character limit
                if (value.length > 2000) {
                    value = value.substring(0, 2000);
                    e.target.value = value;
                }
                this.pendingSystemMessage = value;
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

        // Handle CC toggle button
        const ccBtn = document.getElementById('cc-btn');
        if (ccBtn) {
            this.addEventListenerTracked(ccBtn, 'click', () => {
                this.toggleCaptions();
            });
        }
    }

    async applySettings() {
        // Apply model change if pending
        if (this.pendingModelId !== null && this.pendingModelId !== this.appliedModelId) {
            if (this.pendingModelId === 'smollm2-cpu') {
                // Clear chat when switching models
                this.clearChat();
                
                // If wllama not loaded yet, load it
                if (!this.wllamaLoaded) {
                    console.log('Loading wllama for the first time...');
                    
                    // Disable UI during model loading
                    this.disallowInteraction();
                    
                    // Show progress
                    this.showElement('progressContainer');
                    
                    try {
                        await this.initializeWllama((loaded, total) => {
                            const percentage = Math.round((loaded / total) * 100);
                            this.updateProgress(
                                'progressContainer',
                                'progressFill',
                                'progressText',
                                percentage,
                                `Loading SmolLM2 (CPU): ${percentage}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`,
                                true
                            );
                        });
                        
                        this.usingWllama = true;
                        this.wllamaLoaded = true;
                        this.webllmAvailable = false;
                        this.engine = null;
                        this.appliedModelId = 'smollm2-cpu';
                        
                        this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_CHANGED);
                    } catch (error) {
                        console.error('Failed to load wllama:', error);
                        // Reset to previous selection
                        if (this.elements.modelSelect) {
                            this.elements.modelSelect.value = this.appliedModelId;
                        }
                        this.pendingModelId = this.appliedModelId;
                        alert('Failed to load SmolLM2 (CPU). Please try again.');
                    } finally {
                        // Re-enable UI
                        this.allowInteraction();
                    }
                } else {
                    // Wllama already loaded, just switch to it
                    this.usingWllama = true;
                    this.webllmAvailable = false;
                    this.engine = null;
                    this.appliedModelId = 'smollm2-cpu';
                    this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_CHANGED);
                }
            } else {
                // Switching to WebLLM model
                this.clearChat();
                this.appliedModelId = this.pendingModelId;
                await this.loadModel(this.pendingModelId);
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
            // Validate system message length
            let sanitizedMessage = this.pendingSystemMessage;
            if (sanitizedMessage.length > 2000) {
                sanitizedMessage = sanitizedMessage.substring(0, 2000);
                this.systemMessage.value = sanitizedMessage;
            }
            
            // Remove control characters except newlines
            sanitizedMessage = sanitizedMessage.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
            
            this.appliedSystemMessage = sanitizedMessage;
            this.currentSystemMessage = sanitizedMessage + ' IMPORTANT: Make your responses brief and to the point.';
            this.showToast(ChatPlayground.MESSAGES.TOAST.INSTRUCTIONS_UPDATED);
        }

        // Clear unapplied changes flag and disable Apply button
        this.hasUnappliedChanges = false;
        this.updateApplyButtonState();
    }

    clearChat() {
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

    updateProgress(containerId, fillId, textId, percentage, text, useHTML = false) {
        this.showElement(containerId);
        this.setElementStyle(fillId, 'width', `${percentage}%`);
        const element = this.getElement(textId);
        if (element) {
            if (useHTML) {
                element.innerHTML = text;
            } else {
                element.textContent = text;
            }
        }
        // Update ARIA attribute for accessibility
        const container = this.getElement(containerId);
        if (container) {
            container.setAttribute('aria-valuenow', Math.round(percentage));
        }
    }

    populateVoices() {
        const voiceSelect = document.getElementById('voice-select');
        if (!voiceSelect) return;

        const loadVoices = () => {
            const voices = speechSynthesis.getVoices();
            // Get all English voices (both local and online-capable)
            const englishVoices = voices.filter(voice => voice && voice.lang && voice.lang.startsWith('en'));

            // Preserve currently selected voice before rebuilding
            const currentlySelectedVoice = voiceSelect.value;

            voiceSelect.innerHTML = '';

            if (englishVoices.length > 0) {
                this.voicesAvailable = true;
                englishVoices.forEach((voice, index) => {
                    if (!voice || !voice.name) return;
                    const option = document.createElement('option');
                    option.value = voice.name;
                    // Label local voices as "Local" for clarity; online voices will be unmarked
                    const localLabel = voice.localService ? ' (Local)' : '';
                    option.textContent = `${voice.name} (${voice.lang})${localLabel}`;
                    voiceSelect.appendChild(option);
                });
                
                // Restore the user's selection: prioritize currently selected, then applied, then first voice
                if (currentlySelectedVoice && englishVoices.find(v => v.name === currentlySelectedVoice)) {
                    voiceSelect.value = currentlySelectedVoice;
                    this.speechSettings.voice = currentlySelectedVoice;
                    this.pendingVoice = currentlySelectedVoice;
                } else if (this.appliedVoice) {
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
        const ccBtn = document.getElementById('cc-btn');
        const chatIcon = document.querySelector('.chat-icon');

        if (startBtn) startBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'inline-block';
        if (ccBtn) ccBtn.style.display = 'none';  // Hide during listening to prevent interference
        if (chatIcon) chatIcon.style.animation = 'pulse 1s infinite';
        
        // Reset captions to hidden at start of new conversation
        this.showCaptions = false;
        this.updateCCButton();
        this.updateMessageVisibility();
        
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
        // Validate and sanitize transcript
        if (!transcript || typeof transcript !== 'string') {
            console.error('Invalid transcript received');
            this.resetToWelcomeState();
            return;
        }
        
        // Trim and enforce maximum length
        let sanitizedTranscript = transcript.trim();
        if (sanitizedTranscript.length > 1000) {
            sanitizedTranscript = sanitizedTranscript.substring(0, 1000);
        }
        
        // Remove control characters except newlines
        sanitizedTranscript = sanitizedTranscript.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
        
        // Check if there's still valid content
        if (sanitizedTranscript.length === 0) {
            console.error('Transcript empty after sanitization');
            this.resetToWelcomeState();
            return;
        }
        
        // Update UI - show processing state
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');
        const ccBtn = document.getElementById('cc-btn');

        if (startBtn) startBtn.style.display = 'none';
        if (cancelBtn) cancelBtn.style.display = 'inline-block';
        if (ccBtn) ccBtn.style.display = 'inline-block';  // Show CC button now that listening is done
        
        this.updateWelcomeState('Processing...', 'This can take some time...');

        // Add user message to chat immediately (respecting current CC visibility)
        this.addMessageToChat(sanitizedTranscript, 'user');
        this.pendingUserMessage = sanitizedTranscript;

        // Send to model
        this.generateResponse(sanitizedTranscript);
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
            } else if (this.usingWllama && this.wllama) {
                // Use wllama fallback
                responseText = await this.generateWllamaResponse(userMessage);
            } else {
                // No model available
                responseText = "No AI model is currently available. Please wait for the model to load or refresh the page.";
            }

            // Add assistant message to chat immediately (respecting current CC visibility)
            this.addMessageToChat(responseText, 'assistant');
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

    // Helper function to build ChatML formatted prompt for SmolLM2
    buildChatMLPrompt(userMessage) {
        let prompt = '';
        
        // Get the last turn of conversation history (if exists)
        let previousUserMessage = '';
        let previousAssistantResponse = '';
        
        if (this.conversationHistory.length >= 2) {
            // Get the last pair (user message and assistant response)
            previousAssistantResponse = this.conversationHistory[this.conversationHistory.length - 1].content;
            previousUserMessage = this.conversationHistory[this.conversationHistory.length - 2].content;
        }
        
        // Default format for speech-based interaction
        prompt = '<|im_start|>system\n';
        prompt += 'You are a rules‑driven assistant. Your highest priority is to follow the instructions exactly as written.\n\n';
        prompt += 'Instructions:\n';
        prompt += this.currentSystemMessage + '\n\n';
        prompt += 'Acknowledge these rules by answering the user\'s question correctly.\n';
        prompt += '<|im_end|>\n\n';
        
        // Add previous turn if exists
        if (previousUserMessage) {
            prompt += '<|im_start|>user\n' + previousUserMessage + '\n<|im_end|>\n\n';
            prompt += '<|im_start|>assistant\n' + previousAssistantResponse + '\n<|im_end|>\n\n';
        }
        
        // Add current user message
        prompt += '<|im_start|>user\n' + userMessage + '\n<|im_end|>\n\n';
        prompt += '<|im_start|>assistant\n';
        
        return prompt;
    }

    async generateWllamaResponse(userMessage) {
        // Ensure wllama is loaded
        if (!this.wllama) {
            return 'SmolLM2 is not initialized. Please wait for CPU mode to finish loading.';
        }
        
        try {
            // Build the ChatML prompt
            const chatMLPrompt = this.buildChatMLPrompt(userMessage);
            
            console.log('=== CHATML PROMPT FOR SMOLLM2 ===');
            console.log('Conversation history length:', this.conversationHistory.length);
            console.log('ChatML prompt:');
            console.log(chatMLPrompt);
            console.log('=== END CHATML PROMPT ===');
            
            // Use wllama for generation
            let fullResponse = '';
            
            // Use conservative parameters for wllama
            const wllamaTemp = 0.3;
            const wllamaTopP = 0.7;
            const wllamaPenalty = 1.1;
            
            // Log sampling parameters for debugging
            console.log('SmolLM2 sampling parameters:', {
                temp: wllamaTemp,
                top_k: 40,
                top_p: wllamaTopP,
                penalty_repeat: wllamaPenalty
            });
            
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
            
            // Generate response (non-streaming for speech)
            fullResponse = await this.wllama.createCompletion(chatMLPrompt, {
                nPredict: 200,  // Keep responses brief for speech
                seed: -1,  // Random seed for variation
                sampling: {
                    temp: wllamaTemp,
                    top_k: 40,
                    top_p: wllamaTopP,
                    penalty_repeat: wllamaPenalty,
                    mirostat: 0  // Disable mirostat to ensure temperature is used
                },
                stopTokens: ['<|im_end|>', '<|im_start|>'],
                abortSignal: controller.signal
            });
            
            console.log('Wllama response received:', fullResponse);
            
            // Clear abort controller on successful completion
            this.currentAbortController = null;
            
            // Clear KV cache after successful generation
            console.log('Clearing KV cache after generation');
            await this.wllama.kvClear();
            console.log('KV cache cleared successfully');
            
            return fullResponse.trim() || 'Sorry, I couldn\'t generate a response.';
            
        } catch (error) {
            // Check if this was an abort (expected when user clicks cancel)
            if (error.name === 'AbortError' || error.message?.includes('abort')) {
                console.log('Generation aborted by user');
                // Clear the partial/corrupted state
                await this.wllama.kvClear();
                console.log('KV cache cleared after abort');
                return 'Response cancelled.';
            } else {
                console.error('Error in wllama generation:', error);
                // Clear cache on error too
                try {
                    await this.wllama.kvClear();
                } catch (e) {
                    console.log('Failed to clear cache after error:', e.message);
                }
                return 'Sorry, I encountered an error while generating a response. Please try again.';
            }
        } finally {
            this.currentAbortController = null;
        }
    }

    addMessageToChat(text, role) {
        if (!this.chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;

        messageDiv.appendChild(contentDiv);
        
        // Hide message by default if captions are off and we're in a conversation
        if (!this.showCaptions && (this.isListening || this.isGenerating || this.isSpeaking)) {
            messageDiv.classList.add('hidden');
        }
        
        this.chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
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
        const previewBtn = document.getElementById('preview-voice-btn');
        const selectedVoiceName = voiceSelect ? voiceSelect.value : this.pendingVoice;

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
            previewBtn.style.color = '#d32f2f';
        }

        const utterance = new SpeechSynthesisUtterance('This is my voice.');
        utterance.text = 'This is my voice.';
        utterance.voice = selectedVoice;
        utterance.rate = 1;

        // Show different message for online vs local voices
        const isOnline = !selectedVoice.localService;
        if (isOnline) {
            this.showToast('Testing online voice...');
        }

        const resetButton = () => {
            if (previewBtn) {
                previewBtn.disabled = false;
                previewBtn.textContent = '▶';
                previewBtn.style.color = '';
            }
        };

        utterance.onerror = (event) => {
            const errorCode = event.error || 'unknown error';
            console.error('Voice preview error:', errorCode);
            const errorMessage = isOnline 
                ? `Online voice failed: ${errorCode}. Check internet connection or try a local voice.`
                : `Voice preview failed: ${errorCode}. This voice may not be available.`;
            this.showToast(errorMessage);
            resetButton();
        };

        utterance.onend = () => {
            if (isOnline) {
                this.showToast('Online voice confirmed working!');
            }
            resetButton();
        };

        try {
            speechSynthesis.speak(utterance);
        } catch (error) {
            console.error('Failed to start voice preview:', error);
            this.showToast(`Failed to start voice preview: ${error.message}`);
            resetButton();
        }
    }

    onSpeechComplete() {
        // Messages already added to chat during Processing/Speaking states
        this.pendingUserMessage = null;
        this.pendingAssistantMessage = null;

        // Reset UI to welcome state
        this.resetToWelcomeState();
    }

    resetToWelcomeState() {
        const chatIcon = document.querySelector('.chat-icon');
        const startBtn = document.getElementById('start-btn');
        const cancelBtn = document.getElementById('cancel-btn');
        const ccBtn = document.getElementById('cc-btn');

        if (chatIcon) chatIcon.style.animation = 'none';
        if (startBtn) startBtn.style.display = 'inline-block';
        if (cancelBtn) cancelBtn.style.display = 'none';
        if (ccBtn) ccBtn.style.display = 'none';
        
        // Always show all messages when conversation ends, regardless of CC state
        if (this.chatMessages) {
            const messages = this.chatMessages.querySelectorAll('.message');
            messages.forEach(msg => {
                msg.classList.remove('hidden');
            });
        }
        
        // Reset captions state to off for next conversation
        this.showCaptions = false;
        this.updateCCButton();
        
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

    updateCCButton() {
        const ccBtn = document.getElementById('cc-btn');
        if (!ccBtn) return;
        
        if (this.showCaptions) {
            ccBtn.innerHTML = '[<s>cc</s>]';
        } else {
            ccBtn.innerHTML = '[cc]';
        }
    }

    toggleCaptions() {
        this.showCaptions = !this.showCaptions;
        this.updateCCButton();
        this.updateMessageVisibility();
    }

    updateMessageVisibility() {
        // Show or hide all messages based on showCaptions state
        if (this.chatMessages) {
            const messages = this.chatMessages.querySelectorAll('.message');
            messages.forEach(msg => {
                if (this.showCaptions) {
                    msg.classList.remove('hidden');
                } else {
                    msg.classList.add('hidden');
                }
            });
        }
    }

    async initializeModel() {
        try {
            console.log('initializeModel called');
            
            // Initialize system message settings
            const initialSystemMessage = 'You are a helpful AI assistant that answers spoken questions with vocalized responses.';
            if (this.systemMessage) {
                this.systemMessage.value = initialSystemMessage;
            }
            this.appliedSystemMessage = initialSystemMessage;
            this.pendingSystemMessage = initialSystemMessage;
            this.currentSystemMessage = initialSystemMessage + ' IMPORTANT: Make your responses brief and to the point.';
            
            // Try to initialize WebLLM first, then fall back to wllama
            await this.initializeEngine();
            
        } catch (error) {
            console.error('Error initializing models:', error);
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOAD_ERROR);
        }
    }

    async initializeEngine() {
        try {
            console.log('Attempting to initialize WebLLM first...');
            await this.initializeWebLLM();
            console.log('WebLLM initialized successfully');
            this.webllmAvailable = true;
            this.usingWllama = false;
        } catch (error) {
            console.error('WebLLM initialization failed, loading wllama fallback:', error);
            this.webllmAvailable = false;
            
            try {
                await this.initializeWllama();
                console.log('Wllama initialized successfully as fallback');
                this.usingWllama = true;
                this.wllamaLoaded = true;
            } catch (wllamaError) {
                console.error('Both WebLLM and wllama initialization failed:', wllamaError);
                this.updateProgress(
                    'progressContainer',
                    'progressFill',
                    'progressText',
                    0,
                    'AI models unavailable. Please check your internet connection and refresh the page.'
                );
                setTimeout(() => {
                    this.allowInteraction();
                }, 2000);
            }
        }
    }

    async initializeWebLLM() {
        console.log('initializeWebLLM called - starting model initialization');
        
        // Check if WebLLM is available
        if (!webllm || !webllm.CreateMLCEngine || !webllm.prebuiltAppConfig) {
            console.error('WebLLM not properly loaded');
            throw new Error('WebLLM not properly loaded');
        }

        // Update model select with specific models
        const modelSelect = this.elements.modelSelect;
        
        if (modelSelect) {
            modelSelect.innerHTML = '';

            // Add only the Phi-3 mini model
            const phiOption = document.createElement('option');
            phiOption.value = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
            phiOption.textContent = 'Phi-3-mini (GPU)';
            modelSelect.appendChild(phiOption);
            
            // Add SmolLM2 CPU option
            const cpuOption = document.createElement('option');
            cpuOption.value = 'smollm2-cpu';
            cpuOption.textContent = 'SmolLM2 (CPU)';
            modelSelect.appendChild(cpuOption);
        }

        // Load default model (Phi-3-mini-4k-instruct)
        const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
        if (modelSelect) {
            modelSelect.value = targetModelId;
        }
        this.pendingModelId = targetModelId;
        
        await this.loadModel(targetModelId);
    }

    async initializeWllama(progressCallback) {
        console.log('Initializing wllama...');
        
        const updateProgress = progressCallback || ((loaded, total) => {
            const percentage = Math.round((loaded / total) * 100);
            this.updateProgress(
                'progressContainer',
                'progressFill',
                'progressText',
                percentage,
                `Loading SmolLM2 (CPU): ${percentage}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`,
                true
            );
        });
        
        this.showElement('progressContainer');
        updateProgress(0, 100);
        
        // Configure WASM paths for CDN
        const CONFIG_PATHS = {
            'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm',
            'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm',
        };
        
        // Initialize wllama with CDN-hosted WASM files
        this.wllama = new Wllama(CONFIG_PATHS);
        
        // Load SmolLM2 model from HuggingFace
        await this.wllama.loadModelFromHF(
            'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
            'smollm2-360m-instruct-q8_0.gguf',
            {
                n_ctx: 1024,
                n_threads: navigator.hardwareConcurrency || 4,
                progressCallback: ({ loaded, total }) => {
                    updateProgress(loaded, total);
                }
            }
        );
        
        console.log('Wllama initialized successfully');
        this.wllamaLoaded = true;
        
        // Update model select if not already populated
        const modelSelect = this.elements.modelSelect;
        if (modelSelect && modelSelect.options.length === 0) {
            modelSelect.innerHTML = '';
            
            // Add Phi-3 option (but disabled if WebLLM failed)
            const phiOption = document.createElement('option');
            phiOption.value = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
            phiOption.textContent = 'Phi-3-mini (GPU)';
            phiOption.disabled = !this.webllmAvailable;
            modelSelect.appendChild(phiOption);
            
            // Add SmolLM2 CPU option (selected by default)
            const cpuOption = document.createElement('option');
            cpuOption.value = 'smollm2-cpu';
            cpuOption.textContent = 'SmolLM2 (CPU)';
            cpuOption.selected = true;
            modelSelect.appendChild(cpuOption);
            
            this.pendingModelId = 'smollm2-cpu';
            this.appliedModelId = 'smollm2-cpu';
        }
        
        this.updateProgress(
            'progressContainer',
            'progressFill',
            'progressText',
            100,
            'SmolLM2 (CPU) ready!'
        );
        setTimeout(() => {
            this.hideElement('progressContainer');
            this.allowInteraction();
        }, 1000);
    }

    async loadModel(modelId) {
        if (!modelId || modelId === 'smollm2-cpu') return;

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
            this.usingWllama = false;
            this.hideElement('progressContainer');
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOADED);
            this.allowInteraction();
        } catch (error) {
            console.error(`Error loading model ${modelId}:`, error);
            this.webllmAvailable = false;
            this.engine = null;
            this.showToast(ChatPlayground.MESSAGES.TOAST.MODEL_LOAD_FALLBACK);
            // Hide progress indicator
            this.hideElement('progressContainer');
            // Try loading wllama fallback
            try {
                await this.initializeWllama();
                this.usingWllama = true;
                this.wllamaLoaded = true;
                
                // Update dropdown - disable Phi-3 option and select SmolLM2
                const modelSelect = this.elements.modelSelect;
                if (modelSelect) {
                    // Find and disable the Phi-3 option
                    for (let option of modelSelect.options) {
                        if (option.value === 'Phi-3-mini-4k-instruct-q4f16_1-MLC') {
                            option.disabled = true;
                            break;
                        }
                    }
                    // Select SmolLM2
                    modelSelect.value = 'smollm2-cpu';
                }
                
                this.appliedModelId = 'smollm2-cpu';
                this.allowInteraction();
            } catch (wllamaError) {
                console.error('Failed to load wllama fallback:', wllamaError);
                // Allow interaction even if both fail
                this.allowInteraction();
            }
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
        // Validate and sanitize input
        let text = input.value.trim();
        
        // Enforce maximum length
        if (text.length > 1000) {
            text = text.substring(0, 1000);
        }
        
        // Basic sanitization - remove control characters except newlines and tabs
        text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
        
        if (text.length > 0) {
            window.chatPlaygroundApp.handleSpokenInput(text);
            input.value = '';
            window.closeSpeechErrorModal();
        }
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
            
            // Clear pending messages (they're already in chat or not yet added)
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
