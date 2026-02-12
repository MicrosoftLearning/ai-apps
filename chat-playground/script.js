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
        this.webllmAvailable = false; // Track if WebLLM model successfully loaded
        this.conversationHistory = [];
        this.isGenerating = false;
        this.stopRequested = false;
        this.currentStream = null; // Track current streaming completion
        this.currentAbortController = null; // Track abort controller for wllama
        this.typingState = null;
        this.currentSystemMessage = "You are an AI assistant that helps people find information.";
        this.currentModelId = null;

        // Configuration objects
        this.config = {
            modelParameters: {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 1000,
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
        }
    };

    // Centralized initialization
    initialize() {
        this.initializeElements();
        this.attachEventListeners();
        this.initializeParameterControls();
        this.initializeFileUpload();
        this.setupImageAnalysisToggle();
        this.initializeModel();
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
        if (this.usingWllama) {
            // SmolLM2 (CPU mode) - Lower temperature for consistency
            return {
                temperature: 0.3,
                top_p: 0.7,
                max_tokens: 1000,
                repetition_penalty: 1.1
            };
        } else {
            // Phi-3 (GPU mode) - Standard defaults
            return {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 1000,
                repetition_penalty: 1.1
            };
        }
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
        // Centralized parameter configuration
        this.parameterConfig = [
            { 
                id: 'temperature-slider', 
                valueId: 'temperature-value', 
                param: 'temperature',
                type: 'float',
                displayName: 'Temperature'
            },
            { 
                id: 'top-p-slider', 
                valueId: 'top-p-value', 
                param: 'top_p',
                type: 'float',
                displayName: 'Top P'
            },
            { 
                id: 'max-tokens-slider', 
                valueId: 'max-tokens-value', 
                param: 'max_tokens',
                type: 'int',
                displayName: 'Max Tokens'
            },
            { 
                id: 'repetition-penalty-slider', 
                valueId: 'repetition-penalty-value', 
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
        this.setElementText('fileName', file.name);
        this.setElementText('fileSize', `${(file.size / 1024).toFixed(1)}KB`);
        this.setElementStyle('fileInfo', 'display', 'flex');
        this.hideElement('addDataBtn');
    }
    
    removeFile() {
        // Clear file data
        this.config.fileUpload.content = null;
        this.config.fileUpload.fileName = null;
        
        // Update UI using utility functions
        this.hideElement('fileInfo');
        this.setElementProperty('fileInput', 'value', '');
        this.showElement('addDataBtn');
        
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
        const newChatBtn = document.querySelector('.chat-controls .icon-btn:not(.help-btn)');
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
        
        // Model selection change
        this.modelSelect.addEventListener('change', () => this.handleModelChange());
    }
    
    async initializeModel() {
        try {
            await this.initializeEngine();
        } catch (error) {
            console.error('Failed to initialize AI engine:', error);
        }
    }
    
    async initializeEngine() {
        try {
            console.log('Attempting to initialize WebLLM first...');
            await this.initializeWebLLM();
            console.log('WebLLM initialized successfully');
            this.webllmAvailable = true;
            this.usingWllama = false;
            
            // Set Phi-3 default parameters
            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
        } catch (error) {
            console.error('WebLLM initialization failed, loading wllama fallback:', error);
            this.webllmAvailable = false;
            
            try {
                await this.initializeWllama();
                console.log('Wllama initialized successfully as fallback');
                this.usingWllama = true;
                this.wllamaLoaded = true;
                
                // Set SmolLM2 default parameters
                this.config.modelParameters = this.getModelDefaults();
                this.updateParameterUI();
            } catch (wllamaError) {
                console.error('Both WebLLM and wllama initialization failed:', wllamaError);
                this.updateProgress(0, 'AI models unavailable. Please check your internet connection and refresh the page.', true);
                setTimeout(() => {
                    this.enableUI();
                }, 2000);
            }
        }
    }
    
    async initializeWebLLM() {
        console.log('initializeWebLLM called - starting model initialization');
        this.updateProgress(0, 'Discovering available models...');
        console.log('Starting WebLLM initialization...');
        console.log('WebLLM object:', webllm);
        console.log('WebLLM.CreateMLCEngine:', typeof webllm?.CreateMLCEngine);
        console.log('WebLLM.prebuiltAppConfig:', typeof webllm?.prebuiltAppConfig);
        
        // Check if WebLLM is available
        if (!webllm || !webllm.CreateMLCEngine || !webllm.prebuiltAppConfig) {
            console.error('WebLLM check failed:', {
                webllm: !!webllm,
                CreateMLCEngine: !!webllm?.CreateMLCEngine,
                prebuiltAppConfig: !!webllm?.prebuiltAppConfig
            });
            throw new Error('WebLLM not properly loaded');
        }
        
        // Get available models from WebLLM
        const models = webllm.prebuiltAppConfig.model_list;
        console.log('All available models:', models.map(m => m.model_id));
        
        // Filter for the specific Phi-3 model only
        const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
        let availableModels = models.filter(model => 
            model.model_id === targetModelId
        );
        
        if (availableModels.length === 0) {
            throw new Error('Phi-3-mini-4k-instruct model not found');
        }
        
        console.log('Available models for loading:', availableModels.map(m => m.model_id));
        
        this.updateProgress(10, 'Loading WebLLM model (GPU mode)...');
        
        // Try to load the first available model
        let engineCreated = false;
        
        for (const model of availableModels) {
            try {
                console.log(`Trying to load model: ${model.model_id}`);
                this.updateProgress(15, `Loading ${model.model_id}...`);
                
                this.engine = await webllm.CreateMLCEngine(
                    model.model_id,
                    {
                        initProgressCallback: (progress) => {
                            console.log('Progress:', progress);
                            const percentage = Math.max(15, Math.round(progress.progress * 85) + 15);
                            this.updateProgress(percentage, `Loading ${model.model_id}: ${Math.round(progress.progress * 100)}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`, true);
                        }
                    }
                );
                
                console.log(`Successfully loaded model: ${model.model_id}`);
                this.currentModelId = model.model_id;
                engineCreated = true;
                break;
                
            } catch (modelError) {
                console.error(`Failed to load ${model.model_id}:`, modelError);
                continue;
            }
        }
        
        if (!engineCreated) {
            throw new Error('Failed to load any available models. Please check your internet connection and try again.');
        }
        
        console.log('WebLLM engine created successfully');
        this.updateProgress(100, 'Model ready! (GPU mode)');
        setTimeout(() => {
            this.progressContainer.style.display = 'none';
            this.enableUI();
        }, 1000);
    }
    
    async initializeWllama(progressCallback) {
        console.log('Initializing wllama...');
        
        const updateProgress = progressCallback || ((loaded, total) => {
            const percentage = Math.round((loaded / total) * 100);
            this.updateProgress(percentage, `Loading wllama model (CPU mode): ${percentage}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`, true);
        });
        
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
                n_ctx: 2048,
                n_threads: navigator.hardwareConcurrency || 4,
                progressCallback: ({ loaded, total }) => {
                    updateProgress(loaded, total);
                }
            }
        );
        
        console.log('Wllama initialized successfully');
        this.wllamaLoaded = true;
        this.updateProgress(100, 'CPU model ready!');
        setTimeout(() => {
            this.progressContainer.style.display = 'none';
            this.enableUI();
        }, 1000);
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
        this.userInput.focus();
        
        // Populate model dropdown with available models
        this.populateModelDropdown();
        
        // Set parameter controls based on whether WebLLM is available
        this.setParameterControlsEnabled(this.webllmAvailable || this.wllamaLoaded);
    }
    
    disableUI() {
        this.isModelLoaded = false;
        this.systemMessage.disabled = true;
        this.userInput.disabled = true;
        this.sendBtn.disabled = true;
        this.attachBtn.disabled = true;
    }
    
    async handleModelChange() {
        const selectedValue = this.modelSelect.value;
        
        if (this.isGenerating) {
            console.log('Cannot switch models while generating');
            // Reset to current model
            this.populateModelDropdown();
            return;
        }
        
        // Determine if we're actually switching models
        const previousMode = this.usingWllama;
        const newModeIsWllama = selectedValue === 'phi3-cpu';
        
        if (selectedValue === 'phi3-gpu') {
            if (!this.webllmAvailable) {
                alert('Phi-3 (GPU) is not available. WebGPU is not supported on this device.');
                this.populateModelDropdown(); // Reset selection
                return;
            }
            
            // Clear wllama KV cache if switching from CPU mode
            if (previousMode && this.wllama) {
                await this.wllama.kvClear();
                console.log('Cleared wllama KV cache when switching to GPU mode');
            }
            
            this.usingWllama = false;
            
            // Apply Phi-3 default parameters
            this.config.modelParameters = this.getModelDefaults();
            this.updateParameterUI();
            
            // Clear chat and restart conversation
            if (previousMode !== this.usingWllama) {
                await this.clearChat();
                this.showToast('Switched to Phi-3 (GPU) - Conversation restarted');
            }
            
            console.log('Switched to Phi-3 (GPU) mode');
        } else if (selectedValue === 'phi3-cpu') {
            // Keep WebLLM engine loaded (it uses GPU memory, wllama uses system RAM)
            // If wllama not loaded yet, load it
            if (!this.wllamaLoaded) {
                console.log('Loading wllama for the first time...');
                
                // Disable UI during model loading
                this.disableUI();
                
                // Show progress
                this.progressContainer.style.display = 'block';
                
                try {
                    await this.initializeWllama((loaded, total) => {
                        const percentage = Math.round((loaded / total) * 100);
                        this.updateProgress(percentage, `Loading SmolLM2 (CPU): ${percentage}%<br><small style="font-size: 0.9em; color: #666;">(First-time download may take a few minutes)</small>`, true);
                    });
                    
                    this.usingWllama = true;
                    
                    // Apply SmolLM2 default parameters
                    this.config.modelParameters = this.getModelDefaults();
                    this.updateParameterUI();
                    
                    // Clear chat and restart conversation
                    await this.clearChat();
                    this.showToast('Switched to SmolLM2 (CPU) - Conversation restarted');
                    
                    console.log('Switched to SmolLM2 (CPU) mode');
                } catch (error) {
                    console.error('Failed to load wllama:', error);
                    this.populateModelDropdown(); // Reset to previous selection
                    alert('Failed to load SmolLM2 (CPU). Please try again.');
                    // Re-enable UI even on error
                    this.enableUI();
                }
            } else {
                this.usingWllama = true;
                
                // Apply SmolLM2 default parameters
                this.config.modelParameters = this.getModelDefaults();
                this.updateParameterUI();
                
                // Clear chat and restart conversation
                if (previousMode !== this.usingWllama) {
                    await this.clearChat();
                    this.showToast('Switched to SmolLM2 (CPU) - Conversation restarted');
                }
                
                console.log('Switched to SmolLM2 (CPU) mode');
            }
        }
    }
    
    populateModelDropdown() {
        // Clear existing options
        this.modelSelect.innerHTML = '';
        
        // Add Phi-3 (GPU) option
        const phiOption = document.createElement('option');
        phiOption.value = 'phi3-gpu';
        phiOption.textContent = 'Phi-3-mini (GPU)';
        phiOption.disabled = !this.webllmAvailable;
        if (this.webllmAvailable && !this.usingWllama) {
            phiOption.selected = true;
        }
        this.modelSelect.appendChild(phiOption);
        
        // Add SmolLM2 (CPU) option
        const cpuOption = document.createElement('option');
        cpuOption.value = 'phi3-cpu';
        cpuOption.textContent = 'SmolLM2 (CPU)';
        if (this.usingWllama || !this.webllmAvailable) {
            cpuOption.selected = true;
        }
        this.modelSelect.appendChild(cpuOption);
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
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how',
            'do', 'does', 'did', 'can', 'could', 'would', 'should', 'may', 'might',
            'this', 'these', 'those', 'i', 'you', 'we', 'they', 'my', 'your',
            'am', 'been', 'being', 'have', 'had', 'were', 'there', 'their'
        ]);
        
        // Extract words, convert to lowercase, filter stopwords and short words
        const words = text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopwords.has(word));
        
        // Return unique keywords
        return [...new Set(words)];
    }
    
    // Extract relevant lines from file content based on keywords
    extractRelevantLines(fileContent, keywords) {
        if (!fileContent || !keywords || keywords.length === 0) {
            return '';
        }
        
        const lines = fileContent.split('\n');
        const matchingLines = [];
        
        for (const line of lines) {
            const lineLower = line.toLowerCase();
            // Check if line contains any keyword
            if (keywords.some(keyword => lineLower.includes(keyword))) {
                matchingLines.push(line.trim());
            }
        }
        
        return matchingLines.length > 0 ? matchingLines.join('\n') : '';
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
            const messages = [
                { role: "system", content: this.getEffectiveSystemMessage() }
            ];
            
            // Add last 10 conversation pairs
            // Remove any previous image classifications from history to avoid confusion
            const recentHistory = this.conversationHistory.slice(-20).map(msg => {
                if (msg.role === 'user') {
                    return {
                        ...msg,
                        content: msg.content.replace(/\n\n\[Current image shows:.*?\]$/s, '')
                    };
                }
                return msg;
            });
            messages.push(...recentHistory);
            
            // Add user message with image analysis and file context if available
            let finalUserMessage = userMessage;
            if (imageAnalysis) {
                finalUserMessage += '\n\n[Current image shows: ' + imageAnalysis + ']';
            }
            
            // If file is uploaded, prepend file content to user message
            if (this.config.fileUpload.content) {
                // For Phi-3 (WebLLM/GPU mode), use entire file content for best accuracy
                console.log('Using entire file content for Phi-3 (WebLLM mode) - ' + this.config.fileUpload.content.split('\n').length + ' lines');
                finalUserMessage = 'Use the following information to answer the question:\n\n' + this.config.fileUpload.content + '\n\nQuestion: ' + userMessage;
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
            if (this.usingWllama) {
                await this.handleWllamaMode(messages, thinkingIndicator, userMessage, imageAnalysis);
            } else {
                await this.handleStreamingMode(messages, thinkingIndicator, userMessage);
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
            
            const errorMessage = 'Sorry, I encountered an error while generating a response. Please try again.';
            const assistantMessageEl = this.addMessage('assistant', '');
            const contentEl = assistantMessageEl.querySelector('.message-content');
            
            // Type out the error message
            await this.typeResponse(contentEl, errorMessage);
        } finally {
            this.isGenerating = false;
            this.updateUIForGeneration(false);
        }
    }

    async handleStreamingMode(messages, thinkingIndicator, userMessage) {
        // Streaming Mode: Type as soon as we have content
        let fullResponse = '';
        let hasStartedOutput = false;
        const bufferSize = 30; // Start typing after 30 characters
        let assistantMessageEl = null;
        let contentEl = null;
        
        const completion = await this.engine.chat.completions.create({
            messages: messages,
            temperature: this.modelParameters.temperature,
            top_p: this.modelParameters.top_p,
            max_tokens: this.modelParameters.max_tokens,
            repetition_penalty: this.modelParameters.repetition_penalty,
            stream: true
        });
        
        for await (const chunk of completion) {
            if (!this.isGenerating) break;
            
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
                fullResponse += content;
                
                // Start output once we have enough content buffered
                if (!hasStartedOutput && fullResponse.length >= bufferSize) {
                    // Remove thinking indicator
                    thinkingIndicator.remove();
                    
                    // Create message container
                    assistantMessageEl = this.addMessage('assistant', '');
                    contentEl = assistantMessageEl.querySelector('.message-content');
                    
                    // Start typing animation
                    this.startTypingAnimation(contentEl, fullResponse);
                    hasStartedOutput = true;
                } else if (hasStartedOutput && contentEl) {
                    // Update the content for ongoing typing animation
                    this.updateTypingContent(fullResponse);
                }
            }
        }
        
        // Append file attribution if a file is uploaded (for display only, after streaming completes)
        let displayResponse = fullResponse;
        if (hasStartedOutput && this.config.fileUpload.fileName && fullResponse.trim()) {
            const attribution = `\n(Ref: ${this.config.fileUpload.fileName})`;
            displayResponse = fullResponse + attribution;
            // Update the typing content to include attribution
            this.updateTypingContent(displayResponse);
        }
        
        // Handle case where response is shorter than buffer size
        if (!hasStartedOutput) {
            // Remove thinking indicator
            thinkingIndicator.remove();
            
            if (fullResponse.trim()) {
                // Append file attribution if a file is uploaded (for display only)
                displayResponse = fullResponse;
                if (this.config.fileUpload.fileName) {
                    displayResponse += `\n(Ref: ${this.config.fileUpload.fileName})`;
                }
                
                // Create message container
                assistantMessageEl = this.addMessage('assistant', '');
                contentEl = assistantMessageEl.querySelector('.message-content');
                
                // Type out the short response
                await this.typeResponse(contentEl, displayResponse);
            } else {
                const fallbackMessage = "I apologize, but I couldn't generate a response. Please try again.";
                assistantMessageEl = this.addMessage('assistant', '');
                contentEl = assistantMessageEl.querySelector('.message-content');
                await this.typeResponse(contentEl, fallbackMessage);
            }
        }
        
        // Add to conversation history (without file attribution, to prevent cumulative citations)
        this.conversationHistory.push({ role: "user", content: userMessage });
        this.conversationHistory.push({ role: "assistant", content: fullResponse });
    }
    
    // Helper function to build ChatML formatted prompt for SmolLM2
    buildChatMLPrompt(userMessage, imageAnalysis = '', fileContent = '') {
        let prompt = '';
        
        // Get the last turn of conversation history (if exists)
        let previousUserMessage = '';
        let previousAssistantResponse = '';
        
        if (this.conversationHistory.length >= 2) {
            // Get the last pair (user message and assistant response)
            previousAssistantResponse = this.conversationHistory[this.conversationHistory.length - 1].content;
            previousUserMessage = this.conversationHistory[this.conversationHistory.length - 2].content;
            // Clean any image classification from previous user message
            previousUserMessage = previousUserMessage.replace(/\n\n\[Current image shows:.*?\]$/s, '');
        }
        
        // Determine which format to use
        if (imageAnalysis) {
            // Format for image analysis
            prompt = '<|im_start|>system\n';
            prompt += 'You are a rules‑driven assistant. Your highest priority is to follow the instructions exactly as written and answer questions based on the information below.\n\n';
            prompt += 'Instructions:\n';
            prompt += this.currentSystemMessage + '\n\n';
            prompt += 'Information:\n';
            prompt += 'The user has uploaded an image containing a ' + imageAnalysis + '. Their question relates to this image.\n\n';
            prompt += 'Acknowledge these rules by answering the user\'s question correctly based on the information above.\n';
            prompt += '<|im_end|>\n\n';
            
            // Add previous user message only (not response) if exists
            if (previousUserMessage) {
                prompt += '<|im_start|>user\n' + previousUserMessage + '\n<|im_end|>\n\n';
            }
            
            // Add current user message
            prompt += '<|im_start|>user\n' + userMessage + '\n<|im_end|>\n\n';
            prompt += '<|im_start|>assistant\n';
            
        } else if (fileContent) {
            // Format for file grounding
            prompt = '<|im_start|>system\n';
            prompt += 'You are a rules‑driven assistant. Your highest priority is to follow the instructions exactly as written, and answer questions based only on the information provided.\n\n';
            prompt += 'Instructions:\n';
            prompt += this.currentSystemMessage + '\n\n';
            prompt += 'IMPORTANT: You must answer the user\'s specific question concisely, based only on the following information.\n\n';
            prompt += 'Information:\n';
            prompt += fileContent + '\n\n';
            prompt += 'Base your answer on the information above ONLY. Do NOT include any details that are not present in the information above.\n\n';
            prompt += '<|im_end|>\n\n';
            
            // Add current user message
            prompt += '<|im_start|>user\n' + userMessage + '\n<|im_end|>\n\n';
            prompt += '<|im_start|>assistant\n';
            
        } else {
            // Default format (no file grounding, no image)
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
        }
        
        return prompt;
    }
    
    async handleWllamaMode(messages, thinkingIndicator, userMessage, imageAnalysis = '') {
        // Ensure wllama is loaded
        if (!this.wllama) {
            throw new Error('Wllama is not initialized. Please wait for CPU mode to finish loading.');
        }
        
        // Keep original userMessage for conversation history (without image classification)
        const originalUserMessage = userMessage;
        
        // Remove thinking indicator before starting to stream
        thinkingIndicator.remove();
        
        // Create message container
        const assistantMessageEl = this.addMessage('assistant', '');
        const contentEl = assistantMessageEl.querySelector('.message-content');
        
        // Show thinking indicator with CPU mode notice
        contentEl.innerHTML = '<span class="typing-indicator">●●●</span><p style="font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic;">(Responses may be slow in CPU mode. Thanks for your patience!)</p>';
        
        // Build ChatML formatted prompt
        let fileContentForPrompt = '';
        
        // If file is uploaded, extract relevant lines
        if (this.config.fileUpload.content) {
            const keywords = this.extractKeywords(userMessage);
            console.log('Extracted keywords from user prompt (wllama):', keywords);
            
            const relevantLines = this.extractRelevantLines(this.config.fileUpload.content, keywords);
            
            if (relevantLines) {
                console.log('Found relevant lines from file (' + relevantLines.split('\n').length + ' lines)');
                fileContentForPrompt = relevantLines;
            } else {
                console.log('No relevant lines found in file for the given keywords');
                fileContentForPrompt = this.config.fileUpload.content;
            }
        }
        
        // Build the ChatML prompt
        const chatMLPrompt = this.buildChatMLPrompt(userMessage, imageAnalysis, fileContentForPrompt);
        
        console.log('=== CHATML PROMPT FOR SMOLLM2 ===');
        console.log('Conversation history length:', this.conversationHistory.length);
        console.log('File content included:', !!fileContentForPrompt);
        console.log('Image analysis included:', !!imageAnalysis);
        console.log('ChatML prompt:');
        console.log(chatMLPrompt);
        console.log('=== END CHATML PROMPT ===');
        
        // Use wllama for generation with streaming
        let fullResponse = '';
        
        // Log current model parameters from config
        console.log('Current model parameters from config:', this.config.modelParameters);
        
        // Use parameters from config (set when model is selected)
        // Clamp temperature for wllama (supports range 0-2, but works best between 0.1-1.5)
        const wllamaTemp = Math.max(0.1, Math.min(1.5, this.config.modelParameters.temperature));
        const wllamaTopP = Math.max(0.1, Math.min(1.0, this.config.modelParameters.top_p));
        const wllamaPenalty = Math.max(1.0, Math.min(2.0, this.config.modelParameters.repetition_penalty));
        
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
        
        try {
            const completion = await this.wllama.createCompletion(chatMLPrompt, {
                nPredict: 300,  // SmolLM2 has 2048 context window
                seed: -1,  // Random seed for variation
                sampling: {
                    temp: wllamaTemp,
                    top_k: 40,
                    top_p: wllamaTopP,
                    penalty_repeat: wllamaPenalty,
                    mirostat: 0  // Disable mirostat to ensure temperature is used
                },
                stopTokens: ['<|im_end|>', '<|im_start|>'],
                abortSignal: controller.signal,
                stream: true
            });
            
            this.currentStream = completion;
            
            for await (const chunk of completion) {
                if (chunk.currentText) {
                    fullResponse = chunk.currentText;
                    contentEl.textContent = fullResponse;
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }
            }
            
            // Clear abort controller on successful completion
            this.currentAbortController = null;
            
            // Clear KV cache after successful generation
            console.log('Clearing KV cache after generation');
            await this.wllama.kvClear();
            console.log('KV cache cleared successfully');
            
            // Always add to conversation history to maintain context
            // BUT: Do NOT add stopped responses to history (they're incomplete/corrupted)
            if (fullResponse.trim() && !this.stopRequested) {
                // Append file attribution if a file is uploaded
                let displayResponse = fullResponse;
                if (this.config.fileUpload.fileName) {
                    displayResponse = fullResponse + `\n(Ref: ${this.config.fileUpload.fileName})`;
                }
                
                // Add indicator if stopped
                if (this.stopRequested) {
                    displayResponse += '\n\n[Response stopped by user]';
                }
                
                contentEl.textContent = displayResponse;
                
                // Add to conversation history (without file attribution or stop indicator)
                // Use original message without image classification to avoid persisting it
                this.conversationHistory.push({ role: "user", content: originalUserMessage });
                this.conversationHistory.push({ role: "assistant", content: fullResponse });
            } else if (this.stopRequested && fullResponse.trim()) {
                // Response was stopped - display it but don't add to history
                let displayResponse = fullResponse;
                if (this.config.fileUpload.fileName) {
                    displayResponse = fullResponse + `\n(Ref: ${this.config.fileUpload.fileName})`;
                }
                displayResponse += '\n\n[Response stopped by user - not saved to history]';
                contentEl.textContent = displayResponse;
                
                console.log('Stopped response not added to conversation history to prevent corruption');
            } else {
                contentEl.textContent = 'Sorry, I encountered an error while generating a response. Please try again.';
            }
            
        } catch (error) {
            // Check if this was an abort (expected when user clicks stop)
            if (error.name === 'AbortError' || error.message?.includes('abort')) {
                console.log('Generation aborted by user');
                // Clear the partial/corrupted state
                await this.wllama.kvClear();
                console.log('KV cache cleared after abort');
                
                // Display stopped response but don't add to history
                if (fullResponse.trim()) {
                    let displayResponse = fullResponse;
                    if (this.config.fileUpload.fileName) {
                        displayResponse = fullResponse + `\n(Ref: ${this.config.fileUpload.fileName})`;
                    }
                    displayResponse += '\n\n[Response stopped by user - not saved to history]';
                    contentEl.textContent = displayResponse;
                    console.log('Stopped response not added to conversation history to prevent corruption');
                }
            } else {
                console.error('Error in wllama generation:', error);
                contentEl.textContent = 'Sorry, I encountered an error while generating a response. Please try again.';
                // Clear cache on error too
                try {
                    await this.wllama.kvClear();
                } catch (e) {
                    console.log('Failed to clear cache after error:', e.message);
                }
            }
            this.currentAbortController = null;
        }
    }

    addThinkingIndicator() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'thinking-indicator';
        thinkingDiv.innerHTML = `
            <div class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.chatMessages.appendChild(thinkingDiv);
        
        // Auto-scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        return thinkingDiv;
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
            typingSpeed:5
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
        // Hide welcome message if it exists
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
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
        this.isGenerating = false;
        this.stopRequested = true;
        
        // Stop typing animation
        if (this.typingState) {
            this.typingState.isTyping = false;
        }
        
        // Abort wllama generation properly using AbortController
        if (this.currentAbortController) {
            console.log('Aborting wllama generation via AbortController');
            this.currentAbortController.abort();
            this.currentAbortController = null;
        }
        
        // Clear current stream reference
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
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="chat-icon">💬</div>
                <h3>What do you want to chat about?</h3>
            </div>
        `;
        
        // Clear wllama KV cache when resetting chat to start fresh
        if (this.usingWllama && this.wllama) {
            try {
                console.log('Chat reset: Clearing wllama KV cache...');
                await this.wllama.kvClear();
                console.log('Chat reset: KV cache cleared - ready for fresh start');
            } catch (error) {
                console.error('Error clearing wllama KV cache:', error);
            }
        }
    }
    
    // Removed updateTokenCount function - disclaimer is now static
    
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
window.toggleSection = function(sectionId) {
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

window.resetParameters = function() {
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

window.triggerFileUpload = function() {
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.click();
    }
};

window.removeFile = function() {
    if (window.chatPlaygroundApp) {
        window.chatPlaygroundApp.removeFile();
    }
};

window.openAboutModal = function() {
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

window.closeAboutModal = function() {
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
window.openParametersModal = function() {
    const modal = document.getElementById('parameters-modal');
    if (modal) {
        modal.style.display = 'flex';
        // Sync modal sliders with current values from left pane
        syncParametersToModal();
        // Add click-outside-to-close
        modal.addEventListener('click', handleParametersModalClick);
    }
};

window.closeParametersModal = function() {
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
    // Get values from the left pane sliders
    const sourceIds = [
        { source: 'temperature-slider', target: 'modal-temperature-slider', value: 'modal-temperature-value' },
        { source: 'top-p-slider', target: 'modal-top-p-slider', value: 'modal-top-p-value' },
        { source: 'max-tokens-slider', target: 'modal-max-tokens-slider', value: 'modal-max-tokens-value' },
        { source: 'repetition-penalty-slider', target: 'modal-repetition-penalty-slider', value: 'modal-repetition-penalty-value' }
    ];
    
    sourceIds.forEach(({ source, target, value }) => {
        const sourceEl = document.getElementById(source);
        const targetEl = document.getElementById(target);
        const valueEl = document.getElementById(value);
        
        if (sourceEl && targetEl && valueEl) {
            const currentValue = sourceEl.value;
            targetEl.value = currentValue;
            valueEl.textContent = currentValue;
            targetEl.setAttribute('aria-valuetext', currentValue);
        }
    });
    
    // Add event listeners to modal sliders
    ['modal-temperature-slider', 'modal-top-p-slider', 'modal-max-tokens-slider', 'modal-repetition-penalty-slider'].forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        if (slider) {
            slider.addEventListener('input', handleModalParameterChange);
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

window.resetParametersFromModal = function() {
    // Get model-specific defaults
    const defaults = window.chatPlaygroundApp ? window.chatPlaygroundApp.getModelDefaults() : {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 1000,
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
window.trapFocus = function(modal) {
    const focusableElements = modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];
    
    window.modalKeydownHandler = function(e) {
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

window.removeFocusTrap = function() {
    if (window.modalKeydownHandler) {
        document.removeEventListener('keydown', window.modalKeydownHandler);
        window.modalKeydownHandler = null;
    }
};

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatPlaygroundApp = new ChatPlayground();
    
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
            const parametersModal = document.getElementById('parameters-modal');
            if (parametersModal && parametersModal.style.display !== 'none') {
                window.closeParametersModal();
            }
        }
    });
});