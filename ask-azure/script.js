class AskAndrew {
    constructor() {
        this.conversationHistory = [];
        this.isGenerating = false;
        this.indexData = null; // Contains the category structure from index.json
        this.currentModal = null;
        this.lastFocusedElement = null;
        this.modalFocusTrapHandler = null;
        this.usedVoiceInput = false;
        this.isConfigured = false;
        this.config = {
            endpoint: '',
            apiKey: '',
            deployment: '',
            region: ''
        };
        this.previousResponseId = null;
        this.recognition = null;
        this.isListening = false;
        this.ttsToken = null;
        this.ttsTokenExpiry = null;
        this.recordingTimeout = null;
        
        this.elements = {
            chatContainer: document.getElementById('chat-container'),
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            micBtn: document.getElementById('mic-btn'),
            restartBtn: document.getElementById('restart-btn'),
            searchStatus: document.getElementById('search-status'),
            configBtn: document.getElementById('config-btn'),
            aboutBtn: document.getElementById('about-btn'),
            aboutModal: document.getElementById('about-modal'),
            aboutModalClose: document.getElementById('about-modal-close'),
            aboutModalOk: document.getElementById('about-modal-ok'),
            configModal: document.getElementById('config-modal'),
            configModalClose: document.getElementById('config-modal-close'),
            configCancel: document.getElementById('config-cancel'),
            configSave: document.getElementById('config-save'),
            foundryEndpoint: document.getElementById('foundry-endpoint'),
            foundryKey: document.getElementById('foundry-key'),
            foundryDeployment: document.getElementById('foundry-deployment'),
            foundryRegion: document.getElementById('foundry-region'),
            configStatus: document.getElementById('config-status')
        };
        
        this.systemPrompt = `You are Andrew, a knowledgeable and friendly AI learning assistant who helps students understand AI concepts.

IMPORTANT: Follow these guidelines when responding:
- Explain concepts clearly and concisely in a single paragraph based only on the provided context.
- Keep responses short and focused on the question, with no headings.
- Use examples and analogies when helpful.
- Use simple language suitable for learners in a conversational, friendly tone.
- Provide a general descriptions and overviews, but do NOT provide explicit steps or instructions for developing AI solutions.
- Do not start responses with "A:" or "Q:".
- Keep your responses concise and to the point.`;

        this.initialize();
    }

    async initialize() {
        try {
            // Load configuration
            this.loadConfig();
            
            // Load the index
            await this.loadIndex();
            
            // Initialize speech recognition
            this.initializeSpeechRecognition();
            
            // Update UI state again after speech recognition is initialized
            this.updateUIState();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Show chat interface
            this.showChatInterface();
            
        } catch (error) {
            console.error('Initialization error:', error);
            alert('Failed to initialize. Please refresh the page.');
        }
    }

    loadConfig() {
        const saved = localStorage.getItem('askAndrewFoundryConfig');
        if (saved) {
            try {
                const savedConfig = JSON.parse(saved);
                // Load non-sensitive config from storage
                this.config.endpoint = savedConfig.endpoint || '';
                this.config.deployment = savedConfig.deployment || '';
                this.config.region = savedConfig.region || '';
                // Note: apiKey is NOT loaded from storage for security reasons
                
                this.elements.foundryEndpoint.value = this.config.endpoint;
                this.elements.foundryKey.value = ''; // API key must be re-entered each session
                this.elements.foundryDeployment.value = this.config.deployment;
                this.elements.foundryRegion.value = this.config.region;
                
                // Only mark as configured if API key is present in memory
                if (this.config.endpoint && this.config.apiKey && this.config.deployment) {
                    this.isConfigured = true;
                }
            } catch (e) {
                console.error('Error loading config:', e);
            }
        }
        this.updateUIState();
    }

    saveConfig() {
        const endpoint = this.elements.foundryEndpoint.value.trim();
        const apiKey = this.elements.foundryKey.value.trim();
        const deployment = this.elements.foundryDeployment.value.trim();
        const region = this.elements.foundryRegion.value.trim();
        const statusDiv = this.elements.configStatus;
        
        if (!endpoint || !apiKey || !deployment) {
            statusDiv.textContent = 'Please fill in all fields';
            statusDiv.className = 'config-status error';
            return;
        }
        
        // Validate endpoint format and extract base URL
        let baseEndpoint = endpoint;
        try {
            const url = new URL(endpoint);
            if (url.protocol !== 'https:') {
                statusDiv.textContent = 'Endpoint must use HTTPS';
                statusDiv.className = 'config-status error';
                return;
            }
            
            // Extract substring from beginning to first ".com" (inclusive)
            const comIndex = endpoint.indexOf('.com');
            if (comIndex !== -1) {
                baseEndpoint = endpoint.substring(0, comIndex + 4); // +4 to include ".com"
            }
        } catch (e) {
            statusDiv.textContent = 'Invalid endpoint URL';
            statusDiv.className = 'config-status error';
            return;
        }
        
        this.config.endpoint = baseEndpoint;
        this.config.apiKey = apiKey;
        this.config.deployment = deployment;
        this.config.region = region;
        
        // Save non-sensitive config to localStorage (do not persist apiKey)
        const configToPersist = {
            endpoint: this.config.endpoint,
            deployment: this.config.deployment,
            region: this.config.region
        };
        localStorage.setItem('askAndrewFoundryConfig', JSON.stringify(configToPersist));
        
        this.isConfigured = true;
        statusDiv.textContent = 'Configuration saved successfully!';
        statusDiv.className = 'config-status success';
        
        this.updateUIState();
        
        // Auto-close modal after short delay
        setTimeout(() => {
            this.hideConfigModal();
        }, 1500);
    }

    updateUIState() {
        // Mic button should be enabled if speech is supported, regardless of config
        this.elements.micBtn.disabled = !this.recognition;
        
        // Enable input and send button regardless of config state
        // The sendMessage() function will handle showing a message if not configured
        this.elements.userInput.disabled = false;
        this.elements.sendBtn.disabled = false;
        
        if (this.isConfigured) {
            this.elements.restartBtn.disabled = false;
        } else {
            this.elements.restartBtn.disabled = true;
        }
    }

    async loadIndex() {
        try {
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

    showChatInterface() {
        this.elements.chatContainer.style.display = 'flex';
        this.elements.userInput.focus();
    }

    initializeSpeechRecognition() {
        // Azure Speech API will be used - check for microphone support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn('Microphone access not supported in this browser');
            this.recognition = null;
            return;
        }
        
        this.recognition = true; // Flag to indicate speech is available
        this.mediaRecorder = null;
        this.audioChunks = [];
    }

    stopListening() {
        this.isListening = false;
        this.elements.micBtn.classList.remove('listening');
        this.elements.micBtn.title = 'Voice input';
        this.elements.micBtn.setAttribute('aria-pressed', 'false');
        
        // Clear the auto-stop timeout
        if (this.recordingTimeout) {
            clearTimeout(this.recordingTimeout);
            this.recordingTimeout = null;
        }
        
        // Stop recording if active
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    async startAzureSpeechRecognition() {
        if (!this.isConfigured) {
            this.addMessage('assistant', 'Please configure your Foundry settings to chat.');
            return;
        }
        
        if (!this.config.region) {
            this.addMessage('assistant', 'Please add the Azure region to your configuration for speech-to-text to work.');
            return;
        }

        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000
                } 
            });
            
            this.audioChunks = [];
            
            // Use audio/webm;codecs=opus or check available types
            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
            }
            
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = async () => {
                // Stop all tracks to release the microphone
                stream.getTracks().forEach(track => track.stop());
                
                this.stopListening();
                
                // Clear the auto-stop timeout
                if (this.recordingTimeout) {
                    clearTimeout(this.recordingTimeout);
                    this.recordingTimeout = null;
                }
                
                if (this.audioChunks.length === 0) {
                    return;
                }
                
                try {
                    const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                    
                    // Convert to WAV format for Azure Speech API
                    const wavBlob = await this.convertToWav(audioBlob);
                    
                    // Call Azure Speech API
                    await this.transcribeAudio(wavBlob);
                } catch (error) {
                    console.error('Error processing audio:', error);
                    alert('Failed to process audio. Please try again.');
                }
            };
            
            // Request data in chunks for better compatibility
            this.mediaRecorder.start(100);
            this.isListening = true;
            this.elements.micBtn.classList.add('listening');
            this.elements.micBtn.title = 'Listening... (auto-stops in 5s or click to stop)';
            this.elements.micBtn.setAttribute('aria-pressed', 'true');
            
            // Auto-stop recording after 5 seconds
            this.recordingTimeout = setTimeout(() => {
                if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                }
            }, 5000); // 5 seconds
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions.');
            this.stopListening();
        }
    }

    async convertToWav(audioBlob) {
        try {
            // Create an audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Read the blob as array buffer
            const arrayBuffer = await audioBlob.arrayBuffer();
            
            // Decode the audio data
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV format
            const wavBuffer = this.audioBufferToWav(audioBuffer);
            
            // Create a blob from the WAV buffer
            const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            
            return wavBlob;
        } catch (error) {
            console.error('Error converting to WAV:', error);
            throw new Error('Failed to convert audio to WAV format');
        }
    }

    audioBufferToWav(audioBuffer) {
        const numChannels = 1; // Force mono
        const sampleRate = 16000; // Force 16kHz for Azure Speech
        const format = 1; // PCM
        const bitsPerSample = 16;
        
        // Resample if needed
        let samples;
        if (audioBuffer.sampleRate !== sampleRate) {
            samples = this.resampleAudio(audioBuffer, sampleRate);
        } else {
            samples = audioBuffer.getChannelData(0);
        }
        
        const numSamples = samples.length;
        const buffer = new ArrayBuffer(44 + numSamples * 2);
        const view = new DataView(buffer);
        
        // Write WAV header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + numSamples * 2, true);
        this.writeString(view, 8, 'WAVE');
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
        view.setUint16(32, numChannels * bitsPerSample / 8, true);
        view.setUint16(34, bitsPerSample, true);
        this.writeString(view, 36, 'data');
        view.setUint32(40, numSamples * 2, true);
        
        // Write PCM samples
        let offset = 44;
        for (let i = 0; i < numSamples; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return buffer;
    }

    resampleAudio(audioBuffer, targetSampleRate) {
        const sourceRate = audioBuffer.sampleRate;
        const sourceData = audioBuffer.getChannelData(0);
        const sourceLength = sourceData.length;
        const targetLength = Math.round(sourceLength * targetSampleRate / sourceRate);
        const resampledData = new Float32Array(targetLength);
        
        for (let i = 0; i < targetLength; i++) {
            const sourceIndex = i * sourceRate / targetSampleRate;
            const index = Math.floor(sourceIndex);
            const fraction = sourceIndex - index;
            
            if (index + 1 < sourceLength) {
                resampledData[i] = sourceData[index] * (1 - fraction) + sourceData[index + 1] * fraction;
            } else {
                resampledData[i] = sourceData[index];
            }
        }
        
        return resampledData;
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    async transcribeAudio(audioBlob) {
        try {
            // Construct Azure Speech endpoint using the region
            const speechEndpoint = `https://${this.config.region}.stt.speech.microsoft.com`;
            const url = `${speechEndpoint}/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed`;
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Ocp-Apim-Subscription-Key': this.config.apiKey,
                    'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                    'Accept': 'application/json'
                },
                body: audioBlob
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Speech API error:', errorText);
                throw new Error(`Transcription failed: ${response.status} - ${errorText}`);
            }
            
            const data = await response.json();
            
            // Extract transcript from response
            let transcript = '';
            
            // Check different possible response formats
            if (data.DisplayText) {
                transcript = data.DisplayText;
            } else if (data.NBest && data.NBest.length > 0) {
                transcript = data.NBest[0].Display || data.NBest[0].Lexical;
            } else if (data.RecognitionStatus === 'Success' && data.Offset !== undefined) {
                // Check if we have detailed results
                if (data.NBest && data.NBest.length > 0 && data.NBest[0].Lexical) {
                    transcript = data.NBest[0].Display || data.NBest[0].Lexical;
                }
            }
            
            // Trim and check for empty transcript
            transcript = transcript.trim();
            
            if (transcript) {
                this.elements.userInput.value = transcript;
                this.usedVoiceInput = true;
                this.autoResizeTextarea();
                this.elements.userInput.focus();
                
                // Automatically submit the message
                setTimeout(() => {
                    this.sendMessage();
                }, 100);
            } else {
                // Provide detailed feedback
                let errorMsg = 'No speech detected.';
                if (data.RecognitionStatus) {
                    errorMsg += ` Status: ${data.RecognitionStatus}.`;
                }
                if (data.Offset !== undefined && data.Duration !== undefined) {
                    errorMsg += ` Audio received (${data.Duration / 10000000}s).`;
                }
                errorMsg += ' Please speak louder or try again.';
                
                alert(errorMsg);
            }
            
        } catch (error) {
            console.error('Error transcribing audio:', error);
            alert('Failed to transcribe audio: ' + error.message);
        }
    }

    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => {
            this.sendMessage();
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
        if (this.elements.micBtn) {
            this.elements.micBtn.addEventListener('click', () => {
                this.toggleSpeechRecognition();
            });
        }
        
        // Restart button
        this.elements.restartBtn.addEventListener('click', () => {
            this.restartConversation();
        });
        
        // Config button
        this.elements.configBtn.addEventListener('click', () => {
            this.lastFocusedElement = this.elements.configBtn;
            this.showConfigModal();
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
        
        // Config modal handlers
        this.elements.configModalClose.addEventListener('click', () => {
            this.hideConfigModal();
        });
        
        this.elements.configCancel.addEventListener('click', () => {
            this.hideConfigModal();
        });
        
        this.elements.configSave.addEventListener('click', () => {
            this.saveConfig();
        });
        
        // Close config modal on overlay click
        this.elements.configModal.addEventListener('click', (e) => {
            if (e.target === this.elements.configModal || e.target.classList.contains('modal-overlay')) {
                this.hideConfigModal();
            }
        });
        
        // Close modals on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.elements.configModal.style.display === 'flex') {
                    this.hideConfigModal();
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
        
        if (!userMessage || this.isGenerating) return;
        
        // Store voice input flag before clearing
        const usedVoice = this.usedVoiceInput;
        this.usedVoiceInput = false;
        
        // Clear input and reset height
        this.elements.userInput.value = '';
        this.elements.userInput.style.height = 'auto';
        
        // Add user message to chat
        this.addMessage('user', userMessage);
        
        // Check if configured
        if (!this.isConfigured) {
            this.addMessage('assistant', 'Please configure your Foundry settings to chat.');
            return;
        }
        
        // Disable input while processing
        this.elements.userInput.disabled = true;
        this.elements.sendBtn.disabled = true;
        this.elements.micBtn.disabled = true;
        
        // Check if this is an initial greeting (only if no messages yet)
        const messageCount = this.elements.chatMessages.querySelectorAll('.message').length;
        if (messageCount <= 1) { // Only user's message is in chat
            const greetingPattern = /^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\s!?]*$/i;
            if (greetingPattern.test(userMessage)) {
                // Respond with greeting without searching
                const greetingResponse = "Hello, I'm Andrew. I'm here to help you learn about AI concepts. What would you like to know?";
                this.addMessage('assistant', greetingResponse);
                
                // Synthesize speech if user used voice input
                if (usedVoice && this.config.region) {
                    this.synthesizeSpeech(greetingResponse);
                }
                
                this.elements.userInput.disabled = false;
                this.elements.sendBtn.disabled = false;
                if (this.recognition) this.elements.micBtn.disabled = false;
                return;
            }
        }
        
        // Search for relevant context
        const searchResult = this.searchContext(userMessage);
        
        // Generate response
        await this.generateResponse(userMessage, searchResult, usedVoice);
        
        // Re-enable input
        this.elements.userInput.disabled = false;
        this.elements.sendBtn.disabled = false;
        if (this.recognition) this.elements.micBtn.disabled = false;
        this.elements.userInput.focus();
    }

    
    toggleSpeechRecognition() {
        if (!this.recognition) {
            alert('Microphone access is not supported in your browser.');
            return;
        }
        
        if (this.isListening) {
            this.stopListening();
        } else {
            this.usedVoiceInput = false; // Will be set to true in transcription if successful
            this.startAzureSpeechRecognition();
        }
    }

    addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        messageDiv.setAttribute('role', 'status');
        messageDiv.setAttribute('aria-live', 'polite');
        messageDiv.innerHTML = `<p>${message}</p>`;
        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
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
        
        // Show typing indicator
        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');
        messageTextDiv.innerHTML ='<span class="typing-indicator">●●●</span>';
        
        try {
            // Build prompt with context if available
            let input = userMessage;
            if (context) {
                input = `Context:\n${context}\n\nQuestion: ${userMessage}`;
            }
            
            // Call Foundry Responses API
            const response = await this.callFoundryAPI(input);
            
            // Format and display the response
            let formattedResponse = this.escapeHtml(response).replace(/\n/g, '<br>');
            formattedResponse = `<p>${formattedResponse}</p>`;
            
            // Add learn more links
            if (links && links.length > 0 && categories && categories.length > 0) {
                const linkHtml = links.map((link, index) => {
                    const categoryName = categories[Math.min(index, categories.length - 1)];
                    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
                }).join(' • ');
                formattedResponse += `<hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;"><p><strong>Learn more:</strong> ${linkHtml}</p>`;
            }
            
            // Synthesize speech if user used voice input (only the model response, not the links)
            if (usedVoiceInput && this.config.region) {
                this.synthesizeSpeech(response);
            }
            
            // Animate the typing effect
            await this.animateTyping(messageTextDiv, formattedResponse, 10);
            
            // Add to conversation history
            this.conversationHistory.push({
                role: 'user',
                content: userMessage
            });
            this.conversationHistory.push({
                role: 'assistant',
                content: response
            });
            
        } catch (error) {
            console.error('Error generating response:', error);
            responseMessage.remove();
            this.addMessage('assistant', 'Sorry, I encountered an error: ' + error.message);
        } finally {
            this.isGenerating = false;
            
            // Clear search status after response is complete
            setTimeout(() => {
                this.elements.searchStatus.textContent = '';
            }, 2000);
        }
    }

    async callFoundryAPI(input) {
        const url = `${this.config.endpoint}/openai/v1/responses`;
        
        const requestBody = {
            model: this.config.deployment,
            input: input,
            instructions: this.systemPrompt,
            store: true
        };
        
        // Include previous response ID for conversation continuity
        if (this.previousResponseId) {
            requestBody.previous_response_id = this.previousResponseId;
        }
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': this.config.apiKey
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API request failed: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        
        // Store the response ID for next turn
        if (data.id) {
            this.previousResponseId = data.id;
        }
        
        // Extract response text
        let responseText = '';
        
        if (data.output) {
            if (Array.isArray(data.output)) {
                responseText = data.output
                    .map(item => {
                        if (item.type === 'message' && item.content) {
                            return Array.isArray(item.content) 
                                ? item.content.map(c => c.text || '').join('')
                                : item.content;
                        }
                        if (item.type === 'text') {
                            return item.text || '';
                        }
                        return '';
                    })
                    .join('');
            } else if (typeof data.output === 'string') {
                responseText = data.output;
            }
        } else if (data.choices && data.choices[0]) {
            responseText = data.choices[0].message.content || '';
        } else if (data.message) {
            responseText = data.message.content || data.message;
        }
        
        if (!responseText) {
            throw new Error('No response text found in API response');
        }
        
        return responseText;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async animateTyping(element, htmlContent, speed = 10) {
        // Start with empty content
        element.innerHTML = '';
        
        // Split content into characters for typing effect
        const chars = htmlContent.split('');
        let currentHtml = '';
        
        for (let i = 0; i < chars.length; i++) {
            currentHtml += chars[i];
            
            // Update the element with current content
            // Use a temporary div to parse HTML and get valid structure
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = currentHtml;
            element.innerHTML = tempDiv.innerHTML;
            
            this.scrollToBottom();
            
            // Small delay for typing effect (speed in milliseconds)
            await new Promise(resolve => setTimeout(resolve, speed));
        }
        
        // Ensure final content is complete and properly formatted
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

    async getTtsToken() {
        // Check if we have a valid token
        if (this.ttsToken && this.ttsTokenExpiry && Date.now() < this.ttsTokenExpiry) {
            return this.ttsToken;
        }
        
        // Fetch new token
        const tokenEndpoint = `https://${this.config.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`;
        
        try {
            const response = await fetch(tokenEndpoint, {
                method: 'POST',
                headers: {
                    'Ocp-Apim-Subscription-Key': this.config.apiKey,
                    'Content-Length': '0'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Token fetch failed: ${response.status}`);
            }
            
            this.ttsToken = await response.text();
            // Token valid for 10 minutes, refresh after 9 minutes
            this.ttsTokenExpiry = Date.now() + (9 * 60 * 1000);
            
            return this.ttsToken;
        } catch (error) {
            console.error('Error fetching TTS token:', error);
            throw error;
        }
    }

    async synthesizeSpeech(text) {
        try {
            // Strip HTML tags and convert to plain text using DOMParser (safer than innerHTML)
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, 'text/html');
            const plainText = doc.body.textContent || '';
            
            if (!plainText.trim()) {
                return;
            }
            
            // Use region-based TTS endpoint (same as PowerShell script)
            const speechEndpoint = `https://${this.config.region}.tts.speech.microsoft.com`;
            const url = `${speechEndpoint}/cognitiveservices/v1`;
            
            // Build SSML with Lewis Multilingual voice
            const ssml = `<speak version="1.0" xml:lang="en-US">
  <voice xml:lang="en-US" xml:gender="Male" name="en-US-LewisMultilingualNeural">
    ${this.escapeXml(plainText)}
  </voice>
</speak>`;
            
            // Get TTS token (exchanges API key for OAuth token)
            const token = await this.getTtsToken();
            
            let response;
            try {
                response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/ssml+xml',
                        'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
                    },
                    body: ssml
                });
            } catch (fetchError) {
                throw new Error(`TTS request failed: ${fetchError.message}`);
            }
            
            if (!response.ok) {
                let errorText = '';
                try {
                    errorText = await response.text();
                } catch (e) {
                    errorText = '(could not read error response)';
                }
                
                console.error('TTS request failed');
                console.error('  URL:', url);
                console.error('  Region:', this.config.region);
                console.error('  Status:', response.status);
                console.error('  Status Text:', response.statusText);
                console.error('  Error:', errorText || '(empty error response)');
                console.error('  Response Headers:');
                for (const [key, value] of response.headers.entries()) {
                    console.error(`    ${key}: ${value}`);
                }
                throw new Error(`TTS failed: ${response.status} - ${errorText || 'Unknown error'}`);
            }
            
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            // Play the audio
            audio.play().catch(error => {
                console.error('Error playing synthesized speech:', error);
            });
            
            // Clean up the URL after audio finishes
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
            
        } catch (error) {
            console.error('Error synthesizing speech:', error);
        }
    }

    escapeXml(text) {
        return text.replace(/[<>&'"]/g, (char) => {
            switch (char) {
                case '<': return '&lt;';
                case '>': return '&gt;';
                case '&': return '&amp;';
                case "'": return '&apos;';
                case '"': return '&quot;';
                default: return char;
            }
        });
    }

    restartConversation() {
        if (confirm('Are you sure you want to start a new conversation? This will clear the chat history.')) {
            // Clear conversation history
            this.conversationHistory = [];
            this.previousResponseId = null;
            
            // Clear chat messages (keep welcome message)
            const messages = this.elements.chatMessages.querySelectorAll('.message:not(.welcome-message)');
            messages.forEach(msg => msg.remove());
            
            // Clear search status
            this.elements.searchStatus.textContent = '';
            
            console.log('Conversation restarted');
        }
    }

    showConfigModal() {
        this.elements.configModal.style.display = 'flex';
        this.currentModal = this.elements.configModal;
        this.elements.configModal.setAttribute('aria-hidden', 'false');
        // Clear any previous status
        this.elements.configStatus.textContent = '';
        this.elements.configStatus.className = 'config-status';
        // Set focus to first input
        setTimeout(() => {
            this.elements.foundryEndpoint.focus();
            this.setupModalFocusTrap(this.elements.configModal);
        }, 100);
    }
    
    hideConfigModal() {
        this.elements.configModal.style.display = 'none';
        this.elements.configModal.setAttribute('aria-hidden', 'true');
        this.removeModalFocusTrap();
        this.currentModal = null;
        // Restore focus
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
