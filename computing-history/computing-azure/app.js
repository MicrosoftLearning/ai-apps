// Constants
const DEFAULT_MODEL_INSTRUCTIONS = "You are a helpful AI agent that provides information and advice about the history of computing and historic computer restoration. Don't engage in discussions outside of this topic area. Always try to provide as much helpful information as you can based on your training data, and use web search as a tool to supplement your knowledge when needed, rather than relying on it for every question. Use web search if the user explicitly asks you to search for information or items for sale (for example, on eBay). If you do use web search, make sure to include the most relevant information from the search results in your response, not just the links. ";

// Global state
let config = {
    endpoint: '',
    apiKey: '',
    deployment: '',
    region: ''
};

let conversationHistory = [];
let selectedImage = null;
let isConfigured = false;
let recognition = null;
let isListening = false;
let usedSpeechInput = false;
let isSpeaking = false;
let previousResponseId = null;
let recordingTimeout = null;
let mediaRecorder = null;
let audioChunks = [];
let currentAudio = null;
let speechToken = null;
let speechTokenExpiry = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    loadConfig();
    initializeSpeechRecognition();
    updateUIState();
    setupEventListeners();
});

/**
 * Toggles the configuration panel visibility
 */
function toggleConfig() {
    const content = document.getElementById('configContent');
    const icon = document.getElementById('toggleIcon');
    const header = document.querySelector('.config-header');
    
    content.classList.toggle('hidden');
    icon.classList.toggle('collapsed');
    
    // Update ARIA state
    const isExpanded = !content.classList.contains('hidden');
    header.setAttribute('aria-expanded', isExpanded);
}

/**
 * Saves the Azure OpenAI configuration to memory and localStorage
 */
function saveConfig() {
    const endpoint = document.getElementById('endpoint').value.trim();
    const apiKey = document.getElementById('apiKey').value.trim();
    const deployment = document.getElementById('deployment').value.trim();
    const region = document.getElementById('region').value.trim();
    const statusDiv = document.getElementById('configStatus');
    
    if (!endpoint || !apiKey || !deployment) {
        statusDiv.textContent = 'Please fill in all fields';
        statusDiv.className = 'config-status error';
        return;
    }
    
    // Validate endpoint format and sanitize
    try {
        const url = new URL(endpoint);
        // Only allow https URLs for security
        if (url.protocol !== 'https:') {
            statusDiv.textContent = 'Endpoint must use HTTPS';
            statusDiv.className = 'config-status error';
            return;
        }
    } catch (e) {
        statusDiv.textContent = 'Invalid endpoint URL';
        statusDiv.className = 'config-status error';
        return;
    }
    
    // Truncate endpoint to only include up to and including .com
    let truncatedEndpoint = endpoint;
    const comIndex = endpoint.indexOf('.com');
    if (comIndex !== -1) {
        truncatedEndpoint = endpoint.substring(0, comIndex + 4); // +4 to include ".com"
    }
    
    config.endpoint = truncatedEndpoint;
    config.apiKey = apiKey;
    config.deployment = deployment;
    config.region = region;
    
    // Save to localStorage (exclude apiKey for security)
    const configToPersist = {
        endpoint: config.endpoint,
        deployment: config.deployment,
        region: config.region
    };
    localStorage.setItem('azureOpenAIConfig', JSON.stringify(configToPersist));
    
    isConfigured = true;
    statusDiv.textContent = 'Configuration saved successfully!';
    statusDiv.className = 'config-status success';
    
    updateUIState();
    
    // Auto-collapse config section
    setTimeout(() => {
        const content = document.getElementById('configContent');
        const icon = document.getElementById('toggleIcon');
        content.classList.add('hidden');
        icon.classList.add('collapsed');
    }, 1500);
}

/**
 * Loads saved configuration from localStorage
 */
function loadConfig() {
    const saved = localStorage.getItem('azureOpenAIConfig');
    if (saved) {
        try {
            const savedConfig = JSON.parse(saved);
            
            // Load non-sensitive config from storage (apiKey is NOT loaded for security)
            config.endpoint = savedConfig.endpoint || '';
            config.deployment = savedConfig.deployment || '';
            config.region = savedConfig.region || '';
            config.apiKey = ''; // Always start with empty API key
            
            // Truncate endpoint to only include up to and including .com (in case old value was saved)
            if (config.endpoint) {
                const comIndex = config.endpoint.indexOf('.com');
                if (comIndex !== -1) {
                    config.endpoint = config.endpoint.substring(0, comIndex + 4);
                }
            }
            
            document.getElementById('endpoint').value = config.endpoint || '';
            document.getElementById('apiKey').value = ''; // Never populate API key from storage
            document.getElementById('deployment').value = config.deployment || '';
            document.getElementById('region').value = config.region || '';
            
            // Note: isConfigured will be false since apiKey is not persisted
            // User must re-enter API key each session for security
            if (config.endpoint && config.apiKey && config.deployment) {
                isConfigured = true;
                // Collapse config section if already configured
                const content = document.getElementById('configContent');
                const icon = document.getElementById('toggleIcon');
                content.classList.add('hidden');
                icon.classList.add('collapsed');
            }
        } catch (e) {
            console.error('Error loading config:', e);
        }
    }
}

/**
 * Updates the UI state based on whether the app is configured
 */
function updateUIState() {
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const attachBtn = document.getElementById('attachBtn');
    const micBtn = document.getElementById('micBtn');
    const restartBtn = document.getElementById('restartBtn');
    const viewInstructionsBtn = document.getElementById('viewInstructionsBtn');
    const chatMessages = document.getElementById('chatMessages');
    
    if (isConfigured) {
        userInput.disabled = false;
        sendBtn.disabled = false;
        attachBtn.disabled = false;
        micBtn.disabled = !recognition; // Only enable if speech recognition is available
        restartBtn.disabled = false;
        viewInstructionsBtn.disabled = false;
        
        // Remove welcome message if it exists
        const welcomeMsg = chatMessages.querySelector('.welcome-message');
        if (welcomeMsg && conversationHistory.length === 0) {
            welcomeMsg.textContent = "Let's chat about computing history...";
        }
    } else {
        userInput.disabled = true;
        sendBtn.disabled = true;
        attachBtn.disabled = true;
        micBtn.disabled = true;
        restartBtn.disabled = true;
        viewInstructionsBtn.disabled = true;
    }
}

/**
 * Initializes speech recognition capability check
 */
function initializeSpeechRecognition() {
    // Azure Speech API will be used - check for microphone support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Microphone access not supported in this browser');
        recognition = null;
        return;
    }
    
    recognition = true; // Flag to indicate speech is available
    mediaRecorder = null;
    audioChunks = [];
}

/**
 * Toggles speech recognition on/off
 */
function toggleSpeechRecognition() {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser. Please use a modern browser like Chrome or Edge.');
        return;
    }
    
    if (isListening) {
        stopListening();
    } else {
        startAzureSpeechRecognition();
    }
}

/**
 * Starts Azure Speech recognition
 */
async function startAzureSpeechRecognition() {
    if (!isConfigured) {
        alert('Please configure your Azure OpenAI settings to use voice input.');
        return;
    }
    
    if (!config.region) {
        alert('Please add the Foundry Project Region to your configuration for speech-to-text to work.');
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
        
        audioChunks = [];
        
        // Use audio/webm;codecs=opus or check available types
        let mimeType = 'audio/webm;codecs=opus';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'audio/webm';
        }
        
        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Stop all tracks to release the microphone
            stream.getTracks().forEach(track => track.stop());
            
            stopListening();
            
            // Clear the auto-stop timeout
            if (recordingTimeout) {
                clearTimeout(recordingTimeout);
                recordingTimeout = null;
            }
            
            if (audioChunks.length === 0) {
                return;
            }
            
            try {
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                
                // Convert to WAV format for Azure Speech API
                const wavBlob = await convertToWav(audioBlob);
                
                // Call Azure Speech API
                await transcribeAudio(wavBlob);
            } catch (error) {
                console.error('Error processing audio:', error);
                alert('Failed to process audio. Please try again.');
            }
        };
        
        // Request data in chunks for better compatibility
        mediaRecorder.start(100);
        isListening = true;
        const micBtn = document.getElementById('micBtn');
        micBtn.classList.add('listening');
        micBtn.title = 'Listening... (auto-stops in 5s or click to stop)';
        micBtn.setAttribute('aria-pressed', 'true');
        
        // Auto-stop after 5 seconds
        recordingTimeout = setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 5000);
        
    } catch (error) {
        console.error('Error starting speech recognition:', error);
        let errorMsg = 'Could not start speech recognition: ';
        if (error.name === 'NotAllowedError') {
            errorMsg += 'Microphone access denied.';
        } else if (error.name === 'NotFoundError') {
            errorMsg += 'No microphone detected.';
        } else {
            errorMsg += error.message;
        }
        alert(errorMsg);
        stopListening();
    }
}

/**
 * Stops listening for voice input
 */
function stopListening() {
    isListening = false;
    const micBtn = document.getElementById('micBtn');
    micBtn.classList.remove('listening');
    micBtn.title = 'Voice input';
    micBtn.setAttribute('aria-pressed', 'false');
    
    // Clear the auto-stop timeout
    if (recordingTimeout) {
        clearTimeout(recordingTimeout);
        recordingTimeout = null;
    }
    
    // Stop recording if active
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}

async function convertToWav(audioBlob) {
    // Create an audio context
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
    });
    
    // Read the audio blob as array buffer
    const arrayBuffer = await audioBlob.arrayBuffer();
    
    // Decode the audio data
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Convert to WAV format
    const wavBuffer = audioBufferToWav(audioBuffer);
    
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function audioBufferToWav(audioBuffer) {
    const numOfChan = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numOfChan * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    setUint32(0x46464952); // "RIFF"
    setUint32(36 + length); // file length - 8
    setUint32(0x45564157); // "WAVE"
    
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(numOfChan);
    setUint32(audioBuffer.sampleRate);
    setUint32(audioBuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(numOfChan * 2); // block-align
    setUint16(16); // 16-bit
    
    setUint32(0x61746164); // "data" - chunk
    setUint32(length); // chunk length
    
    // Write interleaved data
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
        channels.push(audioBuffer.getChannelData(i));
    }
    
    while (pos < audioBuffer.length) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][pos]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(44 + offset, sample, true);
            offset += 2;
        }
        pos++;
    }
    
    return buffer;
    
    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }
    
    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }
}

async function transcribeAudio(audioBlob) {
    try {
        // Get Speech token (exchanges API key for OAuth token)
        const token = await getSpeechToken();
        
        // Construct Azure Speech endpoint using the region
        const speechEndpoint = `https://${config.region}.stt.speech.microsoft.com`;
        const url = `${speechEndpoint}/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed`;
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
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
            const userInput = document.getElementById('userInput');
            userInput.value = transcript;
            usedSpeechInput = true;
            userInput.focus();
            
            // Automatically submit the message
            setTimeout(() => {
                sendMessage();
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

function stripMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '$1')  // Bold
        .replace(/\*(.*?)\*/g, '$1')      // Italic
        .replace(/`(.*?)`/g, '$1')        // Code
        .replace(/#{1,6}\s/g, '')         // Headers
        .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')  // Links
        .replace(/!\[([^\]]*)\]\([^\)]+\)/g, '$1') // Images
        .replace(/^\s*[-*+]\s/gm, '')     // Lists
        .replace(/^\s*\d+\.\s/gm, '')     // Numbered lists
        .replace(/^\s*>\s/gm, '')         // Blockquotes
        .replace(/\n{2,}/g, ' ')          // Multiple newlines
        .trim();
}

async function getSpeechToken() {
    // Check if we have a valid token
    if (speechToken && speechTokenExpiry && Date.now() < speechTokenExpiry) {
        return speechToken;
    }
    
    // Fetch new token for Azure Speech services (used by both STT and TTS)
    const tokenEndpoint = `https://${config.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`;
    
    try {
        const response = await fetch(tokenEndpoint, {
            method: 'POST',
            headers: {
                'Ocp-Apim-Subscription-Key': config.apiKey,
                'Content-Length': '0'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Token fetch failed: ${response.status}`);
        }
        
        speechToken = await response.text();
        // Token valid for 10 minutes, refresh after 9 minutes
        speechTokenExpiry = Date.now() + (9 * 60 * 1000);
        
        return speechToken;
    } catch (error) {
        console.error('Error fetching Speech token:', error);
        throw error;
    }
}

function escapeXml(unsafe) {
    if (!unsafe) return '';
    return unsafe.replace(/[<>&'"]/g, function (c) {
        switch (c) {
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '&': return '&amp;';
            case '\'': return '&apos;';
            case '"': return '&quot;';
        }
    });
}

async function synthesizeSpeech(text) {
    if (!config.region) {
        console.warn('Region not configured for TTS');
        return;
    }
    
    isSpeaking = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.textContent = '⬜';
    sendBtn.classList.add('speaking');
    sendBtn.title = 'Stop speech';
    
    try {
        // Strip HTML tags and convert to plain text using DOMParser (safer than innerHTML)
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        const plainText = doc.body.textContent || '';
        
        if (!plainText.trim()) {
            resetSendButton();
            return;
        }
        
        // Use region-based TTS endpoint
        const speechEndpoint = `https://${config.region}.tts.speech.microsoft.com`;
        const url = `${speechEndpoint}/cognitiveservices/v1`;
        
        // Build SSML with Ava DragonHD voice
        const ssml = `<speak version="1.0" xml:lang="en-US">
  <voice name="en-US-Ava:DragonHDLatestNeural">
    ${escapeXml(plainText)}
  </voice>
</speak>`;
        
        // Get Speech token (exchanges API key for OAuth token)
        const token = await getSpeechToken();
        
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
            console.error('  Region:', config.region);
            console.error('  Status:', response.status);
            console.error('  Status Text:', response.statusText);
            console.error('  Error:', errorText || '(empty error response)');
            throw new Error(`TTS failed: ${response.status} - ${errorText || 'Unknown error'}`);
        }
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        // Store reference to current audio
        currentAudio = audio;
        
        // Play the audio and wait for it to finish
        await new Promise((resolve, reject) => {
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                currentAudio = null;
                resetSendButton();
                resolve();
            };
            
            audio.onerror = (error) => {
                URL.revokeObjectURL(audioUrl);
                currentAudio = null;
                resetSendButton();
                reject(error);
            };
            
            audio.play().catch(reject);
        });
        
    } catch (error) {
        console.error('Error synthesizing speech:', error);
        resetSendButton();
    }
}

function stopSpeech() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    resetSendButton();
}

function resetSendButton() {
    isSpeaking = false;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.textContent = '▶';
    sendBtn.classList.remove('speaking');
    sendBtn.title = 'Send message';
}

// Chat functions
/**
 * Handles the send button click
 */
function handleSendButtonClick() {
    if (isSpeaking) {
        stopSpeech();
    } else {
        sendMessage();
    }
}

// Auto-resize textarea as user types
function autoResizeTextarea() {
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    }
}

// Setup all event listeners
function setupEventListeners() {
    // Configuration header - toggle on click and keyboard
    const configHeader = document.getElementById('configHeader');
    if (configHeader) {
        configHeader.addEventListener('click', toggleConfig);
        configHeader.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleConfig();
            }
        });
    }
    
    // Save config button
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', saveConfig);
    }
    
    // View instructions button
    const viewInstructionsBtn = document.getElementById('viewInstructionsBtn');
    if (viewInstructionsBtn) {
        viewInstructionsBtn.addEventListener('click', showInstructions);
    }
    
    // Restart button
    const restartBtn = document.getElementById('restartBtn');
    if (restartBtn) {
        restartBtn.addEventListener('click', restartConversation);
    }
    
    // Close instructions button
    const closeInstructionsBtn = document.getElementById('closeInstructionsBtn');
    if (closeInstructionsBtn) {
        closeInstructionsBtn.addEventListener('click', closeInstructions);
    }
    
    // Attach button
    const attachBtn = document.getElementById('attachBtn');
    if (attachBtn) {
        attachBtn.addEventListener('click', function() {
            document.getElementById('imageInput').click();
        });
    }
    
    // Mic button
    const micBtn = document.getElementById('micBtn');
    if (micBtn) {
        micBtn.addEventListener('click', toggleSpeechRecognition);
    }
    
    // Send button
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', handleSendButtonClick);
    }
    
    // Remove image button
    const removeImageBtn = document.getElementById('removeImageBtn');
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', removeImage);
    }
    
    // Image input
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', handleImageSelect);
    }
    
    // User input textarea - auto-resize and keyboard handling
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.addEventListener('input', autoResizeTextarea);
        userInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Close modal on Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const modal = document.getElementById('instructionsModal');
            if (modal && modal.style.display === 'flex') {
                closeInstructions();
            }
        }
    });
}

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            selectedImage = {
                data: e.target.result,
                type: file.type
            };
            displayImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Displays an image preview
 * @param {string} dataUrl - The image data URL
 */
function displayImagePreview(dataUrl) {
    const preview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImage');
    previewImg.src = dataUrl;
    preview.style.display = 'block';
}

/**
 * Removes the selected image
 */
function removeImage() {
    selectedImage = null;
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('imageInput').value = '';
}

/**
 * Sends a message to Azure OpenAI
 */
async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const messageText = userInput.value.trim();
    
    if (!messageText && !selectedImage) {
        return;
    }
    
    // Track if speech was used for this message
    const shouldUseSpeechOutput = usedSpeechInput;
    const addConciseInstruction = usedSpeechInput;
    usedSpeechInput = false; // Reset for next message
    
    // Disable input while processing
    userInput.disabled = true;
    document.getElementById('sendBtn').disabled = true;
    document.getElementById('attachBtn').disabled = true;
    document.getElementById('micBtn').disabled = true;
    
    // Create user message
    const userMessage = {
        role: 'user',
        content: []
    };
    
    if (messageText) {
        userMessage.content.push({
            type: 'input_text',
            text: messageText
        });
    }
    
    if (selectedImage) {
        userMessage.content.push({
            type: 'input_image',
            image_url: selectedImage.data
        });
    }
    
    // Add to conversation history
    conversationHistory.push(userMessage);
    
    // Display user message
    displayMessage('user', messageText, selectedImage ? selectedImage.data : null);
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto'; // Reset textarea height
    removeImage();
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Call Azure OpenAI API
        const response = await callAzureOpenAI(addConciseInstruction);
        
        // Start speech synthesis immediately if user used speech input (before displaying text)
        if (shouldUseSpeechOutput) {
            synthesizeSpeech(response);
        }
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add assistant response to history
        conversationHistory.push({
            role: 'assistant',
            content: response
        });
        
        // Display assistant message
        displayMessage('assistant', response);
        
    } catch (error) {
        const errorMsg = 'Sorry, there was an error processing your request: ' + error.message;
        
        // Start speech synthesis immediately if needed (before displaying text)
        if (shouldUseSpeechOutput) {
            synthesizeSpeech(errorMsg);
        }
        
        removeTypingIndicator(typingId);
        displayMessage('assistant', errorMsg);
        console.error('Error:', error);
    }
    
    // Re-enable input
    userInput.disabled = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('attachBtn').disabled = false;
    if (recognition) {
        document.getElementById('micBtn').disabled = false;
    }
    userInput.focus();
}

async function callAzureOpenAI(useConciseInstruction = false) {
    const url = `${config.endpoint}/openai/v1/responses`;
    
    // Get the latest user message
    const latestMessage = conversationHistory[conversationHistory.length - 1];
    
    // Format input for Responses API
    let input;
    if (Array.isArray(latestMessage.content)) {
        // For multimodal content, wrap in a message item
        input = [{
            type: "message",
            role: "user",
            content: latestMessage.content
        }];
    } else {
        // For text-only, send as string
        input = latestMessage.content;
    }
    
    // Build instructions
    let instructions = DEFAULT_MODEL_INSTRUCTIONS;
    
    // Add concise instruction if speech input was used
    if (useConciseInstruction) {
        instructions = DEFAULT_MODEL_INSTRUCTIONS + " Respond with a single sentence.";
    }
    
    const requestBody = {
        model: config.deployment,
        input: input,
        instructions: instructions,
        store: true
    };
    
    // Add tools only if not using concise instructions (single sentence responses don't need web search)
    if (!useConciseInstruction) {
        requestBody.tools = [
            { type: "web_search" }
        ];
        requestBody.tool_choice = "auto";
    }
    
    // Include previous response ID for conversation continuity
    if (previousResponseId) {
        requestBody.previous_response_id = previousResponseId;
    }
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'api-key': config.apiKey
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
        previousResponseId = data.id;
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

function displayMessage(role, text, imageUrl = null) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove welcome message if it exists
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.setAttribute('role', 'article');
    messageDiv.setAttribute('aria-label', `${role === 'user' ? 'User' : 'Assistant'} message`);
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (imageUrl) {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'User uploaded image';
        contentDiv.appendChild(img);
    }
    
    if (text) {
        if (role === 'assistant' && typeof marked !== 'undefined') {
            // Render markdown for assistant messages and sanitize to prevent XSS
            const rawHtml = marked.parse(text);
            const cleanHtml = DOMPurify.sanitize(rawHtml, {
                ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                               'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'table', 'thead', 
                               'tbody', 'tr', 'th', 'td', 'hr', 'del', 'ins'],
                ALLOWED_ATTR: ['href', 'target', 'rel'],
                ALLOW_DATA_ATTR: false
            });
            
            // Add target="_blank" to all links
            const finalHtml = addTargetBlankToLinks(cleanHtml);
            
            // Add the content div to the bubble and message first
            bubbleDiv.appendChild(contentDiv);
            messageDiv.appendChild(bubbleDiv);
            chatMessages.appendChild(messageDiv);
            
            // Typewriter effect for assistant messages
            typewriterEffect(contentDiv, finalHtml);
        } else {
            // Plain text for user messages - use textContent to prevent any HTML injection
            const textNode = document.createTextNode(text);
            contentDiv.appendChild(textNode);
            bubbleDiv.appendChild(contentDiv);
            messageDiv.appendChild(bubbleDiv);
            chatMessages.appendChild(messageDiv);
        }
    } else {
        bubbleDiv.appendChild(contentDiv);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
    }
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTargetBlankToLinks(htmlContent) {
    // Parse HTML and add target="_blank" and rel="noopener noreferrer" to all links
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    const links = tempDiv.querySelectorAll('a');
    links.forEach(link => {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    });
    return tempDiv.innerHTML;
}

function typewriterEffect(element, htmlContent) {
    // Create a temporary element to parse the HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    
    const charsPerFrame = 3; // Characters to add per frame for speed
    let currentIndex = 0;
    const plainText = tempDiv.textContent || '';
    
    // Start with empty content
    element.innerHTML = '';
    
    function addNextChars() {
        if (currentIndex < plainText.length) {
            currentIndex += charsPerFrame;
            const currentText = plainText.substring(0, Math.min(currentIndex, plainText.length));
            
            // Re-parse through markdown to maintain formatting
            const partialHtml = marked.parse(currentText);
            const cleanPartialHtml = DOMPurify.sanitize(partialHtml, {
                ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                               'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'table', 'thead', 
                               'tbody', 'tr', 'th', 'td', 'hr', 'del', 'ins'],
                ALLOWED_ATTR: ['href', 'target', 'rel'],
                ALLOW_DATA_ATTR: false
            });
            
            // Add target="_blank" to all links
            const finalPartialHtml = addTargetBlankToLinks(cleanPartialHtml);
            
            element.innerHTML = finalPartialHtml;
            
            // Scroll to bottom during animation
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            requestAnimationFrame(addNextChars);
        } else {
            // Ensure final content is complete with target="_blank" for links
            element.innerHTML = addTargetBlankToLinks(htmlContent);
        }
    }
    
    addNextChars();
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-indicator-msg';
    messageDiv.setAttribute('role', 'status');
    messageDiv.setAttribute('aria-label', 'Assistant is typing');
    messageDiv.setAttribute('aria-live', 'polite');
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    typingDiv.setAttribute('aria-hidden', 'true');
    
    bubbleDiv.appendChild(typingDiv);
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'typing-indicator-msg';
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Restarts the conversation by clearing history
 */
function restartConversation() {
    if (confirm('Are you sure you want to clear the conversation history?')) {
        // Stop any ongoing speech
        stopSpeech();
        if (isListening && recognition) {
            recognition.stop();
        }
        
        conversationHistory = [];
        previousResponseId = null;
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '<div class="welcome-message">Let\'s chat about computing history...</div>';
        removeImage();
        usedSpeechInput = false;
    }
}

/**
 * Shows the instructions modal with focus management
 */
function showInstructions() {
    const modal = document.getElementById('instructionsModal');
    const modelName = document.getElementById('modelName');
    const instructionsText = document.getElementById('instructionsText');
    const toolsList = document.getElementById('toolsList');
    const closeBtn = document.getElementById('closeInstructionsBtn');
    
    // Populate model name
    modelName.textContent = config.deployment || 'Not configured';
    
    // Populate instructions
    instructionsText.textContent = DEFAULT_MODEL_INSTRUCTIONS;
    
    // Populate tools list
    const tools = ['web_search'];
    toolsList.textContent = tools.join(', ');
    
    modal.style.display = 'flex';
    
    // Focus the close button for keyboard accessibility
    if (closeBtn) {
        closeBtn.focus();
    }
}

/**
 * Closes the instructions modal
 */
function closeInstructions() {
    const modal = document.getElementById('instructionsModal');
    const viewInstructionsBtn = document.getElementById('viewInstructionsBtn');
    
    modal.style.display = 'none';
    
    // Return focus to the button that opened the modal
    if (viewInstructionsBtn) {
        viewInstructionsBtn.focus();
    }
}
