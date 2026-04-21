// Constants
const DEFAULT_MODEL_INSTRUCTIONS = "You are a helpful AI agent that provides information and advice about the history of computing and historic computer restoration. Don't engage in discussions outside of this topic area. Use the web search tool if the user explicitly asks you to search for information or items for sale (for example, on eBay). Be sure to include the most relevant information from search results in your response, not just the links. ";
const CONTENT_FILTER_MESSAGE = "I'm sorry, I can't help with that because it triggered a content-safety filtering policy.\nI can only help with information about the history of computing.";

// Global state
let config = {
    endpoint: '',
    apiKey: '',
    deployment: '',
    authMode: 'key', // 'key' or 'entra'
    clientId: '',
    tenantId: ''
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
let msalInstance = null;
let msalAccount = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function () {
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
    const deployment = document.getElementById('deployment').value.trim();
    const statusDiv = document.getElementById('configStatus');
    const authModeToggle = document.getElementById('auth-mode-toggle');
    const authMode = authModeToggle.checked ? 'entra' : 'key';

    if (!endpoint || !deployment) {
        statusDiv.textContent = 'Please fill in endpoint and deployment';
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

        const comIndex = endpoint.indexOf('.com');
        if (comIndex !== -1) {
            baseEndpoint = endpoint.substring(0, comIndex + 4);
        }
    } catch (e) {
        statusDiv.textContent = 'Invalid endpoint URL';
        statusDiv.className = 'config-status error';
        return;
    }

    config.endpoint = baseEndpoint;
    config.deployment = deployment;
    config.authMode = authMode;

    // Validate based on authentication mode
    if (authMode === 'key') {
        const apiKey = document.getElementById('apiKey').value.trim();
        if (!apiKey) {
            statusDiv.textContent = 'Please enter an API key';
            statusDiv.className = 'config-status error';
            return;
        }
        config.apiKey = apiKey;
        config.clientId = '';
        config.tenantId = '';
        msalInstance = null;
        msalAccount = null;
    } else {
        const clientId = document.getElementById('entra-client-id').value.trim();
        const tenantId = document.getElementById('entra-tenant-id').value.trim();

        if (!clientId || !tenantId) {
            statusDiv.textContent = 'Please enter both Client ID and Tenant ID';
            statusDiv.className = 'config-status error';
            return;
        }

        if (!msalAccount) {
            statusDiv.textContent = 'Please sign in with Entra ID before saving';
            statusDiv.className = 'config-status error';
            return;
        }

        config.clientId = clientId;
        config.tenantId = tenantId;
        config.apiKey = '';
    }

    // Save to localStorage (do not persist apiKey)
    const configToPersist = {
        endpoint: config.endpoint,
        deployment: config.deployment,
        authMode: config.authMode,
        clientId: config.clientId,
        tenantId: config.tenantId
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

            // Load non-sensitive config from storage
            config.endpoint = savedConfig.endpoint || '';
            config.deployment = savedConfig.deployment || '';
            config.apiKey = ''; // Always start with empty API key for security
            config.authMode = savedConfig.authMode || 'key';
            config.clientId = savedConfig.clientId || '';
            config.tenantId = savedConfig.tenantId || '';

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
            document.getElementById('entra-client-id').value = config.clientId || '';
            document.getElementById('entra-tenant-id').value = config.tenantId || '';

            // Set toggle state
            const authModeToggle = document.getElementById('auth-mode-toggle');
            if (config.authMode === 'entra') {
                authModeToggle.checked = true;
                document.getElementById('key-auth-fields').style.display = 'none';
                document.getElementById('entra-auth-fields').style.display = 'block';
                document.getElementById('entra-help-btn').style.display = 'inline-flex';
            } else {
                authModeToggle.checked = false;
                document.getElementById('key-auth-fields').style.display = 'block';
                document.getElementById('entra-auth-fields').style.display = 'none';
                document.getElementById('entra-help-btn').style.display = 'none';
            }

            // Initialize MSAL if in Entra ID mode and we have clientId and tenantId
            if (config.authMode === 'entra' && config.clientId && config.tenantId) {
                initializeMSAL();
            }

            // Note: isConfigured will be false since apiKey is not persisted
            // User must re-enter API key each session for security
            if (config.endpoint && config.deployment) {
                if (config.authMode === 'key' && config.apiKey) {
                    isConfigured = true;
                    // Collapse config section if already configured
                    const content = document.getElementById('configContent');
                    const icon = document.getElementById('toggleIcon');
                    content.classList.add('hidden');
                    icon.classList.add('collapsed');
                } else if (config.authMode === 'entra' && msalAccount) {
                    isConfigured = true;
                    // Collapse config section if already configured
                    const content = document.getElementById('configContent');
                    const icon = document.getElementById('toggleIcon');
                    content.classList.add('hidden');
                    icon.classList.add('collapsed');
                }
            }
        } catch (e) {
            console.error('Error loading config:', e);
        }
    }
}

/**
 * Initializes MSAL for Entra ID authentication
 */
function initializeMSAL() {
    if (!config.clientId || !config.tenantId) {
        console.error('Cannot initialize MSAL: missing client ID or tenant ID');
        return;
    }

    try {
        const msalConfig = {
            auth: {
                clientId: config.clientId,
                authority: `https://login.microsoftonline.com/${config.tenantId}`,
                redirectUri: window.location.href.split('?')[0].split('#')[0]
            },
            cache: {
                cacheLocation: 'localStorage',
                storeAuthStateInCookie: false
            },
            system: {
                allowRedirectInIframe: false,
                windowHashTimeout: 60000,
                iframeHashTimeout: 6000,
                loadFrameTimeout: 0
            }
        };

        msalInstance = new msal.PublicClientApplication(msalConfig);

        // Initialize MSAL and handle any redirect responses
        msalInstance.initialize().then(() => {
            return msalInstance.handleRedirectPromise();
        }).then((response) => {
            if (response && response.account) {
                msalAccount = response.account;
                msalInstance.setActiveAccount(msalAccount);
            } else {
                const accounts = msalInstance.getAllAccounts();
                if (accounts.length > 0) {
                    msalAccount = accounts[0];
                    msalInstance.setActiveAccount(msalAccount);
                }
            }

            if (msalAccount) {
                isConfigured = true;
                updateUIState();
                updateSignInButtonState();
            }
        }).catch((error) => {
            console.error('Error handling MSAL redirect:', error);
        });
    } catch (error) {
        console.error('Error initializing MSAL:', error);
    }
}

/**
 * Updates the sign-in status message
 */
function updateSigninStatus(message, isError = false) {
    const signinStatus = document.getElementById('signin-status');
    if (signinStatus) {
        signinStatus.textContent = message;
        signinStatus.style.color = isError ? '#721c24' : '#155724';
    }
}

/**
 * Updates the sign-in button state based on authentication status
 */
function updateSignInButtonState() {
    const entraSigninBtn = document.getElementById('entra-signin-btn');
    if (!entraSigninBtn) return;

    if (msalAccount) {
        entraSigninBtn.textContent = 'Sign Out';
        entraSigninBtn.disabled = false;
        entraSigninBtn.setAttribute('aria-label', 'Sign out of Entra ID');
        updateSigninStatus(`Signed in as ${msalAccount.username}`);
    } else {
        entraSigninBtn.textContent = 'Sign In';
        entraSigninBtn.setAttribute('aria-label', 'Sign in with Entra ID');

        const clientId = document.getElementById('entra-client-id').value.trim();
        const tenantId = document.getElementById('entra-tenant-id').value.trim();
        entraSigninBtn.disabled = !clientId || !tenantId;

        updateSigninStatus('');
    }
}

/**
 * Handles authentication mode toggle between Key and Entra ID
 */
function handleAuthModeToggle() {
    const authModeToggle = document.getElementById('auth-mode-toggle');
    const isEntraMode = authModeToggle.checked;

    const keyAuthFields = document.getElementById('key-auth-fields');
    const entraAuthFields = document.getElementById('entra-auth-fields');
    const entraHelpBtn = document.getElementById('entra-help-btn');

    if (isEntraMode) {
        keyAuthFields.style.display = 'none';
        entraAuthFields.style.display = 'block';
        authModeToggle.setAttribute('aria-checked', 'true');
        entraHelpBtn.style.display = 'inline-flex';
        updateSignInButtonState();
    } else {
        keyAuthFields.style.display = 'block';
        entraAuthFields.style.display = 'none';
        authModeToggle.setAttribute('aria-checked', 'false');
        entraHelpBtn.style.display = 'none';
        updateSigninStatus('');
    }

    const configStatus = document.getElementById('configStatus');
    configStatus.textContent = '';
    configStatus.className = 'config-status';
}

/**
 * Signs in with Entra ID
 */
async function signInWithEntraID() {
    const clientId = document.getElementById('entra-client-id').value.trim();
    const tenantId = document.getElementById('entra-tenant-id').value.trim();
    const endpoint = document.getElementById('endpoint').value.trim();

    if (!clientId || !tenantId) {
        updateSigninStatus('Please enter both Client ID and Tenant ID', true);
        return;
    }

    if (!endpoint) {
        updateSigninStatus('Please enter an endpoint URL first', true);
        return;
    }

    // Validate endpoint
    let baseEndpoint = endpoint;
    try {
        const url = new URL(endpoint);
        if (url.protocol !== 'https:') {
            updateSigninStatus('Endpoint must use HTTPS', true);
            return;
        }

        const comIndex = endpoint.indexOf('.com');
        if (comIndex !== -1) {
            baseEndpoint = endpoint.substring(0, comIndex + 4);
        }
    } catch (e) {
        updateSigninStatus('Invalid endpoint URL', true);
        return;
    }

    config.clientId = clientId;
    config.tenantId = tenantId;

    initializeMSAL();

    if (!msalInstance) {
        updateSigninStatus('Failed to initialize authentication', true);
        return;
    }

    try {
        updateSigninStatus('Opening sign-in window...');

        const loginRequest = {
            scopes: ['https://cognitiveservices.azure.com/.default'],
            prompt: 'select_account',
            redirectUri: window.location.href.split('?')[0].split('#')[0]
        };

        const response = await msalInstance.loginPopup(loginRequest);

        if (response && response.account) {
            msalAccount = response.account;
            msalInstance.setActiveAccount(msalAccount);
            updateSignInButtonState();
            console.log('Successfully signed in with Entra ID');
        } else {
            updateSigninStatus('Sign-in failed', true);
        }
    } catch (error) {
        console.error('Error during sign-in:', error);
        updateSigninStatus(`Sign-in error: ${error.message}`, true);
    }
}

/**
 * Signs out from Entra ID
 */
async function signOutEntraID() {
    if (!msalInstance || !msalAccount) {
        return;
    }

    try {
        updateSigninStatus('Signing out...');

        const logoutRequest = {
            account: msalAccount,
            postLogoutRedirectUri: window.location.href.split('?')[0].split('#')[0]
        };

        await msalInstance.logoutPopup(logoutRequest);

        msalAccount = null;
        isConfigured = false;
        updateSignInButtonState();
        updateUIState();

        console.log('Successfully signed out');
    } catch (error) {
        console.error('Error during sign-out:', error);
        updateSigninStatus(`Sign-out error: ${error.message}`, true);
    }
}

/**
 * Gets an access token for Entra ID authentication
 */
async function getAccessToken() {
    if (!msalInstance || !msalAccount) {
        throw new Error('Not signed in with Entra ID');
    }

    try {
        const tokenRequest = {
            scopes: ['https://cognitiveservices.azure.com/.default'],
            account: msalAccount
        };

        const response = await msalInstance.acquireTokenSilent(tokenRequest);
        return response.accessToken;
    } catch (error) {
        console.error('Silent token acquisition failed, attempting interactive:', error);

        try {
            const tokenRequest = {
                scopes: ['https://cognitiveservices.azure.com/.default'],
                account: msalAccount
            };
            const response = await msalInstance.acquireTokenPopup(tokenRequest);
            return response.accessToken;
        } catch (interactiveError) {
            console.error('Interactive token acquisition failed:', interactiveError);
            throw new Error('Failed to acquire access token');
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
    const aboutBtn = document.getElementById('aboutBtn');
    const viewInstructionsBtn = document.getElementById('viewInstructionsBtn');
    const chatMessages = document.getElementById('chatMessages');

    if (isConfigured) {
        userInput.disabled = false;
        sendBtn.disabled = false;
        attachBtn.disabled = false;
        micBtn.disabled = !recognition; // Only enable if speech recognition is available
        restartBtn.disabled = false;
        aboutBtn.disabled = false;
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

function getSpeechEndpointBase() {
    if (!config.endpoint) {
        throw new Error('A Foundry endpoint is required before using speech services.');
    }

    let parsedEndpoint;
    try {
        parsedEndpoint = new URL(config.endpoint);
    } catch (error) {
        throw new Error('The Foundry endpoint is not a valid URL.');
    }

    const speechHostname = parsedEndpoint.hostname.replace(
        '.services.ai.azure.com',
        '.cognitiveservices.azure.com'
    );

    if (speechHostname === parsedEndpoint.hostname) {
        throw new Error('The Foundry endpoint must use the services.ai.azure.com host pattern to derive the speech endpoint.');
    }

    return `${parsedEndpoint.protocol}//${speechHostname}`;
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
        const speechEndpoint = getSpeechEndpointBase();

        // Build headers based on authentication mode
        const requestHeaders = {
            'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
            'Accept': 'application/json'
        };

        if (config.authMode === 'entra') {
            const accessToken = await getAccessToken();
            requestHeaders['Authorization'] = `Bearer ${accessToken}`;
        } else {
            requestHeaders['Ocp-Apim-Subscription-Key'] = config.apiKey;
        }

        const candidateUrls = [
            `${speechEndpoint}/stt/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed`,
            `${speechEndpoint}/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed`
        ];

        let response = null;
        let errorText = '';

        for (const url of candidateUrls) {
            response = await fetch(url, {
                method: 'POST',
                headers: requestHeaders,
                body: audioBlob
            });

            if (response.ok) {
                break;
            }

            errorText = await response.text();
            console.error('Speech API error:', errorText);

            if (response.status !== 404) {
                throw new Error(`Transcription failed: ${response.status} - ${errorText}`);
            }
        }

        if (!response || !response.ok) {
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
    isSpeaking = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.textContent = '⬜';
    sendBtn.classList.add('speaking');
    sendBtn.title = 'Stop speech';

    try {
        // Strip HTML tags and convert to plain text using DOMParser (safer than innerHTML)
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        let plainText = doc.body.textContent || '';

        if (!plainText.trim()) {
            resetSendButton();
            return;
        }

        // Replace URLs with speakable format
        plainText = plainText.replace(/https?:\/\/(?:www\.)?[^\s]+/gi, function (url) {
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
        plainText = plainText.replace(/www\.[^\s]+/gi, function (url) {
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

        const speechEndpoint = getSpeechEndpointBase();
        const url = `${speechEndpoint}/tts/cognitiveservices/v1`;

        // Build SSML with Ava DragonHD voice
        const ssml = `<speak version="1.0" xml:lang="en-US">
  <voice name="en-US-Ava:DragonHDLatestNeural">
    ${escapeXml(plainText)}
  </voice>
</speak>`;

        // Build headers based on authentication mode
        const headers = {
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        };

        if (config.authMode === 'entra') {
            const accessToken = await getAccessToken();
            headers['Authorization'] = `Bearer ${accessToken}`;
        } else {
            headers['Ocp-Apim-Subscription-Key'] = config.apiKey;
        }

        let response;
        try {
            response = await fetch(url, {
                method: 'POST',
                headers: headers,
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
        configHeader.addEventListener('keydown', function (e) {
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

    // About button
    const aboutBtn = document.getElementById('aboutBtn');
    if (aboutBtn) {
        aboutBtn.addEventListener('click', showAbout);
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

    // Close about button
    const closeAboutBtn = document.getElementById('closeAboutBtn');
    if (closeAboutBtn) {
        closeAboutBtn.addEventListener('click', closeAbout);
    }

    // Close Entra help button
    const closeEntraHelpBtn = document.getElementById('closeEntraHelpBtn');
    if (closeEntraHelpBtn) {
        closeEntraHelpBtn.addEventListener('click', closeEntraHelp);
    }

    // Entra help button
    const entraHelpBtn = document.getElementById('entra-help-btn');
    if (entraHelpBtn) {
        entraHelpBtn.addEventListener('click', showEntraHelp);
    }

    // Auth mode toggle
    const authModeToggle = document.getElementById('auth-mode-toggle');
    if (authModeToggle) {
        authModeToggle.addEventListener('change', handleAuthModeToggle);
    }

    // Entra ID Sign-in/Sign-out button
    const entraSigninBtn = document.getElementById('entra-signin-btn');
    if (entraSigninBtn) {
        entraSigninBtn.addEventListener('click', function () {
            if (msalAccount) {
                signOutEntraID();
            } else {
                signInWithEntraID();
            }
        });
    }

    // Monitor Entra ID credential inputs to enable/disable sign-in button
    const entraClientId = document.getElementById('entra-client-id');
    const entraTenantId = document.getElementById('entra-tenant-id');
    if (entraClientId && entraTenantId) {
        entraClientId.addEventListener('input', updateSignInButtonState);
        entraTenantId.addEventListener('input', updateSignInButtonState);
    }

    // Attach button
    const attachBtn = document.getElementById('attachBtn');
    if (attachBtn) {
        attachBtn.addEventListener('click', function () {
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
        userInput.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            const entraHelpModal = document.getElementById('entraHelpModal');
            if (entraHelpModal && entraHelpModal.style.display === 'flex') {
                closeEntraHelp();
                return;
            }
            const aboutModal = document.getElementById('aboutModal');
            if (aboutModal && aboutModal.style.display === 'flex') {
                closeAbout();
                return;
            }
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
        reader.onload = function (e) {
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
        const errorMsg = error.isContentFilter
            ? error.message
            : 'Sorry, there was an error processing your request: ' + error.message;

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
        store: true,
        tools: [
            { type: "web_search" }
        ],
        tool_choice: "auto"
    };


    // Include previous response ID for conversation continuity
    if (previousResponseId) {
        requestBody.previous_response_id = previousResponseId;
    }

    // Build headers based on authentication mode
    const headers = {
        'Content-Type': 'application/json'
    };

    if (config.authMode === 'entra') {
        // Use Bearer token for Entra ID authentication
        const accessToken = await getAccessToken();
        headers['Authorization'] = `Bearer ${accessToken}`;
    } else {
        // Use API key authentication
        headers['api-key'] = config.apiKey;
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const errorText = await response.text();
        const contentFilterError = parseContentFilterError(errorText);

        if (contentFilterError) {
            throw contentFilterError;
        }

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

function parseContentFilterError(errorText) {
    try {
        const parsedError = JSON.parse(errorText);
        const errorDetails = parsedError?.error;
        const contentFilters = errorDetails?.content_filters;
        const hasBlockedContentFilter = Array.isArray(contentFilters)
            && contentFilters.some(filter => filter?.blocked);
        const mentionsPolicyBlock = typeof errorDetails?.message === 'string'
            && errorDetails.message.includes('content management policy');

        if (errorDetails?.code === 'content_filter' || hasBlockedContentFilter || mentionsPolicyBlock) {
            const error = new Error(CONTENT_FILTER_MESSAGE);
            error.isContentFilter = true;
            return error;
        }
    } catch (parseError) {
        return null;
    }

    return null;
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

/**
 * Shows the Entra ID help modal
 */
function showEntraHelp() {
    const modal = document.getElementById('entraHelpModal');
    const closeBtn = document.getElementById('closeEntraHelpBtn');
    modal.style.display = 'flex';
    if (closeBtn) {
        closeBtn.focus();
    }
}

/**
 * Closes the Entra ID help modal
 */
function closeEntraHelp() {
    const modal = document.getElementById('entraHelpModal');
    const entraHelpBtn = document.getElementById('entra-help-btn');
    modal.style.display = 'none';
    if (entraHelpBtn) {
        entraHelpBtn.focus();
    }
}
