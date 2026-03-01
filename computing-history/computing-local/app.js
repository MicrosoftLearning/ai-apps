import { Wllama } from '@wllama/wllama';

const chatContainer = document.getElementById('chat-messages');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const imageUpload = document.getElementById('image-upload');
const micBtn = document.getElementById('mic-btn');

// State
let model = null;
let featureExtractor = null;
let pendingFile = null;
let isVoiceInput = false; // Tracks if current message was spoken
let isResponding = false; // Tracks if bot is currently responding
let shouldStopResponse = false; // Flag to cancel ongoing response
let wllama = null; // Wllama instance for text generation
let wllamaReady = false; // Track if wllama is initialized
let mobilenetReady = false; // Track if MobileNet is initialized
let conversationHistory = []; // Track conversation for context
let inappropriateWords = []; // Loaded from moderation file
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

/**
 * Initializes the application by loading both ML models in parallel
 */
async function init() {
    // Show loading overlay
    updateLoadingStatus('mobilenet', 'loading', 'Loading...');
    updateLoadingStatus('smollm', 'loading', 'Loading...');
    
    // Load both models in parallel
    const mobilenetPromise = loadModel();
    const wllamaPromise = initWllama();
    const moderationPromise = loadInappropriateWords();
    
    try {
        await Promise.all([mobilenetPromise, wllamaPromise, moderationPromise]);
        
        // Both models loaded successfully
        hideLoadingOverlay();
    } catch (e) {
        console.error("Initialization error:", e);
        // Show error but still try to hide overlay after a delay
        setTimeout(() => {
            hideLoadingOverlay();
            addMessage(`Error loading models: ${e.message}. Some features may be unavailable.`, "bot");
        }, 2000);
    }
}

function reverseWord(text) {
    return text.split('').reverse().join('');
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function loadInappropriateWords() {
    try {
        const response = await fetch('./moderation/prohibited-words.txt');
        if (!response.ok) throw new Error('Failed to load inappropriate words');

        const reversedWordsText = await response.text();
        inappropriateWords = reversedWordsText
            .split(/\r?\n/)
            .map(word => word.trim())
            .filter(word => word.length > 0)
            .map(word => reverseWord(word.toLowerCase()));

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
 * Initializes the Wllama language model for text generation
 * @throws {Error} If Wllama initialization fails
 */
async function initWllama() {
    try {
        console.log("Initializing wllama...");
        updateLoadingStatus('smollm', 'loading', '10%');
        
        // Configure WASM paths for CDN
        const CONFIG_PATHS = {
            'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm',
            'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm',
        };
        
        // Initialize wllama with CDN-hosted WASM files
        wllama = new Wllama(CONFIG_PATHS);
        updateLoadingStatus('smollm', 'loading', '20%');
        
        // Load model from HuggingFace with optimized settings
        await wllama.loadModelFromHF(
            'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
            'smollm2-360m-instruct-q8_0.gguf',
            {
                n_ctx: 768,      // Smaller context for faster processing
                n_threads: 1,     // Single thread can be more stable
                progressCallback: ({ loaded, total }) => {
                    const progress = Math.round((loaded / total) * 100);
                    const adjustedProgress = Math.round(20 + (progress * 0.8)); // 20% to 100%
                    updateLoadingStatus('smollm', 'loading', `${adjustedProgress}%`);
                    console.log(`Loading wllama: ${progress}%`);
                }
            }
        );
        
        wllamaReady = true;
        updateLoadingStatus('smollm', 'ready', '100%');
        console.log("Wllama initialized successfully");
    } catch (error) {
        console.error('Failed to initialize wllama:', error);
        updateLoadingStatus('smollm', 'error', 'Failed');
        wllamaReady = false;
        throw error;
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
function addMessage(text, sender, imageUrl = null) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.setAttribute('role', 'article');
    div.setAttribute('aria-label', `${sender === 'user' ? 'User' : 'Assistant'} message`);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    // Sanitize text to prevent XSS attacks
    if (typeof DOMPurify !== 'undefined') {
        bubble.innerHTML = DOMPurify.sanitize(text, {
            ALLOWED_TAGS: ['b', 'i', 'br', 'small', 'a'],
            ALLOWED_ATTR: ['href', 'target', 'style'],
            ALLOW_DATA_ATTR: false
        });
    } else {
        // Fallback to textContent if DOMPurify not available to prevent XSS
        bubble.textContent = text;
    }
    
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
    if (sender === 'bot' && isVoiceInput) {
        speakText(bubble);
        isVoiceInput = false; // Reset after speaking
    } else if (sender === 'bot') {
        // If no speech, we can end the response state
        endResponse();
    }
}

/**
 * Scrolls the chat container to the bottom
 */
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
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
    sendBtn.textContent = '■';
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
            addMessage("I'm sorry, I can't help with that. I can only help with information about the history of computing.", "bot");
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

    // 3. Text Only Response (Language model, eBay, or Bing search)
    const lowerText = text.toLowerCase();
    const isEbay = lowerText.includes('ebay') || lowerText.includes('for sale') || lowerText.includes('buy');
    const isBing = lowerText.includes('bing') || lowerText.includes('search') || lowerText.includes('find');

    const keywords = extractKeywords(text);
    if (!keywords) {
        addMessage("Please enter a more specific query.", "bot");
        return;
    }

    if (isEbay) {
        addMessage(`Searching eBay for <b>"${keywords}"</b>...`, "bot");
        const url = `https://www.ebay.com/sch/i.html?_nkw=${keywords.replace(/ /g, '+')}`;
        // Brief delay for effect
        setTimeout(() => {
            if (!checkStopResponse()) {
                addMessage(`Found it! <a href="${url}" target="_blank" style="color: #4ade80; text-decoration: underline;">Click here to see results for ${keywords}</a>`, "bot");
            }
        }, 600);
        return;
    }

    if (isBing) {
        addMessage(`Searching Bing for <b>"${keywords}"</b>...`, "bot");
        const url = `https://www.bing.com/search?q=${keywords.replace(/ /g, '+')}`;
        // Brief delay for effect
        setTimeout(() => {
            if (!checkStopResponse()) {
                addMessage(`Found it! <a href="${url}" target="_blank" style="color: #4ade80; text-decoration: underline;">Click here to see results for ${keywords}</a>`, "bot");
            }
        }, 600);
        return;
    }

    showTyping();

    try {
        const summary = await generateComputingInfo(text);
        removeTyping();

        if (checkStopResponse()) return;

        if (summary) {
            addMessage(`${summary}`, "bot");
            // Store in conversation history (truncated to first sentence)
            conversationHistory.push({ 
                user: truncateToFirstSentence(text), 
                assistant: truncateToFirstSentence(summary) 
            });
            // Keep only last 2 exchanges to avoid context overflow
            if (conversationHistory.length > 2) {
                conversationHistory.shift();
            }
        } else {
            addMessage(`I'm sorry. I don't know about that topic. Try uploading an image!`, "bot");
        }
    } catch (e) {
        removeTyping();
        if (checkStopResponse()) return;
        addMessage("Sorry, I had trouble searching via text. " + e.message, "bot");
    }
}

/**
 * Extracts keywords from text by removing stopwords
 * @param {string} text - The text to extract keywords from
 * @returns {string} Space-separated keywords
 */
function extractKeywords(text) {
    // Remove punctuation and split
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);

    // Filter using global STOPWORDS
    return words.filter(w => !STOPWORDS.has(w) && w.length > 0).join(' ');
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
        const lowerText = userText.toLowerCase();

        // Class 6: Unknown - Simple "don't know" message
        if (classIndex === 6) {
            addMessage("I'm sorry. I don't know what this is. I can only recognize computers and circuit boards.", "bot");
            return;
        }

        // Class 5: Printed Circuit Board - Show prediction + OCR
        if (classIndex === 5) {
            const confidence = (topMatch.probability * 100).toFixed(1);
            let reply = `I am <b>${confidence}%</b> sure this is a <b>${topMatch.className}</b>.`;
            
            // Perform OCR using the same approach as info-extractor
            showTyping();
            try {
                addMessage(reply, "bot");
                addMessage("Scanning for text...", "bot");
                
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
                
                if (!rawText || rawText.trim().length === 0) {
                    addMessage("I couldn't extract any text from the board.", "bot");
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
                        addMessage(`The following details are printed on the board:<br><br><i>${cleanText}</i>`, "bot");
                    } else {
                        addMessage("I couldn't extract any text from the board.", "bot");
                    }
                }
            } catch (e) {
                console.error("OCR Failed", e);
                removeTyping();
                addMessage("I couldn't extract any text from the board.", "bot");
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

        // Check if user wants eBay search
        const isEbay = lowerText.includes('ebay') || lowerText.includes('for sale') || lowerText.includes('buy');

        // Classes 0, 1, 2, 3, 4: eBay search or information
        if ([0, 1, 2, 3, 4].includes(classIndex)) {
            if (isEbay) {
                // eBay search
                addMessage(reply, "bot");
                addMessage(`Searching eBay for <b>"${topMatch.className}"</b>...`, "bot");
                const url = `https://www.ebay.com/sch/i.html?_nkw=${topMatch.className.replace(/ /g, '+')}`;
                setTimeout(() => {
                    if (!checkStopResponse()) {
                        addMessage(`Found it! <a href="${url}" target="_blank" style="color: #4ade80; text-decoration: underline;">Click here to search eBay for ${topMatch.className}</a>`, "bot");
                    }
                }, 600);
                return;
            } else {
                // AI-generated info for classes 0, 1, 2, 3
                if ([0, 1, 2, 3].includes(classIndex)) {
                    showTyping();
                    try {
                        const summary = await generateComputingInfo(topMatch.className);
                        if (checkStopResponse()) {
                            removeTyping();
                            return;
                        }
                        if (summary) {
                            reply += `<br>${summary}`;
                        }
                    } catch (e) {
                        console.warn("Info generation failed", e);
                    } finally {
                        removeTyping();
                    }
                }
                
                // Class 4: Computer - Add uncertainty message
                if (classIndex === 4) {
                    reply += `<br><br>Unfortunately, I'm not sure what kind of computer this is.`;
                }
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
 * Generates computing-related information using Wllama LLM
 * @param {string} query - The query to generate information about
 * @returns {Promise<string|null>} Generated text or null if unavailable
 */
async function generateComputingInfo(query) {
    // If wllama is not ready, return a fallback message
    if (!wllamaReady || !wllama) {
        console.warn("Wllama not ready, skipping generation");
        return null;
    }
    
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
        
        // Add current user query (full text, not keywords)
        chatMLPrompt += '<|im_start|>user\n';
        chatMLPrompt += query + '\n';
        chatMLPrompt += '<|im_end|>\n\n';
        chatMLPrompt += '<|im_start|>assistant\n';
        
        console.log('Generating info for:', query);
        
        // Generate response
        let responseText = '';
        const completion = await wllama.createCompletion(chatMLPrompt, {
            nPredict: 250,  // Allow for more complete responses
            sampling: {
                temp: 0.3,  // Lower temperature for more factual, less creative responses
                top_k: 40,
                top_p: 0.9,
                penalty_repeat: 1.1
            },
            stopTokens: ['<|im_end|>', '<|im_start|>'],
            stream: true
        });
        
        for await (const chunk of completion) {
            if (chunk.currentText) {
                responseText = chunk.currentText;
            }
        }
        
        // Clear KV cache after generation to free memory
        // Suppress munmap warnings - these are harmless WASM memory management messages
        try {
            await wllama.kvClear();
        } catch (error) {
            // Silently ignore - kvClear can throw harmless warnings
        }
        
        // Clean up the response
        responseText = responseText.trim();
        
        // Remove incomplete last sentence (doesn't end with . ? !)
        if (responseText && !responseText.match(/[.!?]$/)) {
            // Find the last complete sentence
            const lastCompleteMatch = responseText.match(/(.*[.!?])/);
            if (lastCompleteMatch) {
                responseText = lastCompleteMatch[1].trim();
            }
        }
        
        // If response is too short or empty, return null
        if (!responseText || responseText.length < 10) {
            return null;
        }
        
        return responseText;
        
    } catch (error) {
        console.error('Error generating info:', error);
        // Clear cache on error (suppress warnings)
        try {
            await wllama.kvClear();
        } catch (e) {
            // Silently ignore kvClear errors
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

// Voice Input (Speech-to-Text)
micBtn.addEventListener('click', handleVoiceInput);

/**
 * Handles voice input using Web Speech API
 */
function handleVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        addMessage("Sorry! Speech input is not currently available.", "bot");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    micBtn.classList.add('listening');

    recognition.start();

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        textInput.value = transcript;
        isVoiceInput = true;
        handleSend();
    };

    recognition.onerror = (event) => {
        micBtn.classList.remove('listening');
        addMessage("Sorry! Speech input is not currently available.", "bot");
        console.error("Speech recognition error:", event.error);
    };

    recognition.onend = () => {
        micBtn.classList.remove('listening');
    };
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
    sendBtn.textContent = '■';
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
        
        // Clear model's KV cache to completely reset context
        if (wllama && wllamaReady) {
            try {
                await wllama.kvClear();
                console.log('Model KV cache cleared');
            } catch (error) {
                // Silently ignore - kvClear can throw harmless warnings
                console.debug('KV cache clear warning (harmless):', error);
            }
        }
    }
}

/**
 * Shows the app details modal with focus management
 */
function showAppDetails() {
    const modal = document.getElementById('appDetailsModal');
    const closeBtn = document.getElementById('closeAppDetailsBtn');
    
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
document.addEventListener('click', function(event) {
    const modal = document.getElementById('appDetailsModal');
    if (event.target === modal) {
        closeAppDetails();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modal = document.getElementById('appDetailsModal');
        if (modal.style.display === 'flex') {
            closeAppDetails();
        }
    }
});

// Start
init();
