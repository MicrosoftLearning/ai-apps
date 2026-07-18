import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/index.js";

const WASM_PATHS = {
    default: "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.1/esm/wasm/wllama.wasm"
};

const MODEL_REPO = "bartowski/Phi-3.5-mini-instruct-GGUF";
const MODEL_QUANT = "Q4_K_M";

// Information Extractor Application
// Extract structured information from images

// Utility function to escape HTML and prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class InfoExtractorApp {
    constructor() {
        this.uploadedImages = [];
        this.selectedImageIndex = -1;
        this.ocrData = null;
        this.extractedFields = null;
        this.wllama = null;
        this.isModelLoaded = false;
        this.useAI = true; // Default to using AI if available
        this.isLoadingModel = false; // Track if model is currently loading
        this.cancelModelLoad = false; // Flag to cancel model loading
        this.gpuFailed = false; // True after a GPU inference failure; forces CPU-only on next load
        this.wllamaUsedGPU = false; // True when the loaded model is using GPU acceleration
        this.wllamaShouldFailoverToBasic = false; // Set to true when wllama fails, triggers failover to pattern-based mode
        this.debugConfig = { enabled: false, forceWllamaGenerationFail: false }; // Debug flags for testing
        this.analysisMode = 'ocr-read'; // Default to OCR/Read mode

        // Zoom functionality
        this.zoomLevel = 1.0;
        this.minZoom = 0.1;
        this.maxZoom = 5.0;
        this.zoomStep = 0.2;

        this.initializeElements();
        this.bindEvents();
        this.applyTheme();
        this.debugConfig = this.parseDebugConfig();
        this.showModelLoading();
        this.initializeModel();
    }

    parseDebugConfig() {
        const params = new URLSearchParams(window.location.search);
        const enabled = params.get('debug') === 'true';
        const forceWllamaGenerationFail = params.get('forceWllamaFail') === 'true';
        if (enabled) {
            console.log('[DEBUG MODE] Debug config:', { enabled, forceWllamaGenerationFail });
        }
        return { enabled, forceWllamaGenerationFail };
    }

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

    showModelLoading() {
        // Show the model loading overlay
        this.modelLoadingSection.style.display = 'flex';

        // Disable the UI
        this.uploadSidebar.classList.add('ui-disabled');
        this.resultsPanel.classList.add('ui-disabled');
        this.analyzeBtn.disabled = true;
        this.aiToggle.disabled = true;

        // Update loading text and progress
        this.updateModelLoadingProgress(0, 'Initializing model...');
    }

    hideModelLoading() {
        // Hide the model loading overlay
        this.modelLoadingSection.style.display = 'none';

        // Enable the UI
        this.uploadSidebar.classList.remove('ui-disabled');
        this.resultsPanel.classList.remove('ui-disabled');

        // Enable analyze button if image is selected (OCR still works without model)
        if (this.selectedImageIndex >= 0) {
            this.analyzeBtn.disabled = false;
        }

        // Always re-enable AI toggle when hiding loading screen
        // It will be explicitly disabled in error handler if model failed with an error
        this.aiToggle.disabled = false;
    }

    updateModelLoadingProgress(percentage, text) {
        this.modelProgressFill.style.width = `${percentage}%`;
        this.modelLoadingText.textContent = text;

        // Update ARIA attributes
        const progressBar = this.modelLoadingSection.querySelector('.model-progress-bar');
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.setAttribute('aria-valuetext', `${Math.round(percentage)}% - ${text}`);
    }

    initializeElements() {
        // Get all DOM elements
        this.imageInput = document.getElementById('imageInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.thumbnailsContainer = document.getElementById('thumbnailsContainer');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.analysisModeSel = document.getElementById('analysisMode');
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.imageContainer = document.getElementById('imageContainer');
        this.selectedImage = document.getElementById('selectedImage');
        this.annotatedCanvas = document.getElementById('annotatedCanvas');
        this.imagePlaceholder = document.getElementById('imagePlaceholder');
        this.fieldsList = document.getElementById('fieldsList');
        this.fieldsTab = document.getElementById('fieldsTab');
        this.resultTab = document.getElementById('resultTab');
        this.ocrResult = document.getElementById('ocrResult');
        this.resultContent = document.getElementById('resultContent');
        this.errorSection = document.getElementById('errorSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');

        // Model loading elements
        this.modelLoadingSection = document.getElementById('modelLoadingSection');
        this.modelLoadingText = document.getElementById('modelLoadingText');
        this.modelProgressFill = document.getElementById('modelProgressFill');
        this.cancelModelLoadLink = document.getElementById('cancelModelLoadLink');

        // AI toggle
        this.aiToggle = document.getElementById('aiToggle');

        // Theme toggle
        this.themeToggle = document.getElementById('theme-toggle');

        // Zoom controls
        this.zoomInBtn = document.getElementById('zoomInBtn');
        this.zoomOutBtn = document.getElementById('zoomOutBtn');
        this.resetZoomBtn = document.getElementById('resetZoomBtn');
        this.zoomLevelDisplay = document.getElementById('zoomLevel');
        this.imageViewer = document.querySelector('.image-viewer');

        // Tab buttons
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');

        // UI containers for disabling
        this.uploadSidebar = document.querySelector('.upload-sidebar');
        this.resultsPanel = document.querySelector('.results-panel');
    }

    bindEvents() {
        // File input and upload area
        this.uploadArea.addEventListener('click', () => {
            if (!this.uploadArea.classList.contains('disabled')) {
                this.triggerFileInput();
            }
        });
        this.uploadArea.addEventListener('keydown', (e) => {
            if (!this.uploadArea.classList.contains('disabled') && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                this.triggerFileInput();
            }
        });
        this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeCurrentImage());

        // Tab switching
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Zoom controls
        this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        this.resetZoomBtn.addEventListener('click', () => this.resetZoom());

        // Mouse wheel zoom
        this.imageViewer.addEventListener('wheel', (e) => this.handleWheelZoom(e));

        // Retry button
        this.retryBtn.addEventListener('click', () => this.hideError());

        // AI toggle
        this.aiToggle.addEventListener('change', (e) => this.handleAIToggle(e));

        // Theme toggle
        this.themeToggle.addEventListener('change', () => this.handleThemeToggle());

        // Cancel model loading
        this.cancelModelLoadLink.addEventListener('click', (e) => {
            e.preventDefault();
            this.cancelModelLoading();
        });

        // Analysis mode change
        this.analysisModeSel.addEventListener('change', (e) => this.handleModeChange(e));

        // Drag and drop functionality
        this.setupDragAndDrop();
    }

    async handleAIToggle(event) {
        this.useAI = event.target.checked;
        console.log('AI toggle changed. Use AI:', this.useAI);

        // If turning on AI and model is not loaded, try to load it
        if (this.useAI && !this.isModelLoaded && !this.isLoadingModel) {
            console.log('AI enabled but model not loaded. Attempting to load model...');
            this.showModelLoading();
            await this.initializeModel();
        } else if (!this.useAI) {
            console.log('AI disabled, using basic mode');
        }
    }

    async handleModeChange(event) {
        const newMode = event.target.value;
        console.log('Analysis mode changed:', newMode);
        this.analysisMode = newMode;

        // Clear all uploaded images
        this.uploadedImages.forEach(img => URL.revokeObjectURL(img.url));
        this.uploadedImages = [];
        this.thumbnailsContainer.innerHTML = '';
        this.selectedImageIndex = -1;
        this.imageContainer.style.display = 'none';
        this.imagePlaceholder.style.display = 'flex';
        this.analyzeBtn.disabled = true;
        this.resetResults();

        // Load appropriate sample and configure UI
        if (newMode === 'ocr-read') {
            // OCR/Read mode: load biz-card.png and hide Fields tab
            this.hideFieldsTab();
            await this.loadSampleImage('./biz-card.png', 'biz-card.png');
        } else if (newMode === 'receipt-fields') {
            // Receipt fields mode: load receipt.png and show Fields tab
            this.showFieldsTab();
            await this.loadSampleImage('./receipt.png', 'receipt.png');
        }
    }

    hideFieldsTab() {
        const fieldsTabBtn = document.querySelector('[data-tab="fields"]');
        if (fieldsTabBtn) {
            fieldsTabBtn.style.display = 'none';
        }
        // Switch to Result tab if currently on Fields tab
        if (this.fieldsTab.classList.contains('active')) {
            this.switchTab('result');
        }
    }

    showFieldsTab() {
        const fieldsTabBtn = document.querySelector('[data-tab="fields"]');
        if (fieldsTabBtn) {
            fieldsTabBtn.style.display = '';
        }
        // Switch back to Fields tab
        this.switchTab('fields');
    }

    applyTheme() {
        // Load saved theme from localStorage or use system preference
        const saved = localStorage.getItem('info-extractor-theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const dark = saved ? saved === 'dark' : prefersDark;
        this.themeToggle.checked = dark;
        document.body.classList.toggle('dark', dark);
    }

    handleThemeToggle() {
        const dark = this.themeToggle.checked;
        document.body.classList.toggle('dark', dark);
        localStorage.setItem('info-extractor-theme', dark ? 'dark' : 'light');
    }

    setupDragAndDrop() {
        const dropZone = this.uploadArea;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                if (!dropZone.classList.contains('disabled')) {
                    dropZone.classList.add('dragover');
                }
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        if (this.uploadArea.classList.contains('disabled')) {
            return;
        }
        const dt = e.dataTransfer;
        const files = dt.files;
        this.processFiles(files);
    }

    triggerFileInput() {
        this.imageInput.click();
    }

    handleFileSelect(event) {
        const files = event.target.files;
        this.processFiles(files);
    }

    processFiles(files) {
        const validFiles = Array.from(files).filter(file => this.isValidImageFile(file));

        if (validFiles.length === 0) {
            this.showError('Please select valid image files (.jpg, .jpeg, or .png)');
            return;
        }

        if (this.uploadedImages.length + validFiles.length > 5) {
            this.showError('Maximum 5 images allowed. Please remove some images first.');
            return;
        }

        validFiles.forEach(file => this.addImage(file));
    }

    isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        return validTypes.includes(file.type) && file.size <= 10 * 1024 * 1024; // 10MB limit
    }

    async addImage(file, isSample = false) {
        const imageUrl = URL.createObjectURL(file);
        const imageData = {
            file: file,
            url: imageUrl,
            name: file.name,
            isSample: isSample
        };

        this.uploadedImages.push(imageData);
        this.createThumbnail(imageData, this.uploadedImages.length - 1);

        // Select the newly uploaded image automatically
        this.selectImage(this.uploadedImages.length - 1);
    }

    createThumbnail(imageData, index) {
        const thumbnailItem = document.createElement('div');
        thumbnailItem.className = 'thumbnail-item';
        thumbnailItem.setAttribute('role', 'listitem');
        thumbnailItem.setAttribute('tabindex', '0');
        const ariaLabel = imageData.isSample
            ? `Sample receipt image ${index + 1}: ${imageData.name}`
            : `Receipt image ${index + 1}: ${imageData.name}`;
        thumbnailItem.setAttribute('aria-label', ariaLabel);

        const sampleOverlay = imageData.isSample ? '<div class="sample-overlay">[Sample]</div>' : '';
        const escapedName = escapeHtml(imageData.name);

        thumbnailItem.innerHTML = `
            <img src="${escapeHtml(imageData.url)}" alt="Thumbnail of ${escapedName}" />
            ${sampleOverlay}
            <button class="thumbnail-remove" onclick="app.removeImage(${index})" 
                    aria-label="Remove ${escapedName}">×</button>
        `;

        thumbnailItem.addEventListener('click', (e) => {
            if (!e.target.classList.contains('thumbnail-remove') && !thumbnailItem.classList.contains('disabled')) {
                this.selectImage(index);
            }
        });

        // Add keyboard navigation
        thumbnailItem.addEventListener('keydown', (e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !e.target.classList.contains('thumbnail-remove') && !thumbnailItem.classList.contains('disabled')) {
                e.preventDefault();
                this.selectImage(index);
            }
        });

        this.thumbnailsContainer.appendChild(thumbnailItem);
    }

    selectImage(index) {
        if (index < 0 || index >= this.uploadedImages.length) return;

        // Update selection state
        this.selectedImageIndex = index;

        // Update thumbnail selection
        document.querySelectorAll('.thumbnail-item').forEach((item, i) => {
            item.classList.toggle('selected', i === index);
        });

        // Display selected image
        const imageData = this.uploadedImages[index];
        this.selectedImage.src = imageData.url;
        this.selectedImage.alt = `Receipt image: ${imageData.name}`;
        this.imageContainer.style.display = 'flex';
        this.imagePlaceholder.style.display = 'none';
        this.annotatedCanvas.style.display = 'none';
        this.selectedImage.style.display = 'block';

        // Announce to screen readers
        this.imagePlaceholder.textContent = `Selected image: ${imageData.name}`;
        this.imagePlaceholder.setAttribute('aria-live', 'polite');

        // Enable analyze button (OCR works even without AI model)
        this.analyzeBtn.disabled = false;

        // Set zoom to fit the image in the viewing area
        this.fitImageToView();

        // Reset results
        this.resetResults();
    }

    fitImageToView() {
        // Wait for image to load before calculating zoom
        if (this.selectedImage.complete && this.selectedImage.naturalWidth > 0) {
            this.calculateFitZoom();
        } else {
            this.selectedImage.onload = () => {
                this.calculateFitZoom();
            };
        }
    }

    calculateFitZoom() {
        const viewer = this.imageViewer;
        const img = this.selectedImage;

        if (!viewer || !img.naturalWidth || !img.naturalHeight) return;

        const viewerWidth = viewer.clientWidth;
        const viewerHeight = viewer.clientHeight;
        const imageWidth = img.naturalWidth;
        const imageHeight = img.naturalHeight;

        // Calculate zoom to fit within viewer (with some padding)
        const padding = 40; // pixels of padding
        const scaleX = (viewerWidth - padding) / imageWidth;
        const scaleY = (viewerHeight - padding) / imageHeight;

        // Use the smaller scale to ensure the entire image fits
        let fitZoom = Math.min(scaleX, scaleY, 1.0); // Don't zoom in beyond 100%

        // Clamp to min/max zoom levels
        fitZoom = Math.max(this.minZoom, Math.min(this.maxZoom, fitZoom));

        this.zoomLevel = fitZoom;
        this.updateZoom();
    }

    zoomIn() {
        if (this.zoomLevel < this.maxZoom) {
            this.zoomLevel = Math.min(this.zoomLevel + this.zoomStep, this.maxZoom);
            this.updateZoom();
        }
    }

    zoomOut() {
        if (this.zoomLevel > this.minZoom) {
            this.zoomLevel = Math.max(this.zoomLevel - this.zoomStep, this.minZoom);
            this.updateZoom();
        }
    }

    resetZoom() {
        this.zoomLevel = 1.0;
        this.updateZoom();
    }

    updateZoom() {
        const activeImage = this.selectedImage.style.display !== 'none' ? this.selectedImage : this.annotatedCanvas;
        if (activeImage) {
            activeImage.style.transform = `scale(${this.zoomLevel})`;
            this.zoomLevelDisplay.textContent = `${Math.round(this.zoomLevel * 100)}%`;

            // Update button states
            this.zoomInBtn.disabled = this.zoomLevel >= this.maxZoom;
            this.zoomOutBtn.disabled = this.zoomLevel <= this.minZoom;
        }
    }

    handleWheelZoom(e) {
        e.preventDefault();

        const delta = e.deltaY > 0 ? -1 : 1;
        const zoomChange = delta * 0.1;

        const newZoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoomLevel + zoomChange));
        if (newZoom !== this.zoomLevel) {
            this.zoomLevel = newZoom;
            this.updateZoom();
        }
    }

    removeImage(index) {
        if (index < 0 || index >= this.uploadedImages.length) return;

        // Don't allow removal during analysis
        const removeButtons = document.querySelectorAll('.thumbnail-remove');
        if (removeButtons[index] && removeButtons[index].disabled) {
            return;
        }

        // Clean up URL
        URL.revokeObjectURL(this.uploadedImages[index].url);

        // Remove from array
        this.uploadedImages.splice(index, 1);

        // Rebuild thumbnails
        this.thumbnailsContainer.innerHTML = '';
        this.uploadedImages.forEach((imageData, i) => {
            this.createThumbnail(imageData, i);
        });

        // Update selection
        if (this.selectedImageIndex === index) {
            if (this.uploadedImages.length > 0) {
                const newIndex = Math.min(index, this.uploadedImages.length - 1);
                this.selectImage(newIndex);
            } else {
                this.selectedImageIndex = -1;
                this.imageContainer.style.display = 'none';
                this.imagePlaceholder.style.display = 'flex';
                this.analyzeBtn.disabled = true;
                this.resetResults();
            }
        } else if (this.selectedImageIndex > index) {
            this.selectedImageIndex--;
        }
    }

    async loadSampleImage(imagePath, imageName) {
        try {
            console.log(`Attempting to load sample image: ${imagePath}`);
            const response = await fetch(imagePath);
            if (!response.ok) {
                console.log('Sample image not found, status:', response.status);
                return;
            }

            console.log('Sample image fetched successfully, creating file object...');
            const blob = await response.blob();
            const file = new File([blob], imageName, { type: 'image/png' });

            console.log('File object created, adding image to app...');
            await this.addImage(file, true);
            console.log('Sample image loaded and displayed successfully');

        } catch (error) {
            console.log('Could not load sample image:', error.message);
            console.error('Full error:', error);
        }
    }

    async loadDefaultReceipt() {
        // Check if default sample is already loaded
        const hasSample = this.uploadedImages.some(img => img.isSample);
        if (hasSample) {
            console.log('Default sample already loaded, skipping...');
            return;
        }

        // Load appropriate sample based on current mode
        if (this.analysisMode === 'ocr-read') {
            // Hide Fields tab for OCR/Read mode
            this.hideFieldsTab();
            await this.loadSampleImage('./biz-card.png', 'biz-card.png');
        } else {
            // Show Fields tab for Receipt fields mode
            this.showFieldsTab();
            await this.loadSampleImage('./receipt.png', 'receipt.png');
        }
    }

    cancelModelLoading() {
        console.log('User cancelled model loading');
        this.cancelModelLoad = true;
        this.isLoadingModel = false;
        this.hideModelLoading();

        // Turn off AI toggle but keep it enabled for future use
        this.aiToggle.checked = false;
        this.useAI = false;

        // Load the default receipt image
        this.loadDefaultReceipt();
    }

    async initializeModel() {
        // Check hardware requirements before attempting to load model
        if (!this.checkHardwareRequirements()) {
            this.isModelLoaded = false;
            this.useAI = false;
            this.hideModelLoading();

            // Disable AI toggle since model cannot load
            if (this.aiToggle) {
                this.aiToggle.disabled = true;
                this.aiToggle.checked = false;
            }

            // Load the default receipt image
            this.loadDefaultReceipt();

            return;
        }

        // Reset cancellation flag and set loading state
        this.cancelModelLoad = false;
        this.isLoadingModel = true;
        this.wllamaUsedGPU = false;

        try {
            console.log('Initializing Phi 3.5-mini (wllama)...');
            this.updateModelLoadingProgress(10, 'Initializing Phi 3.5-mini...');

            if (this.cancelModelLoad) {
                console.log('Model loading cancelled by user');
                return;
            }

            // Detect GPU vendor and skip WebGPU for known-problematic GPUs
            let gpuEnabled = !this.gpuFailed && !!navigator.gpu;
            if (gpuEnabled) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        const info = adapter.info ?? await adapter.requestAdapterInfo?.();
                        const vendor = (info?.vendor || '').toLowerCase();
                        if (vendor.includes('qualcomm') || vendor.includes('adreno')) {
                            // Open bug: ggml-org/llama.cpp#23558 — garbled output on Qualcomm WebGPU
                            console.warn('WebGPU disabled: Qualcomm/Adreno GPU detected. Using CPU.');
                            gpuEnabled = false;
                        } else if (vendor.includes('amd') || vendor.includes('advanced micro')) {
                            // Flashattention bug fixed in wllama 3.2.3+ (llama.cpp PR #23040); app uses 3.1.1
                            console.warn('WebGPU disabled: AMD GPU detected. Using CPU.');
                            gpuEnabled = false;
                        }
                    } else {
                        gpuEnabled = false;
                    }
                } catch (e) {
                    console.warn('Could not query WebGPU adapter info:', e);
                    gpuEnabled = false;
                }
            }

            const useMultiThread = window.crossOriginIsolated === true;
            const availableThreads = navigator.hardwareConcurrency || 4;
            const preferredThreads = useMultiThread ? Math.max(1, availableThreads - 2) : 1;

            const progressCallback = ({ loaded, total }) => {
                if (this.cancelModelLoad) return;
                if (!total) {
                    this.updateModelLoadingProgress(20, 'Loading Phi 3.5-mini...');
                    return;
                }
                const pct = Math.round((loaded / total) * 100);
                this.updateModelLoadingProgress(20 + Math.round(pct * 0.75), `Downloading Phi 3.5-mini: ${pct}%`);
            };

            const modelRef = { repo: MODEL_REPO, quant: MODEL_QUANT };

            const attemptLoad = async (n_gpu_layers, n_threads) => {
                if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
                this.wllama = new Wllama(WASM_PATHS);
                await this.wllama.loadModelFromHF(modelRef, { n_ctx: 768, n_gpu_layers, n_threads, progressCallback });
            };

            const loadWithFallback = async () => {
                if (gpuEnabled) {
                    try {
                        console.log('Attempting GPU load (32 layers)...');
                        this.updateModelLoadingProgress(15, 'Loading with GPU acceleration...');
                        await attemptLoad(32, preferredThreads);
                        this.wllamaUsedGPU = true;
                        console.log('Model loaded with GPU acceleration.');
                        return;
                    } catch (gpuErr) {
                        console.warn('GPU load failed, falling back to CPU:', gpuErr);
                        this.wllamaUsedGPU = false;
                    }
                }
                if (preferredThreads > 1) {
                    try {
                        console.log('Attempting CPU load (multi-thread)...');
                        this.updateModelLoadingProgress(15, 'Loading with CPU (multi-thread)...');
                        await attemptLoad(0, preferredThreads);
                        return;
                    } catch (multiErr) {
                        console.warn('CPU multi-thread load failed, trying single-thread:', multiErr);
                    }
                }
                console.log('Attempting CPU load (single-thread)...');
                this.updateModelLoadingProgress(15, 'Loading with CPU (single-thread)...');
                await attemptLoad(0, 1);
            };

            await loadWithFallback();

            if (this.cancelModelLoad) {
                console.log('Model loading cancelled by user after download');
                if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
                return;
            }

            this.isModelLoaded = true;
            this.isLoadingModel = false;
            this.updateModelLoadingProgress(100, 'Phi 3.5-mini ready!');
            console.log('Phi 3.5-mini loaded successfully');

            setTimeout(() => {
                this.hideModelLoading();
                this.loadDefaultReceipt();
            }, 1000);

        } catch (error) {
            if (!this.cancelModelLoad) {
                console.error('Failed to initialize Phi 3.5-mini:', error);
                this.isModelLoaded = false;
                this.isLoadingModel = false;
                if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }

                this.updateModelLoadingProgress(100, 'Using pattern-based extraction mode...');

                setTimeout(() => {
                    this.hideModelLoading();
                    this.aiToggle.checked = false;
                    this.aiToggle.disabled = true;
                    this.useAI = false;
                    console.log('Fallback mode activated: Using OCR and pattern-matching for field extraction');
                    this.loadDefaultReceipt();
                }, 1000);
            } else {
                console.log('Model loading was cancelled, no error handling needed');
            }
        }
    }

    /**
     * Tears down the current wllama instance and reloads the model on CPU only.
     * Called when GPU inference produces empty output at runtime.
     */
    async _reloadModelOnCpu() {
        this.gpuFailed = true;
        this.wllamaUsedGPU = false;
        this.isModelLoaded = false;
        if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }

        const useMultiThread = window.crossOriginIsolated === true;
        const preferredThreads = useMultiThread ? Math.max(1, (navigator.hardwareConcurrency || 4) - 2) : 1;
        const modelRef = { repo: MODEL_REPO, quant: MODEL_QUANT };

        const tryLoad = async (n_threads) => {
            if (this.wllama) { try { await this.wllama.exit(); } catch (_) { } this.wllama = null; }
            this.wllama = new Wllama(WASM_PATHS);
            await this.wllama.loadModelFromHF(modelRef, { n_ctx: 768, n_gpu_layers: 0, n_threads, progressCallback: () => { } });
        };

        if (preferredThreads > 1) {
            try { await tryLoad(preferredThreads); } catch (_) { await tryLoad(1); }
        } else {
            await tryLoad(1);
        }
        this.isModelLoaded = true;
    }

    async analyzeCurrentImage() {
        if (this.selectedImageIndex < 0 || !this.uploadedImages[this.selectedImageIndex]) {
            this.showError('Please select an image first');
            return;
        }

        try {
            this.showProgress();
            this.disableUploadAndSelection();
            const imageData = this.uploadedImages[this.selectedImageIndex];

            // Step 1: OCR with Tesseract
            this.updateProgress(10, 'Extracting text with OCR...');
            await this.performOCR(imageData.file);

            // Check analysis mode
            if (this.analysisMode === 'ocr-read') {
                // OCR/Read mode: Only display OCR results, no field extraction
                console.log('OCR/Read mode: Displaying OCR results only');
                this.updateProgress(90, 'Preparing results...');
                this.displayOCRResultOnly();

                this.updateProgress(100, 'Analysis complete!');
                setTimeout(() => {
                    this.hideProgress();
                    this.enableUploadAndSelection();
                }, 1000);
                return;
            }

            // Receipt fields mode: Continue with field extraction
            // Step 2: Extract fields with LLM (only if model is loaded AND user wants to use AI)
            if (this.isModelLoaded && this.wllama && this.useAI) {
                console.log('Model is loaded and AI is enabled, proceeding with field extraction');
                this.updateProgress(60, 'Extracting field values with AI (may be slow on some devices)...');
                try {
                    await this.extractFields();

                    // Check if failover was triggered during extraction
                    if (this.wllamaShouldFailoverToBasic) {
                        console.log('Failover to pattern-based mode triggered');

                        // Switch to pattern-based mode
                        this.useAI = false;
                        if (this.aiToggle) {
                            this.aiToggle.checked = false;
                        }

                        // Show error notification in results
                        this.updateProgress(60, 'Switching to pattern-based extraction...');

                        // Add error message to results
                        const errorMsg = '<i>I experienced an error using the AI model on this device; so I\'m switching to pattern-based extraction...</i>';

                        // Retry with heuristic extraction
                        console.log('Retrying with pattern-based extraction');
                        this.extractFieldsHeuristic();

                        // Store error message to display with results
                        this.failoverErrorMessage = errorMsg;
                    }
                } catch (aiError) {
                    // If AI extraction fails and failover flag is set, use pattern-based extraction
                    if (this.wllamaShouldFailoverToBasic) {
                        console.log('AI extraction failed, failover flag set, switching to pattern-based mode');

                        // Switch to pattern-based mode
                        this.useAI = false;
                        if (this.aiToggle) {
                            this.aiToggle.checked = false;
                        }

                        this.updateProgress(60, 'Switching to pattern-based extraction...');

                        // Add error message
                        const errorMsg = '<i>I experienced an error using the AI model on this device; so I\'m switching to pattern-based extraction...</i>';

                        // Retry with heuristic extraction
                        console.log('Retrying with pattern-based extraction');
                        this.extractFieldsHeuristic();

                        // Store error message to display with results
                        this.failoverErrorMessage = errorMsg;
                    } else {
                        // Original error handling for non-failover cases
                        console.error('AI extraction failed:', aiError);
                        this.hideProgress();
                        this.enableUploadAndSelection();

                        let aiErrorMessage = 'An error occurred during AI field extraction.\n\n';

                        // Check for specific error types
                        if (aiError.message.includes('memory') || aiError.message.includes('OOM') || aiError.message.includes('allocation')) {
                            aiErrorMessage += 'This may be due to insufficient memory.\n\n';
                        } else {
                            aiErrorMessage += `Error: ${aiError.message}\n\n`;
                        }

                        aiErrorMessage += 'Try disabling the "Use Generative AI" toggle to use pattern-based extraction instead.';

                        this.showError(aiErrorMessage, 'AI Extraction Failed');
                        return;
                    }
                }
            } else {
                if (!this.useAI) {
                    console.log('AI disabled by user, using heuristic field extraction');
                } else {
                    console.log('AI model not loaded, using heuristic field extraction. isModelLoaded:', this.isModelLoaded, 'wllama:', !!this.wllama);
                }
                this.updateProgress(60, 'Extracting fields using pattern matching...');
                this.extractFieldsHeuristic();
            }

            // Step 3: Display results
            this.updateProgress(90, 'Preparing results...');
            this.displayResults();

            this.updateProgress(100, 'Analysis complete!');
            setTimeout(() => {
                this.hideProgress();
                this.enableUploadAndSelection();
            }, 1000);

        } catch (error) {
            console.error('Analysis failed:', error);
            let errorMessage = 'Failed to analyze the image. Please try again.';

            // Provide more specific error messages
            if (error.message.includes('AI model not loaded')) {
                errorMessage = 'AI model is still loading. Please wait a moment and try again.';
            } else if (error.message.includes('No text extracted')) {
                errorMessage = 'No text could be extracted from this image. Please try a clearer image.';
            } else if (error.message.includes('OCR')) {
                errorMessage = 'Failed to read text from the image. Please try a different image.';
            } else if (error.message.includes('model')) {
                errorMessage = 'AI model error. Please refresh the page and try again.';
            }

            this.showError(errorMessage);
            this.hideProgress();
            this.enableUploadAndSelection();
        }
    }

    displayOCRResultOnly() {
        // Show annotated image with bounding boxes
        this.createAnnotatedImage();

        // Display full OCR result
        this.displayOCRResult();

        // Switch to result tab (Fields tab is hidden in OCR/Read mode)
        this.switchTab('result');
    }

    async performOCR(file) {
        try {
            console.log('Starting OCR for file:', file.name, 'Size:', file.size);

            // Initialize Tesseract with progress tracking
            const worker = await Tesseract.createWorker('eng', 1, {
                logger: m => {
                    if (m.status === 'recognizing text') {
                        const ocrProgress = Math.round(m.progress * 40);
                        this.updateProgress(10 + ocrProgress, `OCR: ${Math.round(m.progress * 100)}%`);
                    }
                }
            });

            // Perform OCR
            console.log('Performing OCR recognition...');
            const result = await worker.recognize(file, {}, { blocks: true });
            this.ocrData = result.data;

            // Tesseract.js v6+ no longer provides a flat words array — extract from blocks
            if (this.ocrData.blocks) {
                this.ocrData.words = this.ocrData.blocks
                    .flatMap(block => block.paragraphs
                        .flatMap(para => para.lines
                            .flatMap(line => line.words)));
            }

            console.log('OCR completed. Text length:', this.ocrData.text?.length || 0);
            console.log('OCR text preview:', this.ocrData.text?.substring(0, 200) || 'No text');

            // Clean up
            await worker.terminate();

            if (!this.ocrData.text || this.ocrData.text.trim().length === 0) {
                throw new Error('No text extracted from image');
            }

        } catch (error) {
            console.error('OCR Error details:', error);
            throw new Error('Failed to extract text from image: ' + error.message);
        }
    }

    async extractFields() {
        if (!this.isModelLoaded || !this.wllama) {
            console.error('Model not loaded. Model loaded:', this.isModelLoaded, 'wllama:', !!this.wllama);
            throw new Error('AI model not loaded. Please wait and try again.');
        }

        if (!this.ocrData || !this.ocrData.text.trim()) {
            throw new Error('No text extracted from image');
        }

        try {
            // Truncate OCR text to avoid overflowing the model's context window
            const MAX_OCR_CHARS = 1200;
            const ocrText = this.ocrData.text.length > MAX_OCR_CHARS
                ? this.ocrData.text.substring(0, MAX_OCR_CHARS) + '...'
                : this.ocrData.text;

            const prompt = `The following text was extracted from a scanned receipt:
---
${ocrText}
---
Please identify the most likely values for these fields:
- Vendor
- Vendor-Address
- Vendor-Phone
- Receipt-Date
- Receipt-Time
- Total-spent

Respond as a list of fields with their values.`;

            console.log('Sending prompt to Phi 3.5-mini. Prompt length:', prompt.length);

            // Debug mode: Force failure for testing failover
            if (this.debugConfig.enabled && this.debugConfig.forceWllamaGenerationFail) {
                console.log('[DEBUG MODE] Forcing wllama generation failure');
                throw new Error('Forced failure for testing');
            }

            const abortController = new AbortController();
            const timeoutId = setTimeout(() => {
                console.warn('AI extraction timed out after 120 seconds, aborting');
                abortController.abort();
            }, 120000);

            // Estimate expected output length to scale the progress bar (60→88%)
            const MAX_TOKENS = 200;
            const EXPECTED_CHARS = MAX_TOKENS * 4; // ~4 chars per token
            let accumulatedText = '';

            try {
                const stream = await this.wllama.createChatCompletion({
                    messages: [
                        {
                            role: "system",
                            content: "You are a helpful assistant that extracts structured information from receipt text. Always respond with a clear list of field names and their values."
                        },
                        {
                            role: "user",
                            content: prompt
                        }
                    ],
                    max_tokens: MAX_TOKENS,
                    temperature: 0.1,
                    top_k: 30,
                    top_p: 0.85,
                    repeat_penalty: 1.1,
                    cache_prompt: false,
                    stream: true,
                    abortSignal: abortController.signal,
                });

                for await (const chunk of stream) {
                    if (abortController.signal.aborted) break;
                    const delta = chunk?.choices?.[0]?.delta?.content ?? '';
                    if (delta) {
                        accumulatedText += delta;
                        const pct = Math.min(accumulatedText.length / EXPECTED_CHARS, 1);
                        this.updateProgress(60 + Math.round(pct * 28), `Matching extracted values to fields...`);
                    }
                }
            } finally {
                clearTimeout(timeoutId);
            }

            if (abortController.signal.aborted) {
                throw new Error('AI extraction timed out. Please try again, or disable the AI toggle to use pattern-based extraction instead.');
            }

            console.log('Phi 3.5-mini response:', accumulatedText);

            if (!accumulatedText || accumulatedText.trim().length === 0) {
                throw new Error('Empty response from AI model');
            }

            this.extractedFields = this.parseFieldsFromResponse(accumulatedText);
            console.log('Parsed fields:', this.extractedFields);

        } catch (error) {
            console.error('Phi 3.5-mini extraction failed:', error);

            // Set failover flag and clean up wllama instance
            this.wllamaShouldFailoverToBasic = true;
            if (this.wllama) {
                try {
                    await this.wllama.exit();
                } catch (exitErr) {
                    console.error('Error cleaning up wllama:', exitErr);
                }
                this.wllama = null;
            }
            this.isModelLoaded = false;

            throw new Error('Failed to extract field information: ' + error.message);
        }
    }

    parseFieldsFromResponse(response) {
        // Parse the LLM response to extract field values
        const fields = {
            'Vendor': '',
            'Vendor-Address': '',
            'Vendor-Phone': '',
            'Receipt-Date': '',
            'Receipt-Time': '',
            'Total-spent': ''
        };

        const lines = response.split('\n');

        for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine) continue;

            // Try to match patterns like "- Vendor: Value" or "Vendor: Value"
            // Process in order of specificity to avoid partial matches
            const fieldOrder = ['Vendor-Address', 'Vendor-Phone', 'Receipt-Date', 'Receipt-Time', 'Total-spent', 'Vendor'];

            for (const fieldName of fieldOrder) {
                if (fields[fieldName]) continue; // Skip if already found

                const patterns = [
                    new RegExp(`^\\s*-\\s*${fieldName}\\s*:?\\s*(.+)`, 'i'),
                    new RegExp(`^\\s*\\*\\s*${fieldName}\\s*:?\\s*(.+)`, 'i'),
                    new RegExp(`^\\s*\\d+\\.\\s*${fieldName}\\s*:?\\s*(.+)`, 'i'),
                    new RegExp(`^\\s*${fieldName}\\s*:?\\s*(.+)`, 'i')
                ];

                for (const pattern of patterns) {
                    const match = trimmedLine.match(pattern);
                    if (match && match[1]) {
                        const value = match[1].trim();
                        // Make sure we didn't capture another field name
                        if (!fieldOrder.some(otherField => otherField !== fieldName && value.toLowerCase().includes(otherField.toLowerCase()))) {
                            fields[fieldName] = value;
                            break;
                        }
                    }
                }

                if (fields[fieldName]) break; // Found this field, move to next line
            }
        }

        return fields;
    }

    extractFieldsHeuristic() {
        if (!this.ocrData || !this.ocrData.text.trim()) {
            this.extractedFields = null;
            return;
        }

        console.log('Using heuristic field extraction');
        const text = this.ocrData.text;
        const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
        const allWords = text.split(/\s+/).map(word => word.trim()).filter(word => word.length > 0);

        const fields = {
            'Vendor': '',
            'Vendor-Address': '',
            'Vendor-Phone': '',
            'Receipt-Date': '',
            'Receipt-Time': '',
            'Total-spent': ''
        };

        // Vendor: probably the first meaningful text value
        if (lines.length > 0) {
            // Skip lines that look like addresses or phone numbers for vendor name
            for (const line of lines) {
                if (line.length > 2 &&
                    !this.isPhoneNumber(line) &&
                    !this.isDate(line) &&
                    !line.match(/^\d+/) && // Skip lines starting with numbers
                    !line.match(/^(st|ave|blvd|rd|street|avenue|boulevard|road)/i)) { // Skip address-like lines
                    fields['Vendor'] = line;
                    break;
                }
            }
        }

        // Phone: first value matching phone pattern (check individual words and combinations)
        for (let i = 0; i < allWords.length; i++) {
            // Check single word
            if (this.isPhoneNumber(allWords[i])) {
                fields['Vendor-Phone'] = allWords[i];
                break;
            }
            // Check combinations of 2-4 consecutive words for patterns like "123 456 7890" or "(555) 123-4567"
            for (let j = 2; j <= 4 && i + j <= allWords.length; j++) {
                const words = allWords.slice(i, i + j);

                // Try space-separated combination (most common)
                const spaceCombination = words.join(' ');
                if (this.isPhoneNumber(spaceCombination)) {
                    fields['Vendor-Phone'] = spaceCombination;
                    break;
                }

                // Try dash-separated combination for cases like "123" "456" "7890" → "123-456-7890"
                const dashCombination = words.join('-');
                if (this.isPhoneNumber(dashCombination)) {
                    fields['Vendor-Phone'] = dashCombination;
                    break;
                }

                // Try direct concatenation for cases like "(555)" "1234567" → "(555)1234567"
                const directCombination = words.join('');
                if (this.isPhoneNumber(directCombination)) {
                    fields['Vendor-Phone'] = directCombination;
                    break;
                }
            }
            if (fields['Vendor-Phone']) break;
        }

        // Date: first value matching date pattern (check individual words and combinations)
        for (let i = 0; i < allWords.length; i++) {
            // Check single word
            if (this.isDate(allWords[i])) {
                fields['Receipt-Date'] = allWords[i];
                break;
            }
            // Check combinations of 2-4 consecutive words for patterns like "Aug 25, 2025"
            for (let j = 2; j <= 4 && i + j <= allWords.length; j++) {
                const wordCombination = allWords.slice(i, i + j).join(' ');
                if (this.isDate(wordCombination)) {
                    fields['Receipt-Date'] = wordCombination;
                    break;
                }
            }
            if (fields['Receipt-Date']) break;
        }

        // Time: first value matching time pattern (check individual words and combinations)
        for (let i = 0; i < allWords.length; i++) {
            // Check single word
            if (this.isTime(allWords[i])) {
                fields['Receipt-Time'] = allWords[i];
                break;
            }
            // Check combinations of 2 consecutive words for patterns like "10:30 AM"
            if (i + 1 < allWords.length) {
                const wordCombination = allWords.slice(i, i + 2).join(' ');
                if (this.isTime(wordCombination)) {
                    fields['Receipt-Time'] = wordCombination;
                    break;
                }
            }
            if (fields['Receipt-Time']) break;
        }

        // Total: largest monetary value in the document
        let largestValue = 0;
        let largestValueText = '';
        for (const word of allWords) {
            if (this.isMonetaryValue(word)) {
                const numericValue = parseFloat(word.replace(/[$,]/g, ''));
                if (numericValue > largestValue) {
                    largestValue = numericValue;
                    largestValueText = word;
                }
            }
        }
        if (largestValueText) {
            fields['Total-spent'] = largestValueText;
        }

        // Address: Find first line starting with a number < 10000 followed by text
        // that hasn't been used by another field
        const usedValues = Object.values(fields).filter(v => v.length > 0);
        for (const line of lines) {
            // Skip lines already allocated to other fields
            if (usedValues.some(value => line.includes(value) || value.includes(line))) {
                continue;
            }

            // Check if line starts with a number < 10000 followed by text
            const addressPattern = /^(\d+)\s+(.+)$/;
            const match = line.match(addressPattern);

            if (match) {
                const streetNumber = parseInt(match[1]);
                const streetName = match[2];

                // Street number should be < 10000 and there should be text after it
                if (streetNumber < 10000 && streetName.trim().length > 0) {
                    fields['Vendor-Address'] = line;
                    break;
                }
            }
        }

        this.extractedFields = fields;
        console.log('Heuristic extraction results:', fields);
    }

    isPhoneNumber(text) {
        // Match various phone number formats including international
        const phonePatterns = [
            // US formats
            /^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/, // (123) 456-7890, 123-456-7890, 123.456.7890
            /^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/, // +1 123 456 7890
            /^\d{10}$/, // 1234567890
            /^\d{3}\s+\d{3}\s+\d{4}$/, // 123 456 7890
            /^\(\d{3}\)\s+\d{3}\s+\d{4}$/, // (123) 456 7890
            /^\(\d{3}\)\d{7}$/, // (555)1234567
            /^\(\d{3}\)\d{3}-?\d{4}$/, // (555)123-4567 or (555)1234567

            // International formats
            /^\+\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$/, // +44 20 1234 5678, +33 1 23 45 67 89
            /^\d{3}[-.\s]?\d{4}[-.\s]?\d{6}$/, // 123 4567 123456
            /^\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4,6}$/, // 123 456 7890 or 123 4567 123456
            /^\(\d{2,4}\)[-.\s]?\d{3,4}[-.\s]?\d{4,6}$/, // (44) 20 1234 5678
            /^\+?\d{8,15}$/, // Simple 8-15 digit number with optional +
            /^\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4,8}$/ // General international: 44 20 12345678
        ];
        const normalizedText = text.replace(/\s+/g, ' ').trim();

        // Additional validation: must contain at least 7 digits total
        const digitCount = (normalizedText.match(/\d/g) || []).length;
        if (digitCount < 7 || digitCount > 15) {
            return false;
        }

        return phonePatterns.some(pattern => pattern.test(normalizedText));
    }

    isTime(text) {
        // Match various time formats
        const timePatterns = [
            /^\d{1,2}:\d{2}$/, // H:MM or HH:MM (basic format)
            /^\d{1,2}:\d{2}\s?(am|pm)$/i, // H:MM AM/PM or HH:MM AM/PM
            /^\d{1,2}:\d{2}:\d{2}$/, // H:MM:SS or HH:MM:SS
            /^\d{1,2}:\d{2}:\d{2}\s?(am|pm)$/i // H:MM:SS AM/PM or HH:MM:SS AM/PM
        ];
        const normalizedText = text.trim().toLowerCase();

        // Additional validation: check if it's a reasonable time
        if (timePatterns.some(pattern => pattern.test(normalizedText))) {
            // Extract hours and minutes for validation
            const timeParts = normalizedText.split(':');
            const hours = parseInt(timeParts[0]);
            const minutes = parseInt(timeParts[1]);

            // Validate ranges: hours 0-23 (or 1-12 for AM/PM), minutes 0-59
            const hasAmPm = /\s?(am|pm)$/i.test(normalizedText);
            const validHours = hasAmPm ? (hours >= 1 && hours <= 12) : (hours >= 0 && hours <= 23);
            const validMinutes = minutes >= 0 && minutes <= 59;

            return validHours && validMinutes;
        }

        return false;
    }

    isDate(text) {
        // Match various date formats
        const datePatterns = [
            /^\d{1,2}\/\d{1,2}\/\d{2,4}$/, // MM/DD/YYYY, M/D/YY
            /^\d{1,2}-\d{1,2}-\d{2,4}$/, // MM-DD-YYYY, M-D-YY
            /^\d{1,2}\.\d{1,2}\.\d{2,4}$/, // MM.DD.YYYY
            /^\d{4}\/\d{1,2}\/\d{1,2}$/, // YYYY/MM/DD
            /^\d{4}-\d{1,2}-\d{1,2}$/, // YYYY-MM-DD
            /^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}$/i, // Month DD, YYYY or Month DD YYYY
            /^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}$/i // Full month names
        ];
        const normalizedText = text.trim().toLowerCase();
        return datePatterns.some(pattern => pattern.test(normalizedText));
    }

    isMonetaryValue(text) {
        // Match monetary values that include exactly two decimal places
        const moneyPatterns = [
            /^\$?\d+\.\d{2}$/, // $12.34, 12.34
            /^\$?\d{1,3}(,\d{3})*\.\d{2}$/ // $1,234.56, 1,234.56
        ];
        return moneyPatterns.some(pattern => pattern.test(text.trim())) && parseFloat(text.replace(/[$,]/g, '')) > 0;
    }

    displayResults() {
        // Show annotated image with bounding boxes
        this.createAnnotatedImage();

        // Display extracted fields
        this.displayFields();

        // Display full OCR result
        this.displayOCRResult();

        // Switch to fields tab
        this.switchTab('fields');
    }

    createAnnotatedImage() {
        const canvas = this.annotatedCanvas;
        const ctx = canvas.getContext('2d');
        const img = this.selectedImage;

        // Set canvas size to match image
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // Draw the original image
        ctx.drawImage(img, 0, 0);

        // Draw bounding boxes for words
        if (this.ocrData && this.ocrData.words) {
            ctx.strokeStyle = '#4A90E2';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(74, 144, 226, 0.1)';

            this.ocrData.words.forEach(word => {
                if (word.text && word.text.trim().length > 0) {
                    const { x0, y0, x1, y1 } = word.bbox;
                    const margin = 3; // Small margin in pixels
                    const width = (x1 - x0) + (margin * 2);
                    const height = (y1 - y0) + (margin * 2);
                    const x = x0 - margin;
                    const y = y0 - margin;

                    ctx.strokeRect(x, y, width, height);
                    ctx.fillRect(x, y, width, height);
                }
            });
        }

        // Switch to annotated view
        this.selectedImage.style.display = 'none';
        this.annotatedCanvas.style.display = 'block';

        // Apply current zoom level to canvas
        this.updateZoom();
    }

    displayFields() {
        const fieldsContainer = this.fieldsList;
        fieldsContainer.innerHTML = '';

        console.log('displayFields called. extractedFields:', this.extractedFields, 'isModelLoaded:', this.isModelLoaded);

        if (!this.extractedFields) {
            // Show message that AI extraction is not available only if model wasn't loaded
            if (!this.isModelLoaded) {
                console.log('Showing AI model not available message');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'field-message';
                messageDiv.innerHTML = `
                    <p><strong>AI Model Not Available</strong></p>
                    <p>Field extraction requires the AI model, which could not be loaded. The extracted text is still available in the Result tab.</p>
                `;
                fieldsContainer.appendChild(messageDiv);
                this.fieldsList.style.display = 'flex';
                this.fieldsTab.querySelector('.results-placeholder').style.display = 'none';
            }
            // If model IS loaded but no fields extracted, just return (keep placeholder visible)
            return;
        }

        // Show failover error message if it exists
        if (this.failoverErrorMessage) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'field-message';
            errorDiv.style.backgroundColor = '#fff3cd';
            errorDiv.style.borderColor = '#ffc107';
            errorDiv.style.color = '#856404';
            errorDiv.style.marginBottom = '1rem';
            errorDiv.innerHTML = this.failoverErrorMessage;
            fieldsContainer.appendChild(errorDiv);
            // Clear the message after displaying
            this.failoverErrorMessage = null;
        }

        console.log('Displaying extracted fields:', Object.keys(this.extractedFields).length, 'fields');
        Object.entries(this.extractedFields).forEach(([fieldName, fieldValue]) => {
            const fieldItem = document.createElement('div');
            fieldItem.className = 'field-item';
            fieldItem.setAttribute('role', 'listitem');

            const label = document.createElement('div');
            label.className = 'field-label';
            label.textContent = fieldName.replace('-', ' ');
            label.id = `label-${fieldName}`;

            const value = document.createElement('div');
            value.className = `field-value ${fieldValue ? '' : 'empty'}`;
            value.textContent = fieldValue || 'Not found';
            value.setAttribute('aria-labelledby', `label-${fieldName}`);
            value.setAttribute('aria-description', fieldValue ? `${fieldName.replace('-', ' ')}: ${fieldValue}` : `${fieldName.replace('-', ' ')}: Not found in receipt`);

            fieldItem.appendChild(label);
            fieldItem.appendChild(value);
            fieldsContainer.appendChild(fieldItem);
        });

        // Show fields list and hide placeholder
        this.fieldsList.style.display = 'flex';
        this.fieldsTab.querySelector('.results-placeholder').style.display = 'none';

        // Announce completion to screen readers
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'sr-only';
        announcement.textContent = `Analysis complete. ${Object.keys(this.extractedFields).length} fields extracted from receipt.`;
        document.body.appendChild(announcement);
        setTimeout(() => document.body.removeChild(announcement), 1000);
    }

    displayOCRResult() {
        if (!this.ocrData) return;

        // Just display the extracted text, not the full JSON
        this.ocrResult.textContent = this.ocrData.text || 'No text extracted';

        // Show result content and hide placeholder
        this.resultContent.style.display = 'block';
        this.resultTab.querySelector('.results-placeholder').style.display = 'none';
    }

    switchTab(tabName) {
        // Update tab buttons
        this.tabBtns.forEach(btn => {
            btn.classList.remove('active');
            btn.setAttribute('aria-selected', 'false');
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
            }
        });

        // Update tab content
        this.tabContents.forEach(content => {
            content.classList.remove('active');
        });

        if (tabName === 'fields') {
            this.fieldsTab.classList.add('active');
        } else if (tabName === 'result') {
            this.resultTab.classList.add('active');
        }

        // Announce tab change to screen readers
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) {
            activeTab.focus();
        }
    }

    resetResults() {
        // Reset extracted data
        this.ocrData = null;
        this.extractedFields = null;

        // Hide results and show placeholders
        this.fieldsList.style.display = 'none';
        this.fieldsTab.querySelector('.results-placeholder').style.display = 'block';
        this.resultContent.style.display = 'none';
        this.resultTab.querySelector('.results-placeholder').style.display = 'block';

        // Reset to original image view
        this.selectedImage.style.display = 'block';
        this.annotatedCanvas.style.display = 'none';

        // Reset zoom
        this.updateZoom();

        // Switch back to fields tab
        this.switchTab('fields');
    }

    showProgress() {
        this.progressSection.style.display = 'block';
        this.progressFill.style.width = '0%';
        this.progressText.textContent = 'Starting analysis...';
        this.analyzeBtn.disabled = true;
    }

    updateProgress(percentage, text) {
        this.progressFill.style.width = `${percentage}%`;
        this.progressText.textContent = text;

        // Update ARIA attributes
        const progressBar = this.progressSection.querySelector('.progress-bar');
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.setAttribute('aria-valuetext', `${Math.round(percentage)}% - ${text}`);
    }

    hideProgress() {
        this.progressSection.style.display = 'none';
        this.analyzeBtn.disabled = false;
    }

    disableUploadAndSelection() {
        // Disable file upload area
        this.uploadArea.classList.add('disabled');
        this.uploadArea.setAttribute('tabindex', '-1');
        this.uploadArea.setAttribute('aria-disabled', 'true');
        this.imageInput.disabled = true;

        // Disable all thumbnail selection
        const thumbnails = document.querySelectorAll('.thumbnail-item');
        thumbnails.forEach(thumbnail => {
            thumbnail.classList.add('disabled');
            thumbnail.setAttribute('tabindex', '-1');
            thumbnail.setAttribute('aria-disabled', 'true');
        });

        // Disable remove buttons
        const removeButtons = document.querySelectorAll('.thumbnail-remove');
        removeButtons.forEach(button => {
            button.disabled = true;
            button.setAttribute('aria-disabled', 'true');
        });
    }

    enableUploadAndSelection() {
        // Enable file upload area
        this.uploadArea.classList.remove('disabled');
        this.uploadArea.setAttribute('tabindex', '0');
        this.uploadArea.setAttribute('aria-disabled', 'false');
        this.imageInput.disabled = false;

        // Enable all thumbnail selection
        const thumbnails = document.querySelectorAll('.thumbnail-item');
        thumbnails.forEach(thumbnail => {
            thumbnail.classList.remove('disabled');
            thumbnail.setAttribute('tabindex', '0');
            thumbnail.setAttribute('aria-disabled', 'false');
        });

        // Enable remove buttons
        const removeButtons = document.querySelectorAll('.thumbnail-remove');
        removeButtons.forEach(button => {
            button.disabled = false;
            button.setAttribute('aria-disabled', 'false');
        });
    }

    showError(message, title = 'Error') {
        // Update the heading
        const errorHeading = this.errorSection.querySelector('h2');
        errorHeading.textContent = title;

        // Handle newlines in the message - use textContent and convert \n to <br> elements safely
        this.errorMessage.textContent = '';
        const lines = message.split('\n');
        lines.forEach((line, index) => {
            this.errorMessage.appendChild(document.createTextNode(line));
            if (index < lines.length - 1) {
                this.errorMessage.appendChild(document.createElement('br'));
            }
        });

        // Force styling for fallback mode — reserved for future use
        this.errorSection.style.display = 'block';
        this.hideProgress();
    }

    hideError() {
        this.errorSection.style.display = 'none';
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new InfoExtractorApp();
});

// Release blob URLs and shut down the wllama worker on unload to avoid leaks
window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.uploadedImages.forEach(img => URL.revokeObjectURL(img.url));
        if (window.app.wllama) {
            window.app.wllama.exit().catch(() => { });
        }
    }
});

// Handle any uncaught errors
window.addEventListener('error', (event) => {
    console.error('Uncaught error:', event.error);
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});