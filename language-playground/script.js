import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";
import { Wllama } from '@wllama/wllama';

(function () {
    "use strict";

    const MAX_FILE_SIZE = 10 * 1024;

    // AI Model state
    let engine = null; // WebLLM engine for GPU mode
    let wllama = null; // Wllama instance for CPU mode
    let webGPUAvailable = false; // Track if WebGPU is available
    let usingWllama = false; // Track which engine is active
    let aiModelReady = false; // Track if AI model is loaded

    const LANGUAGES = [
        { name: "English", code: "en", script: "latin", words: ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"] },
        { name: "French", code: "fr", script: "latin", words: ["le", "de", "un", "etre", "et", "a", "il", "avoir", "ne", "je", "son", "que", "se", "qui", "ce", "dans", "en", "du", "elle", "au", "pour", "pas", "plus", "par", "avec", "tout", "faire", "mettre", "autre", "on", "mais", "nous", "comme", "ou", "si", "leur", "dire", "devoir", "avant", "deux", "meme", "prendre", "aussi", "celui", "donner", "bien", "fois", "vous", "encore", "nouveau", "aller", "entre", "premier", "vouloir", "deja", "grand", "mon", "me", "moins", "aucun", "lui", "temps", "tres", "savoir", "falloir", "voir", "quelque", "sans", "raison", "notre", "dont", "non", "an", "monde", "jour", "demander", "alors", "apres", "trouver", "personne", "rendre", "part", "dernier", "venir", "pendant", "passer", "peu", "suite"] },
        { name: "Spanish", code: "es", script: "latin", words: ["de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", "o", "pero", "sus", "le", "ya", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "esto", "mi", "antes", "algunos", "unos", "yo", "otro", "otras", "otra", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mis", "tu", "te", "ti", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os"] },
        { name: "Portuguese", code: "pt", script: "latin", words: ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "nao", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "seu", "sua", "ou", "ser", "quando", "muito", "ha", "nos", "ja", "esta", "eu", "tambem", "so", "pelo", "pela", "ate", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estao", "voce", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "minha", "numa", "pelos", "elas", "havia", "seja", "qual", "sera", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "voces", "vos", "lhes"] },
        { name: "German", code: "de", script: "latin", words: ["der", "die", "und", "in", "den", "von", "zu", "mit", "sich", "des", "auf", "fur", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach", "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "uber", "einen", "so", "zum", "war", "haben", "nur", "oder", "aber", "vor", "zur", "bis", "mehr", "durch", "man", "sein", "wurde", "sei", "hatte", "kann", "gegen", "vom", "konnte", "schon", "wenn", "habe", "seine", "ihre", "dann", "unter", "wir", "soll", "ich", "eines", "jahr", "zwei", "diese", "dieser", "wieder", "keine", "seiner", "worden", "will", "zwischen", "immer", "was", "sagte", "gibt", "alle", "diesen", "seit", "muss", "doch", "jetzt"] },
        { name: "Italian", code: "it", script: "latin", words: ["di", "e", "il", "la", "che", "a", "per", "un", "in", "del", "le", "si", "una", "dei", "da", "non", "con", "su", "piu", "al", "come", "ma", "ha", "sono", "o", "anche", "nel", "mi", "io", "se", "gli", "ci", "questa", "questo", "quando", "dopo", "tra", "dove", "chi", "tutto", "tutti", "essere", "fare", "stato", "molto", "poi", "della", "delle", "degli", "alla", "dai", "dalle", "suo", "sua", "loro", "noi", "voi", "lui", "lei", "quello", "quella", "prima", "ancora", "bene", "male", "solo", "sempre", "oggi", "ieri", "domani", "anno", "anni", "giorno", "giorni", "tempo", "casa", "uomo", "donna", "ragazzo", "ragazza", "persona", "persone", "lavoro", "vita", "mano", "occhi", "cuore", "mondo", "parte", "storia", "paese", "stessa", "stesso", "uno", "due", "tre", "quattro", "cinque", "niente", "qualcosa", "cosi"] },
        { name: "Simplified Chinese", code: "zh", script: "han", words: ["的", "一", "是", "在", "不", "了", "有", "和", "人", "这", "中", "大", "为", "上", "个", "国", "我", "以", "要", "他", "时", "来", "用", "们", "生", "到", "作", "地", "于", "出", "就", "分", "对", "成", "会", "可", "主", "发", "年", "动", "同", "工", "也", "能", "下", "过", "子", "说", "产", "种", "面", "而", "方", "后", "多", "定", "行", "学", "法", "所", "民", "得", "经", "十", "三", "之", "进", "着", "等", "部", "度", "家", "电", "力", "里", "如", "水", "化", "高", "自", "二", "理", "起", "小", "物", "现", "实", "加", "量", "都", "两", "体", "制", "机", "当", "使", "点", "从", "业", "本", "去", "把", "性", "好"] },
        { name: "Japanese", code: "ja", script: "japanese", words: ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その", "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ", "なる", "へ", "か", "だ", "これ", "によって", "により", "おり", "より", "による", "ず", "なり", "られる", "において", "ば", "なかっ", "なく", "しかし", "について", "せ", "だっ", "その後", "できる", "それ", "う", "ので", "なお", "のみ", "でき", "き", "つ", "における", "および", "いう", "さらに", "でも", "ら", "たり", "その他", "に関する", "たち", "ます", "ん", "なら", "に対して", "特に", "せる", "及び", "これら", "とき", "では", "にて", "ほか", "ながら", "うち", "そして", "ただし"] },
        { name: "Hindi", code: "hi", script: "devanagari", words: ["और", "का", "है", "में", "की", "से", "को", "पर", "यह", "था", "हैं", "एक", "नहीं", "हो", "गया", "कर", "ने", "कि", "तो", "ही", "या", "थे", "लेकिन", "उस", "उन", "लिए", "यदि", "जब", "तक", "अपने", "कुछ", "बहुत", "बाद", "फिर", "जहाँ", "भी", "किसी", "साथ", "पहले", "किया", "दिया", "रहा", "रही", "रहे", "हम", "आप", "मैं", "तुम", "वह", "वे", "इस", "उसके", "उनके", "अपना", "सब", "सभी", "अभी", "क्यों", "कैसे", "क्या", "कौन", "कब", "कहाँ", "बिना", "मगर", "अगर", "ऐसा", "वैसा", "जाना", "आना", "रखना", "देना", "लेना", "बोलना", "सोचना", "समय", "दिन", "रात", "साल", "घर", "पानी", "काम", "जीवन", "दुनिया", "बड़ा", "छोटा", "अच्छा", "खराब", "नया", "पुराना", "ज्यादा", "कम", "तीन", "चार", "पांच", "लोग", "देश", "शहर"] },
        { name: "Arabic", code: "ar", script: "arabic", words: ["في", "من", "إلى", "على", "أن", "هذا", "هذه", "هو", "هي", "كان", "كانت", "يكون", "التي", "الذي", "ما", "لا", "مع", "عن", "ذلك", "هناك", "قد", "تم", "كما", "بعد", "قبل", "عند", "كل", "أي", "بين", "حتى", "أو", "ثم", "إن", "إذا", "لكن", "لأن", "منذ", "حول", "أمام", "خلف", "اليوم", "غدا", "أمس", "الآن", "دائما", "أبدا", "هنا", "الناس", "رجل", "امرأة", "طفل", "بيت", "مدينة", "دولة", "عمل", "حياة", "وقت", "يوم", "سنة", "جيد", "سيئ", "كبير", "صغير", "أكثر", "أقل", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "قال", "يقول", "فعل", "يفعل", "ذهب", "عاد", "أخذ", "أعطى", "يريد", "يمكن", "يجب", "نحن", "أنا", "أنت", "هم", "هن", "له", "لها", "لهم", "بها", "فيها", "عليه", "إليها", "إليهم", "اللغة", "الكتاب", "الطريق"] },
        { name: "Russian", code: "ru", script: "cyrillic", words: ["и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой"] }
    ];

    const SCRIPTS = {
        han: /[\u4E00-\u9FFF]/g,
        japanese: /[\u3040-\u30FF]/g,
        devanagari: /[\u0900-\u097F]/g,
        arabic: /[\u0600-\u06FF]/g,
        cyrillic: /[\u0400-\u04FF]/g,
        latin: /[A-Za-z]/g
    };

    const SAMPLE_TEXTS = {
        language: [
            { label: "English paragraph", text: "Hello, my name is John and I live in Seattle. I work for a software company and enjoy reading books in the evening." },
            { label: "French paragraph", text: "Bonjour, je m'appelle Sophie et j'habite a Lyon. Je travaille dans une petite entreprise de technologie et j'aime lire des livres le soir." },
            { label: "Japanese paragraph", text: "私は東京に住んでいます。毎朝電車で会社に行き、昼休みに友達と一緒にご飯を食べます。週末は公園で散歩します。" },
            { label: "Arabic paragraph", text: "انا اسكن في دبي واعمل في شركة برمجيات. في عطلة نهاية الاسبوع ازور عائلتي واقرا كتابا جديدا." }
        ],
        pii: [
            { label: "Contact note", text: "John Smith can be reached by email at john@contoso.com, or by phone on +1 555 123 4567. Mailing address: 123 Anystreet, Anytown, WA, USA, 01234." },
            { label: "Customer form", text: "Customer Feedback Form\nDate: 4/1/2026\nCustomer: Mario Gizzi\nEmail: mario@adventure-works.com\nPhone: 555 123 0987\nRating: 5\nComment: Thanks for the great service. I received my delivery at 1482 Westward Way, Seattle. Everything looks great!" }
        ]
    };

    const LANGUAGE_MARKERS = {
        en: ["hello", "this", "with", "from", "would", "people", "work", "time"],
        fr: ["bonjour", "avec", "dans", "une", "habite", "travaille", "livre", "soir"],
        es: ["hola", "esta", "gracias", "trabaja", "vive", "ciudad", "buenos", "dias"],
        pt: ["ola", "voce", "trabalha", "cidade", "obrigado", "tenho", "noite", "livro"],
        de: ["hallo", "nicht", "ich", "eine", "arbeit", "stadt", "guten", "tag"],
        it: ["ciao", "grazie", "lavora", "vive", "sera", "libro", "citta", "buona"],
        zh: ["我们", "他们", "这个", "那个", "喜欢", "工作", "时间", "今天"],
        ja: ["です", "ます", "私", "今日", "仕事", "好き", "東京", "友達"],
        hi: ["नमस्ते", "मैं", "आप", "काम", "घर", "समय", "लोग", "शहर"],
        ar: ["مرحبا", "انا", "نحن", "العمل", "المدينة", "اليوم", "الكتاب", "الوقت"],
        ru: ["привет", "это", "город", "работа", "люди", "книга", "сегодня", "время"]
    };

    const NORMALIZED_LANGUAGES = LANGUAGES.map(function (language) {
        return {
            name: language.name,
            code: language.code,
            script: language.script,
            words: language.words.map(function (word) {
                return normalizeForLanguage(word, language.script);
            })
        };
    });

    const state = {
        mode: "language",
        locked: false,
        originalText: ""
    };

    const elements = {
        analyzerSelect: document.getElementById("analyzer-select"),
        sourceText: document.getElementById("source-text"),
        sampleSelect: document.getElementById("sample-select"),
        attachBtn: document.getElementById("attach-btn"),
        detectBtn: document.getElementById("detect-btn"),
        fileInput: document.getElementById("file-input"),
        fileStatus: document.getElementById("file-status"),
        placeholder: document.getElementById("placeholder"),
        placeholderText: document.getElementById("placeholder-text"),
        placeholderSubtext: document.getElementById("placeholder-subtext"),
        results: document.getElementById("results"),
        announcer: document.getElementById("aria-announcer"),
        themeToggle: document.getElementById("theme-toggle"),
        modelStatus: document.getElementById("model-status"),
        modelToggle: document.getElementById("model-toggle")
    };

    function announce(text) {
        elements.announcer.textContent = "";
        window.setTimeout(function () {
            elements.announcer.textContent = text;
        }, 10);
    }

    function showPlaceholder() {
        elements.placeholder.hidden = false;
        elements.placeholder.style.display = "grid";
        elements.results.hidden = true;
        elements.results.style.display = "none";
    }

    function showResults() {
        elements.placeholder.hidden = true;
        elements.placeholder.style.display = "none";
        elements.results.hidden = false;
        elements.results.style.display = "flex";
    }

    function showLoading() {
        elements.placeholder.hidden = true;
        elements.placeholder.style.display = "none";
        elements.results.hidden = false;
        elements.results.style.display = "flex";
        elements.results.innerHTML = "<div class='loading-state'><div class='loading-dots'><span></span><span></span><span></span></div></div>";
    }

    function clearResults() {
        elements.results.innerHTML = "";
    }

    function updatePlaceholder() {
        elements.placeholderText.textContent = state.mode === "language"
            ? "Add text or a file to detect language"
            : "Add text or a file to detect PII";
        elements.placeholderSubtext.textContent = "Type in text, select a sample, or upload a text file to get started.";
        showPlaceholder();
    }

    function showStatus(message, isError) {
        elements.fileStatus.textContent = message;
        elements.fileStatus.classList.toggle("error-text", Boolean(isError));
        announce(message);
    }

    function escapeRegExp(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    /**
     * Updates the loading status for the AI model
     */
    function updateModelStatus(message, isLoading) {
        if (!elements.modelStatus) return;

        elements.modelStatus.textContent = message;
        elements.modelStatus.className = 'model-status-text ' + (isLoading ? 'loading' : 'ready');
    }

    /**
     * Disables PII-related UI elements during model loading
     * but keeps Language Detection features enabled
     */
    function disableUI() {
        // Keep the analyzer dropdown enabled, but disable the PII option
        const piiOption = elements.analyzerSelect.querySelector('option[value="pii"]');
        if (piiOption) piiOption.disabled = true;

        // Keep text input, sample select, attach button enabled for language detection
        // elements.sourceText, elements.sampleSelect, elements.attachBtn remain enabled

        // Disable GPU/CPU toggle until model loads
        if (elements.modelToggle) elements.modelToggle.disabled = true;

        // Detect button state is managed by updateDetectButton()
        updateDetectButton();
    }

    /**
     * Enables PII-related UI elements after model is loaded
     */
    function enableUI() {
        // Enable the PII option in the analyzer dropdown
        const piiOption = elements.analyzerSelect.querySelector('option[value="pii"]');
        if (piiOption) piiOption.disabled = false;

        // Enable GPU/CPU toggle now that model is loaded
        if (elements.modelToggle) elements.modelToggle.disabled = false;

        // Other elements (text input, etc.) were never disabled
        updateDetectButton();
    }

    /**
     * Checks if WebGPU is available in the browser
     */
    function checkWebGPUSupport() {
        if (!navigator.gpu) {
            console.log('WebGPU not supported in this browser');
            return false;
        }
        return true;
    }

    /**
     * Initializes WebLLM for GPU mode
     */
    async function initializeWebLLM() {
        try {
            updateModelStatus('Loading Phi-3-mini (GPU)...', true);

            const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';

            engine = await webllm.CreateMLCEngine(
                targetModelId,
                {
                    initProgressCallback: function (progress) {
                        console.log('WebLLM progress object:', progress);

                        // Use the text property if available for more detailed status
                        if (progress.text) {
                            updateModelStatus(progress.text, true);
                        } else {
                            const percentage = Math.round(progress.progress * 100);
                            updateModelStatus('Loading Phi-3-mini: ' + percentage + '%', true);
                        }
                    }
                }
            );

            updateModelStatus('Phi-3-mini (GPU) Ready', false);
            console.log('WebLLM engine initialized successfully');
            webGPUAvailable = true;
            usingWllama = false;
            aiModelReady = true;

        } catch (error) {
            console.error('Failed to initialize WebLLM:', error);
            throw error; // Re-throw to trigger fallback
        }
    }

    /**
     * Initializes the Wllama language model for CPU mode text generation
     */
    async function initWllama() {
        try {
            if (wllama) {
                console.log('Wllama already initialized');
                return;
            }

            console.log("Initializing wllama...");
            updateModelStatus('Loading SmolLM2 (CPU)...', true);

            // Configure WASM paths for CDN
            const CONFIG_PATHS = {
                'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/single-thread/wllama.wasm',
                'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.7/esm/multi-thread/wllama.wasm',
            };

            const internalProgressCallback = ({ loaded, total }) => {
                const progress = loaded / total;
                const percentage = Math.round((progress * 100));
                const statusText = 'Loading SmolLM2: ' + percentage + '%';
                console.log('Wllama progress:', percentage + '%', 'loaded:', loaded, 'total:', total);
                updateModelStatus(statusText, true);
                // Force DOM update
                if (elements.modelStatus) {
                    elements.modelStatus.offsetHeight;
                }
            };

            // Try multithreaded (4 threads) first, fall back to single-threaded
            const useMultiThread = window.crossOriginIsolated === true;
            const preferredThreads = useMultiThread ? 4 : 1;
            console.log('Cross-origin isolated: ' + window.crossOriginIsolated + ', attempting ' + preferredThreads + ' thread(s)');

            try {
                wllama = new Wllama(CONFIG_PATHS);

                await wllama.loadModelFromHF(
                    'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                    'smollm2-360m-instruct-q8_0.gguf',
                    {
                        n_ctx: 768,
                        n_threads: preferredThreads,
                        progressCallback: internalProgressCallback
                    }
                );
                console.log('Wllama initialized successfully with ' + preferredThreads + ' thread(s)');
            } catch (multiErr) {
                if (preferredThreads > 1) {
                    console.warn('Multi-threaded init failed (' + multiErr.message + '), falling back to single thread');

                    wllama = new Wllama(CONFIG_PATHS);
                    await wllama.loadModelFromHF(
                        'ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF',
                        'smollm2-360m-instruct-q8_0.gguf',
                        {
                            n_ctx: 768,
                            n_threads: 1,
                            progressCallback: internalProgressCallback
                        }
                    );
                    console.log("Wllama initialized successfully with 1 thread (fallback)");
                } else {
                    throw multiErr;
                }
            }

            updateModelStatus('SmolLM2 (CPU) Ready', false);
            usingWllama = true;
            aiModelReady = true;
        } catch (error) {
            console.error('Failed to initialize wllama:', error);
            updateModelStatus('Model loading failed', false);
            throw error;
        }
    }

    /**
     * Initializes AI models on page load
     */
    async function initAIModels() {
        disableUI();

        try {
            // Check for WebGPU support
            const hasWebGPU = checkWebGPUSupport();

            if (!hasWebGPU) {
                console.log('WebGPU not available, using wllama (CPU mode)');
                if (elements.modelToggle) {
                    elements.modelToggle.checked = false;
                }
                await initWllama();
            } else {
                // Try WebLLM first (faster with GPU)
                try {
                    if (elements.modelToggle) {
                        elements.modelToggle.checked = true;
                    }
                    await initializeWebLLM();
                } catch (error) {
                    console.log('WebLLM initialization failed, falling back to wllama');
                    if (elements.modelToggle) {
                        elements.modelToggle.checked = false;
                    }
                    await initWllama();
                }
            }

            enableUI();
        } catch (error) {
            console.error("AI model initialization error:", error);
            updateModelStatus("Model loading failed", false);
            showStatus("Error loading AI model: " + error.message + ". PII detection may be unavailable.", true);
            enableUI();
        }
    }

    /**
     * Switches between GPU and CPU models
     */
    async function switchModel(useGPU) {
        // Clear UI and reset state when switching models
        resetUi();

        // Switch to Language Detection mode and disable PII until new model loads
        state.mode = "language";
        elements.analyzerSelect.value = "language";
        populateSamples();
        updatePlaceholder();

        disableUI();
        aiModelReady = false;

        try {
            if (useGPU) {
                // Load GPU model
                if (engine) {
                    console.log('WebLLM already loaded');
                    updateModelStatus('Phi-3-mini (GPU) Ready', false);
                    usingWllama = false;
                    aiModelReady = true;
                } else {
                    await initializeWebLLM();
                }
            } else {
                // Load CPU model
                if (wllama) {
                    console.log('Wllama already loaded');
                    updateModelStatus('SmolLM2 (CPU) Ready', false);
                    usingWllama = true;
                    aiModelReady = true;
                } else {
                    await initWllama();
                }
                usingWllama = true;
            }
            enableUI();
        } catch (error) {
            console.error('Model switch failed:', error);
            updateModelStatus('Model loading failed', false);
            showStatus('Failed to load model: ' + error.message, true);
            enableUI();
        }
    }

    function stripDiacritics(text) {
        return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    }

    function normalizeForLanguage(text, script) {
        const lowered = String(text || "").toLowerCase();
        if (script === "latin") {
            return stripDiacritics(lowered);
        }
        return lowered;
    }

    function tokenize(text) {
        return stripDiacritics(text)
            .toLowerCase()
            .replace(/[0-9.,!?;:()\[\]{}"'“”‘’<>/\\_-]+/g, " ")
            .split(/\s+/)
            .map(function (part) { return part.trim(); })
            .filter(Boolean);
    }

    function countSubstringMatches(text, words) {
        const matchedTerms = new Set();
        let hits = 0;

        words.forEach(function (word) {
            if (word && text.includes(word)) {
                matchedTerms.add(word);
                hits += 1;
            }
        });

        return {
            hits: hits,
            uniqueHits: matchedTerms.size
        };
    }

    function getLanguageMatchStats(language, tokens, sourceText) {
        if (language.script === "japanese" || language.script === "han") {
            return countSubstringMatches(sourceText, language.words);
        }

        const wordSet = new Set(language.words);
        const matchedTerms = new Set();
        let hits = 0;

        tokens.forEach(function (token) {
            if (wordSet.has(token)) {
                hits += 1;
                matchedTerms.add(token);
            }
        });

        return {
            hits: hits,
            uniqueHits: matchedTerms.size
        };
    }

    function getScriptSignals(text) {
        const charCount = Math.max(text.length, 1);
        return {
            han: ((text.match(SCRIPTS.han) || []).length * 1.2) / charCount,
            japanese: ((text.match(SCRIPTS.japanese) || []).length * 1.2) / charCount,
            devanagari: ((text.match(SCRIPTS.devanagari) || []).length * 1.2) / charCount,
            arabic: ((text.match(SCRIPTS.arabic) || []).length * 1.2) / charCount,
            cyrillic: ((text.match(SCRIPTS.cyrillic) || []).length * 1.2) / charCount,
            latin: ((text.match(SCRIPTS.latin) || []).length * 1.0) / charCount
        };
    }

    function detectLanguage(text) {
        const tokens = tokenize(text);
        const scriptSignals = getScriptSignals(text);
        const normalizedLatinText = normalizeForLanguage(text, "latin");
        const scores = NORMALIZED_LANGUAGES.map(function (language) {
            const markerSet = new Set((LANGUAGE_MARKERS[language.code] || []).map(function (marker) {
                return normalizeForLanguage(marker, language.script);
            }));
            const matchedMarkers = new Set();
            const sourceText = language.script === "latin" ? normalizedLatinText : normalizeForLanguage(text, language.script);
            const matchStats = getLanguageMatchStats(language, tokens, sourceText);

            if (language.script === "japanese" || language.script === "han") {
                markerSet.forEach(function (marker) {
                    if (marker && sourceText.includes(marker)) {
                        matchedMarkers.add(marker);
                    }
                });
            } else {
                tokens.forEach(function (token) {
                    if (markerSet.has(token)) {
                        matchedMarkers.add(token);
                    }
                });
            }

            const scoreBase = language.script === "japanese" || language.script === "han"
                ? Math.max(language.words.length * 0.18, 1)
                : Math.max(tokens.length, 1);

            const wordScore = matchStats.hits / scoreBase;
            const uniqueScore = language.words.length ? matchStats.uniqueHits / language.words.length : 0;
            let scriptScore = scriptSignals[language.script] || 0;
            const markerScore = markerSet.size ? matchedMarkers.size / markerSet.size : 0;

            if (language.code === "ja") {
                scriptScore += scriptSignals.japanese * 2.6;
                scriptScore += scriptSignals.han * 0.35;
            }

            if (language.code === "zh") {
                scriptScore += scriptSignals.han * 1.5;
                if (scriptSignals.japanese > 0.01) {
                    scriptScore *= 0.30;
                }
            }

            const rawScore = (wordScore * 0.50) + (uniqueScore * 0.16) + (markerScore * 0.12) + (scriptScore * 0.22);

            return {
                name: language.name,
                code: language.code,
                rawScore: rawScore,
                hits: matchStats.hits,
                uniqueHits: matchStats.uniqueHits,
                markerHits: matchedMarkers.size
            };
        }).sort(function (left, right) {
            return right.rawScore - left.rawScore;
        });

        const best = scores[0];
        const second = scores[1] || { rawScore: 0 };
        const totalScore = scores.reduce(function (sum, item) {
            return sum + item.rawScore;
        }, 0) || 1;
        const normalizedScores = scores.map(function (item) {
            return {
                name: item.name,
                code: item.code,
                rawScore: item.rawScore,
                probability: (item.rawScore / totalScore) * 100,
                hits: item.hits,
                uniqueHits: item.uniqueHits,
                markerHits: item.markerHits
            };
        });

        const bestNormalized = normalizedScores[0];
        const secondNormalized = normalizedScores[1] || { probability: 0 };
        const margin = Math.max(0, bestNormalized.probability - secondNormalized.probability);
        const confidence = Math.max(1, Math.min(99, Math.round(bestNormalized.probability)));

        return {
            language: bestNormalized.name,
            code: bestNormalized.code,
            confidence: confidence,
            probabilities: normalizedScores.slice(0, 4),
            alternatives: normalizedScores.slice(1, 4)
        };
    }

    function addEntity(list, value, type) {
        const trimmed = String(value || "").trim();
        if (!trimmed) {
            return;
        }
        const exists = list.some(function (item) {
            return item.value === trimmed && item.type === type;
        });
        if (!exists) {
            list.push({ value: trimmed, type: type });
        }
    }

    async function detectPii(text) {
        if (!aiModelReady) {
            throw new Error("AI model is not ready yet. Please wait for model initialization.");
        }

        const entities = [];
        let aiResponse = "";

        try {
            // Generate prompt based on the model being used
            let prompt;
            if (usingWllama) {
                // SmolLM2 prompt using ChatML format - kept concise for context limits
                prompt = '<|im_start|>system\n';
                prompt += 'Extract people\'s names, phone numbers, emails, and addresses from text. Find ALL instances. Use format: - TYPE: value where TYPE is PERSON, PHONE, EMAIL, or ADDRESS. Never invent values.\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>user\n';
                prompt += 'List all PERSON, PHONE, EMAIL, and ADDRESS values you find in the following text:\n---\n';
                prompt += 'Contact Sarah Johnson at sarah@company.com or 555-123-4567. Office: 123 Main St, Boston.\n';
                prompt += '---\nIMPORTANT: List all of the PERSON, PHONE, EMAIL, and ADDRESS values you find; but DO NOT create new values that are not in the text.\n\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>assistant\n';
                prompt += '- PERSON: Sarah Johnson\n';
                prompt += '- EMAIL: sarah@company.com\n';
                prompt += '- PHONE: 555 123 4567\n';
                prompt += '- ADDRESS: 123 Main St, Boston\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>user\n';
                prompt += 'List all PERSON, PHONE, EMAIL, and ADDRESS values you find in the following text:\n---\n';
                prompt += 'Customer: Lisa Chen called about a delivery to 12 Tree Ave, Toytown, WA. Call her back on 555 234 5678, or her cellphone (206 555-9999); or email her at lisa.chen@contoso.com.\n';
                prompt += '---\nIMPORTANT: List all of the PERSON, PHONE, EMAIL, and ADDRESS values you find; but DO NOT create new values that are not in the text.\n\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>assistant\n';
                prompt += '- PERSON: Lisa Chen\n';
                prompt += '- ADDRESS: 12 Tree Ave, Toytown, WA\n';
                prompt += '- PHONE: 555 234 5678\n';
                prompt += '- PHONE: 206 555 9999\n';
                prompt += '- EMAIL: lisa.chen@contoso.com\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>user\n';
                prompt += 'List all PERSON, PHONE, EMAIL, and ADDRESS values you find in the following text:\n---\n';
                prompt += text + '\n';
                prompt += '---\nIMPORTANT: List all of the PERSON, PHONE, EMAIL, and ADDRESS values you find; but DO NOT create new values that are not in the text.\n\n';
                prompt += '<|im_end|>\n\n';

                prompt += '<|im_start|>assistant\n';
            } else {
                // Phi 3-mini prompt
                prompt = "Extract people's names, telephone numbers, email addresses, and street addresses from the following text.\n\n";
                prompt += "IMPORTANT RULES:\n";
                prompt += "- Extract ONLY values that are ACTUALLY PRESENT in the text\n";
                prompt += "- Find ALL instances of each type (there may be multiple names, multiple phone numbers, etc.)\n";
                prompt += "- PII can appear anywhere - in sentences, forms, labels, or free-form text\n";
                prompt += "- ALL telephone numbers (phone, cellphone, mobile, etc.) must be classified as PHONE\n";
                prompt += "- Phone numbers can be in various formats: 555-123-4567, 555 123 0987, (555) 123-4567, etc.\n";
                prompt += "- NEVER invent, guess, or make up values\n";
                prompt += "- If a type is not found, do not list it at all\n";
                prompt += "- You MUST use ONLY these 4 types - no other types are allowed\n\n";
                prompt += "Return a bulleted list with each item in this format:\n";
                prompt += "- TYPE: value\n\n";
                prompt += "TYPE must be EXACTLY one of: PERSON, PHONE, EMAIL, ADDRESS\n";
                prompt += "Do NOT use any other type names like CELLPHONE, MOBILE, etc.\n\n";
                prompt += "Examples:\n\n";
                prompt += "Input: Contact Sarah Johnson at sarah@email.com or call 555-123-4567.\n";
                prompt += "Output:\n";
                prompt += "- PERSON: Sarah Johnson\n";
                prompt += "- EMAIL: sarah@email.com\n";
                prompt += "- PHONE: 555-123-4567\n\n";
                prompt += "Input: Phone: 555 123 0987\nCellphone: 543 123 8765\n";
                prompt += "Output:\n";
                prompt += "- PHONE: 555 123 0987\n";
                prompt += "- PHONE: 543 123 8765\n\n";
                prompt += "Input: I received a delivery at 100 Oak Street, Portland. Great service!\n";
                prompt += "Output:\n";
                prompt += "- ADDRESS: 100 Oak Street, Portland\n\n";
                prompt += "Now extract from this text:\n" + text;
            }

            // Call the appropriate model
            if (usingWllama) {
                // Use Wllama (SmolLM2) with very low temperature to prevent hallucination
                const response = await wllama.createCompletion(prompt, {
                    nPredict: 512,
                    sampling: {
                        temp: 0.0,
                        top_p: 0.9,
                        penalty_repeat: 1.1
                    },
                    stopTokens: ['<|im_end|>', '<|im_start|>']
                });
                aiResponse = response;
            } else {
                // Use WebLLM (Phi 3-mini) with low temperature for factual extraction
                const messages = [
                    { role: "user", content: prompt }
                ];
                const response = await engine.chat.completions.create({
                    messages: messages,
                    temperature: 0.0,
                    max_tokens: 512
                });
                aiResponse = response.choices[0].message.content;
            }

            console.log("AI Response:", aiResponse);

            // Parse the AI response to extract PII terms
            const lines = aiResponse.split('\n');
            const typeMapping = {
                'PERSON': 'Person',
                'PHONE': 'Phone',
                'EMAIL': 'Email',
                'ADDRESS': 'Address'
            };

            lines.forEach(function (line) {
                line = line.trim();
                if (!line || line.length < 3) return;

                // Remove bullet points and list markers
                line = line.replace(/^[-*•]\s*/, '');

                // Try to match format: "TYPE: value" or "TYPE (PERSON, PHONE, etc): value"
                let match = line.match(/^(PERSON|PHONE|EMAIL|ADDRESS)\s*[:(]\s*(.+)$/i);
                if (!match) {
                    // Try alternative format: "value (TYPE)" or "value - TYPE"
                    match = line.match(/^(.+?)\s*[-:(]\s*(PERSON|PHONE|EMAIL|ADDRESS)\s*[)]?$/i);
                    if (match) {
                        match = [match[0], match[2], match[1]];
                    }
                }

                if (match) {
                    const typeRaw = match[1].toUpperCase();
                    let value = match[2].trim();

                    // Clean up the value
                    value = value.replace(/^[:\s)]+|[:\s)]+$/g, '').trim();

                    if (value && typeRaw in typeMapping) {
                        const type = typeMapping[typeRaw];
                        addEntity(entities, value, type);
                    }
                }
            });

        } catch (error) {
            console.error("Error during PII detection:", error);
            throw new Error("Failed to detect PII using AI model: " + error.message);
        }

        // Remove duplicates
        const uniqueEntities = [];
        const seen = new Set();
        entities.forEach(function (entity) {
            const key = entity.type + '|' + entity.value.toLowerCase();
            if (!seen.has(key)) {
                seen.add(key);
                uniqueEntities.push(entity);
            }
        });

        const replacements = {
            Person: "[PERSON]",
            Phone: "[PHONENUMBER]",
            Email: "[EMAIL]",
            Address: "[ADDRESS]"
        };

        const redactedText = uniqueEntities
            .slice()
            .sort(function (left, right) { return right.value.length - left.value.length; })
            .reduce(function (current, entity) {
                return current.replace(new RegExp(escapeRegExp(entity.value), "g"), replacements[entity.type]);
            }, text);

        return {
            entities: uniqueEntities,
            redactedText: redactedText
        };
    }

    function renderLanguage(result) {
        clearResults();
        showResults();

        const radius = 54;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - ((result.confidence / 100) * circumference);

        const confidenceCard = document.createElement("div");
        confidenceCard.className = "result-card";
        confidenceCard.innerHTML = "<h2 class='result-title'>Confidence</h2>";

        const gaugeWrap = document.createElement("div");
        gaugeWrap.className = "gauge-wrap";
        gaugeWrap.setAttribute("aria-label", "Prediction confidence " + result.confidence + " percent");
        gaugeWrap.innerHTML = "<svg class='gauge-svg' viewBox='0 0 140 140' role='img' aria-hidden='true'><circle class='gauge-bg' cx='70' cy='70' r='54'></circle><circle class='gauge-fill' cx='70' cy='70' r='54'></circle></svg><div class='gauge-text'></div><div class='gauge-subtext'>Prediction confidence</div>";
        gaugeWrap.querySelector(".gauge-fill").style.strokeDasharray = String(circumference);
        gaugeWrap.querySelector(".gauge-fill").style.strokeDashoffset = String(offset);
        gaugeWrap.querySelector(".gauge-text").textContent = result.confidence + "%";
        confidenceCard.appendChild(gaugeWrap);

        const resultCard = document.createElement("div");
        resultCard.className = "result-card";
        resultCard.innerHTML = "<h2 class='result-title'>Primary language</h2>";

        const row = document.createElement("div");
        row.className = "lang-row";

        const name = document.createElement("div");
        name.className = "lang-name";
        name.textContent = result.language;

        const code = document.createElement("div");
        code.className = "lang-code";
        code.textContent = result.code;

        row.appendChild(name);
        row.appendChild(code);
        resultCard.appendChild(row);

        elements.results.appendChild(confidenceCard);
        elements.results.appendChild(resultCard);
    }

    function renderPii(result) {
        clearResults();
        showResults();

        const card = document.createElement("div");
        card.className = "result-card";

        const title = document.createElement("h2");
        title.className = "result-title";
        title.textContent = "Detected PII terms";
        card.appendChild(title);

        if (!result.entities.length) {
            const empty = document.createElement("p");
            empty.textContent = "No PII terms detected.";
            card.appendChild(empty);
        } else {
            const list = document.createElement("ul");
            list.className = "pii-list";
            result.entities.forEach(function (entity) {
                const item = document.createElement("li");
                item.className = "pii-item";

                const value = document.createElement("span");
                value.textContent = entity.value;

                const type = document.createElement("span");
                type.className = "pii-type";
                type.textContent = entity.type;

                item.appendChild(value);
                item.appendChild(type);
                list.appendChild(item);
            });
            card.appendChild(list);
        }

        elements.results.appendChild(card);
    }

    function renderError(message) {
        clearResults();
        showResults();

        const card = document.createElement("div");
        card.className = "result-card";

        const title = document.createElement("h2");
        title.className = "result-title error-text";
        title.textContent = "Analysis error";

        const body = document.createElement("p");
        body.textContent = message;

        card.appendChild(title);
        card.appendChild(body);
        elements.results.appendChild(card);
    }

    function updateDetectButton() {
        if (state.locked) {
            return;
        }
        elements.detectBtn.disabled = elements.sourceText.value.trim().length === 0;
    }

    function lockEditor() {
        state.locked = true;
        elements.sourceText.readOnly = true;
        elements.sampleSelect.disabled = true;
        elements.attachBtn.disabled = true;
        elements.detectBtn.disabled = false;
        elements.detectBtn.textContent = "Edit";
        elements.detectBtn.title = "Switch back to editing";
        elements.detectBtn.setAttribute("aria-label", "Switch back to editing");
    }

    function unlockEditor() {
        state.locked = false;
        elements.sourceText.readOnly = false;
        elements.sampleSelect.disabled = false;
        elements.attachBtn.disabled = false;
        elements.detectBtn.textContent = "Detect";
        elements.detectBtn.title = "Analyze text";
        elements.detectBtn.setAttribute("aria-label", "Analyze text");
        elements.sourceText.value = state.originalText;
        clearResults();
        updatePlaceholder();
        updateDetectButton();
        announce("Editing enabled");
    }

    function resetUi() {
        state.locked = false;
        state.originalText = "";
        elements.sourceText.readOnly = false;
        elements.sampleSelect.disabled = false;
        elements.sourceText.value = "";
        elements.sampleSelect.value = "";
        elements.attachBtn.disabled = false;
        elements.detectBtn.textContent = "Detect";
        elements.detectBtn.title = "Analyze text";
        elements.detectBtn.setAttribute("aria-label", "Analyze text");
        elements.detectBtn.disabled = true;
        elements.fileStatus.textContent = "";
        clearResults();
        updatePlaceholder();
    }

    function populateSamples() {
        const samples = SAMPLE_TEXTS[state.mode];
        elements.sampleSelect.innerHTML = "";

        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = "Select a sample";
        elements.sampleSelect.appendChild(defaultOption);

        samples.forEach(function (sample) {
            const option = document.createElement("option");
            option.value = sample.text;
            option.textContent = sample.label;
            elements.sampleSelect.appendChild(option);
        });
    }

    function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".txt") && file.type !== "text/plain") {
            showStatus("Please upload a .txt file.", true);
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            showStatus("File is larger than 10 KB.", true);
            return;
        }

        const reader = new FileReader();
        reader.onload = function () {
            elements.sourceText.value = String(reader.result || "");
            clearResults();
            updatePlaceholder();
            updateDetectButton();
            showStatus("Loaded: " + file.name + " (" + Math.ceil(file.size / 1024) + " KB)", false);
        };
        reader.onerror = function () {
            showStatus("Unable to read file.", true);
        };
        reader.readAsText(file);
    }

    async function analyze() {
        const text = elements.sourceText.value.trim();
        if (!text) {
            showStatus("Add text before running detection.", true);
            return;
        }

        state.originalText = text;

        // Show loading state
        showLoading();

        try {
            if (state.mode === "language") {
                renderLanguage(detectLanguage(text));
                announce("Detected language");
                lockEditor();
            } else {
                // Disable button and show loading state for PII detection
                elements.detectBtn.disabled = true;
                elements.detectBtn.textContent = "Analyzing...";
                showStatus("Detecting PII using AI model...", false);

                const result = await detectPii(text);
                elements.sourceText.value = result.redactedText;
                renderPii(result);
                announce("PII extraction complete");
                showStatus("", false);
                lockEditor();

                elements.detectBtn.disabled = false;
            }
        } catch (error) {
            console.error(error);
            renderError(error instanceof Error ? error.message : "Unexpected error while analyzing text.");
            elements.detectBtn.disabled = false;
            elements.detectBtn.textContent = "Detect";
            showStatus(error.message || "Error during analysis", true);
        }
    }

    function applyTheme() {
        const saved = localStorage.getItem("language-playground-theme");
        const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        const dark = saved ? saved === "dark" : prefersDark;
        elements.themeToggle.checked = dark;
        document.body.classList.toggle("dark", dark);
    }

    function initialize() {
        applyTheme();
        populateSamples();
        elements.results.style.display = "none";
        updatePlaceholder();
        updateDetectButton();

        // Initialize AI models for PII detection
        // Language detection is enabled immediately, but PII extraction
        // and GPU/CPU toggle are disabled until the model loads
        initAIModels();

        elements.analyzerSelect.addEventListener("change", function () {
            state.mode = elements.analyzerSelect.value;
            populateSamples();
            resetUi();
            announce("Analyzer changed");
        });

        elements.sampleSelect.addEventListener("change", function () {
            if (!elements.sampleSelect.value) {
                return;
            }
            elements.sourceText.value = elements.sampleSelect.value;
            clearResults();
            updatePlaceholder();
            updateDetectButton();
        });

        elements.sourceText.addEventListener("input", updateDetectButton);

        elements.attachBtn.addEventListener("click", function () {
            if (!state.locked) {
                elements.fileInput.click();
            }
        });

        elements.fileInput.addEventListener("change", function (event) {
            const file = event.target.files && event.target.files[0];
            if (file) {
                handleFileUpload(file);
            }
            elements.fileInput.value = "";
        });

        elements.detectBtn.addEventListener("click", function () {
            if (state.locked) {
                unlockEditor();
            } else {
                analyze();
            }
        });

        elements.themeToggle.addEventListener("change", function () {
            const dark = elements.themeToggle.checked;
            document.body.classList.toggle("dark", dark);
            localStorage.setItem("language-playground-theme", dark ? "dark" : "light");
        });

        if (elements.modelToggle) {
            elements.modelToggle.addEventListener("change", function () {
                const useGPU = elements.modelToggle.checked;
                switchModel(useGPU);
            });
        }
    }

    initialize();
})();
