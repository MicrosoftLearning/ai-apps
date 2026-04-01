(function () {
    "use strict";

    const MAX_FILE_SIZE = 10 * 1024;

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
            { label: "Contact note", text: "John Smith can be reached at john.smith@contoso.com or +1 555 123 4567. Mailing address: 123 Anystreet, Anytown, WA, USA, 01234." },
            { label: "Customer form", text: "Customer Feedback Form\nDate: 4/1/2026\nCustomer: Mario Gizzi\nEmail: mario@adventure-works.com\nPhone: 555 123 0987\nRating: 5\nComment: Thanks for the great service. I received my delivery at 1482 Westward Way, Seattle on Saturday morning - everything looks great!" }
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

    const PERSON_HEADING_WORDS = new Set([
        "address",
        "billing",
        "city",
        "contact",
        "customer",
        "dear",
        "delivery",
        "email",
        "hello",
        "hey",
        "hi",
        "mailing",
        "message",
        "note",
        "order",
        "phone",
        "record",
        "regards",
        "shipping",
        "subject",
        "thanks"
    ]);

    const PERSON_NON_NAME_WORDS = new Set([
        "abbey",
        "account",
        "address",
        "administrator",
        "airport",
        "arena",
        "avenue",
        "bar",
        "bay",
        "beach",
        "billing",
        "blvd",
        "boulevard",
        "bridge",
        "cafe",
        "castle",
        "cathedral",
        "center",
        "centre",
        "chapel",
        "church",
        "city",
        "close",
        "college",
        "company",
        "contact",
        "corp",
        "court",
        "customer",
        "delivery",
        "drive",
        "edinburgh",
        "email",
        "england",
        "estate",
        "february",
        "garden",
        "gardens",
        "gate",
        "group",
        "grove",
        "hall",
        "harbor",
        "harbour",
        "hello",
        "hey",
        "hi",
        "heights",
        "hill",
        "hospital",
        "hotel",
        "inc",
        "island",
        "january",
        "july",
        "june",
        "lake",
        "lane",
        "limited",
        "llc",
        "london",
        "ltd",
        "mailing",
        "mall",
        "manager",
        "manor",
        "march",
        "market",
        "memorial",
        "message",
        "monument",
        "motel",
        "mountain",
        "museum",
        "note",
        "october",
        "order",
        "park",
        "phone",
        "pier",
        "place",
        "plaza",
        "record",
        "regards",
        "restaurant",
        "river",
        "road",
        "scotland",
        "september",
        "services",
        "shipping",
        "shop",
        "square",
        "stadium",
        "station",
        "store",
        "street",
        "subject",
        "sunday",
        "systems",
        "team",
        "technologies",
        "thanks",
        "theater",
        "theatre",
        "thursday",
        "tower",
        "town",
        "tuesday",
        "uk",
        "university",
        "usa",
        "valley",
        "way",
        "wednesday"
    ]);

    const PERSON_TITLES = /^(?:mr|mrs|ms|miss|dr|prof)\.?\s+/i;

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
        themeToggle: document.getElementById("theme-toggle")
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

    function normalizePersonCandidate(value) {
        return String(value || "")
            .replace(PERSON_TITLES, "")
            .replace(/^[\s,.:;!?()\[\]"'-]+|[\s,.:;!?()\[\]"'-]+$/g, "")
            .replace(/\s+/g, " ")
            .trim();
    }

    function shouldKeepPersonCandidate(value) {
        const normalized = normalizePersonCandidate(value);
        const parts = normalized.split(/\s+/).filter(Boolean);
        if (!parts.length || parts.length > 3) {
            return false;
        }

        const lowered = parts.map(function (part) {
            return part.toLowerCase();
        });

        if (parts.some(function (part) { return part.length < 2; })) {
            return false;
        }

        if (PERSON_HEADING_WORDS.has(lowered[0]) || PERSON_HEADING_WORDS.has(lowered[1] || "")) {
            return false;
        }

        if (lowered.some(function (part) { return PERSON_NON_NAME_WORDS.has(part); })) {
            return false;
        }

        if (parts.length === 1) {
            return /^[A-Z][A-Za-z'-]+$/.test(parts[0]);
        }

        return parts.every(function (part) {
            return /^[A-Z][A-Za-z'-]+$/.test(part);
        });
    }

    function addPersonCandidate(entities, value) {
        const normalized = normalizePersonCandidate(value);
        if (shouldKeepPersonCandidate(normalized)) {
            addEntity(entities, normalized, "Person");
        }
    }

    function collectMatchSpans(regex, text, type) {
        const spans = [];
        const globalRegex = new RegExp(regex.source, regex.flags);
        let match;

        while ((match = globalRegex.exec(text)) !== null) {
            spans.push({
                start: match.index,
                end: match.index + match[0].length,
                value: match[0],
                type: type
            });

            if (match.index === globalRegex.lastIndex) {
                globalRegex.lastIndex += 1;
            }
        }

        return spans;
    }

    function spansOverlap(left, right) {
        return left.start < right.end && right.start < left.end;
    }

    function trimAddressOverlap(addressSpan, protectedSpans, text) {
        let start = addressSpan.start;
        let end = addressSpan.end;

        protectedSpans.forEach(function (span) {
            if (!spansOverlap({ start: start, end: end }, span)) {
                return;
            }

            if (span.start <= start && span.end >= end) {
                start = end;
                return;
            }

            if (span.start <= start) {
                start = span.end;
                return;
            }

            if (span.end >= end) {
                end = span.start;
                return;
            }

            end = span.start;
        });

        if (start >= end) {
            return "";
        }

        return text.slice(start, end)
            .replace(/^[\s,.;:-]+|[\s,.;:-]+$/g, "")
            .trim();
    }

    function looksLikeAddress(value) {
        if (value.length > 100) { return false; }
        const hasStreetSuffix = /\b(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Close|Place|Pl)\b/i.test(value);
        const hasPostalShape = /^\d{1,5}\s+[^,.\n]+(?:,\s*[^,.\n]+){2,4}$/.test(value);
        return /\b\d{1,5}\s+/.test(value)
            && (hasStreetSuffix || hasPostalShape)
            && value.length >= 10;
    }

    function filterRedundantEntities(entities) {
        return entities.filter(function (entity, index, allEntities) {
            if (entity.type !== "Address") {
                return true;
            }

            return !allEntities.some(function (other, otherIndex) {
                if (otherIndex === index || other.type !== "Address") {
                    return false;
                }

                const currentValue = entity.value.trim();
                const otherValue = other.value.trim();

                if (currentValue === otherValue) {
                    return otherIndex < index;
                }

                // If this address contains another address and starts earlier,
                // prefer the cleaner nested address.
                return currentValue.length > otherValue.length
                    && currentValue.includes(otherValue)
                    && currentValue.indexOf(otherValue) > 0;
            });
        });
    }

    function collectCapitalizedPeople(text, entities) {
        const properCaseSequenceRegex = /\b([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){1,2})\b/g;
        let match;

        while ((match = properCaseSequenceRegex.exec(text)) !== null) {
            const candidate = normalizePersonCandidate(match[1]);
            const parts = candidate.split(/\s+/).filter(Boolean);
            const beforeIndex = Math.max(0, match.index - 24);
            const afterIndex = Math.min(text.length, match.index + match[0].length + 24);
            const leftContext = text.slice(beforeIndex, match.index).toLowerCase();
            const rightContext = text.slice(match.index + match[0].length, afterIndex).toLowerCase();

            if (parts.length < 2) {
                continue;
            }

            if (!shouldKeepPersonCandidate(candidate)) {
                continue;
            }

            // Avoid obvious place/address patterns while still allowing narrative prose names.
            if (/\b(?:street|road|avenue|boulevard|lane|drive|court|city|country|mount|lake)\b/.test(rightContext)) {
                continue;
            }

            if (/\b(?:in|at|from|of|to|near)\s+$/.test(leftContext) && parts.length === 2) {
                // Keep if the surrounding sentence looks person-oriented.
                const combinedContext = leftContext + rightContext;
                if (!/\b(?:said|wrote|called|emailed|asked|met|fan|works|lives|sailed|invented|painted|created|discovered|is|was)\b/.test(combinedContext)) {
                    continue;
                }
            }

            addPersonCandidate(entities, candidate);
        }
    }

    function collectRegexPeople(text, entities) {
        const patterns = [
            /(?:^|[\r\n]+|[.!?]\s+)([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){1,2})(?=(?:\s+(?:can|called|emailed|asked|is|was|works|lives|from|at|reached|contacted|sent|wrote|phoned|said))|[,:]|\s*$)/gm,
            /\b(?:i am|i'm|im|my name is|this is|name is|name:)[^\S\r\n]+([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){0,2})(?=(?:\s+(?:from|at|in|on|with|and|can|called|emailed|asked|is|was|works|lives|reached|contacted|sent|wrote|phoned))|[,.!:;)]|\s*$)/gim,
            /\b(?:hello|hi|hey)[^\S\r\n]*,?[^\S\r\n]*(?:i am|i'm|im|this is)?[^\S\r\n]*([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){0,2})(?=(?:\s+(?:from|at|in|on|with|and|can|called|emailed|asked|is|was|works|lives|reached|contacted|sent|wrote|phoned))|[,.!:;)]|\s*$)/gim,
            /\b(?:mr|mrs|ms|miss|dr|prof)\.?[^\S\r\n]+([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){0,2})(?=(?:\s+(?:can|called|emailed|asked|is|was|works|lives|from|at|reached|contacted|sent|wrote|phoned))|[,.!:;)]|\s*$)/gm,
            /\b(?:contact|reach|reached|called by|emailed by|spoken with|met with)[^\S\r\n]+([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){0,2})(?=[,.!:;)]|\s|$)/gim,
            /(?:^|[^A-Za-z])([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){1,2})(?=\s+(?:sailed|said|wrote|discovered|invented|led|found|founded|met|visited|traveled|travelled|built|created|won|lost|became|died|ruled|painted|composed|studied|explored|crossed)\b)/gm,
            /(?:^|[^A-Za-z])([A-Z][A-Za-z'-]+(?:[^\S\r\n]+[A-Z][A-Za-z'-]+){1,2})(?=\s+(?:from|of)\s+[A-Z][A-Za-z'-]+)/gm
        ];

        patterns.forEach(function (pattern) {
            let match;
            while ((match = pattern.exec(text)) !== null) {
                addPersonCandidate(entities, match[1]);
            }
        });
    }

    function detectPii(text) {
        const entities = [];
        const emailRegex = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
        const phoneRegex = /(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}\b/g;
        const streetAddressRegex = /\b\d{1,5}\s+(?:[A-Za-z0-9'-]+\s+){0,5}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Close|Place|Pl)\b(?:,\s*(?:[A-Za-z0-9'-]+\s*){1,4}){0,4}/gi;
        const mailingAddressRegex = /\b\d{1,5}\s+(?:[A-Za-z0-9'-]+\s+){0,4}[A-Za-z0-9'-]+(?:,\s*(?:[A-Za-z0-9'-]+\s*){1,4}){2,4}/gi;
        const protectedSpans = [];

        collectMatchSpans(emailRegex, text, "Email").forEach(function (span) {
            protectedSpans.push(span);
            addEntity(entities, span.value, "Email");
        });

        collectMatchSpans(phoneRegex, text, "Phone").forEach(function (span) {
            protectedSpans.push(span);
            addEntity(entities, span.value, "Phone");
        });

        [streetAddressRegex, mailingAddressRegex].forEach(function (addressRegex) {
            collectMatchSpans(addressRegex, text, "Address").forEach(function (span) {
                const trimmedAddress = trimAddressOverlap(span, protectedSpans, text);
                if (trimmedAddress && looksLikeAddress(trimmedAddress)) {
                    addEntity(entities, trimmedAddress, "Address");
                }
            });
        });

        if (window.nlp && typeof window.nlp === "function") {
            window.nlp(text).people().out("array").forEach(function (value) {
                addPersonCandidate(entities, value);
            });
        }

        collectRegexPeople(text, entities);
        collectCapitalizedPeople(text, entities);

        const filteredEntities = filterRedundantEntities(entities);

        const replacements = {
            Person: "[PERSON]",
            Phone: "[PHONENUMBER]",
            Email: "[EMAIL]",
            Address: "[ADDRESS]"
        };

        const redactedText = filteredEntities
            .slice()
            .sort(function (left, right) { return right.value.length - left.value.length; })
            .reduce(function (current, entity) {
                return current.replace(new RegExp(escapeRegExp(entity.value), "g"), replacements[entity.type]);
            }, text);

        return {
            entities: filteredEntities,
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

    function analyze() {
        const text = elements.sourceText.value.trim();
        if (!text) {
            showStatus("Add text before running detection.", true);
            return;
        }

        state.originalText = text;

        try {
            if (state.mode === "language") {
                renderLanguage(detectLanguage(text));
                announce("Detected language");
            } else {
                const result = detectPii(text);
                elements.sourceText.value = result.redactedText;
                renderPii(result);
                announce("PII extraction complete");
            }
            lockEditor();
        } catch (error) {
            console.error(error);
            renderError(error instanceof Error ? error.message : "Unexpected error while analyzing text.");
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
    }

    initialize();
})();
