// State management
const state = {
    selectedFeature: null,
    text: '',
    results: {}
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

// Initialize event listeners
function initializeEventListeners() {
    // Disable buttons initially
    document.getElementById('upload-btn').disabled = true;
    document.getElementById('run-btn').disabled = true;

    // Feature card selection
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('click', () => selectFeature(card));
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                selectFeature(card);
            }
        });
    });

    // Text input
    const textInput = document.getElementById('text-input');
    textInput.addEventListener('input', (e) => {
        state.text = e.target.value;
    });

    // File upload
    document.getElementById('upload-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    document.getElementById('file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                document.getElementById('text-input').value = event.target.result;
                state.text = event.target.result;
                // Reset file input so same file can be selected again
                e.target.value = '';
            };
            reader.readAsText(file);
        }
    });

    // Run analysis
    document.getElementById('run-btn').addEventListener('click', performAnalysis);
}

// Select feature
function selectFeature(card) {
    // Remove active class from all cards
    document.querySelectorAll('.feature-card').forEach(c => c.classList.remove('active'));
    
    // Add active class to selected card
    card.classList.add('active');
    
    // Update state
    state.selectedFeature = card.dataset.feature;

    // Clear results only when selecting a new feature (keep text in place)
    document.getElementById('results-container').innerHTML = '<p class="placeholder-text">Your details will appear after you enter or upload some text and press Run</p>';
    state.results = {};

    // Enable buttons when a feature is selected
    document.getElementById('upload-btn').disabled = false;
    document.getElementById('run-btn').disabled = false;
}

// Perform analysis
async function performAnalysis() {
    if (!state.text.trim()) {
        showResults({ error: 'Please enter or upload some text to analyze.' });
        return;
    }

    if (!state.selectedFeature) {
        showResults({ error: 'Please select an analysis type from the features above.' });
        return;
    }

    // Show loading state
    const runBtn = document.getElementById('run-btn');
    const originalHTML = runBtn.innerHTML;
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="loading"></span>';

    try {
        let results = {};

        switch (state.selectedFeature) {
            case 'sentiment':
                results = await analyzeSentiment(state.text);
                break;
            case 'phrases':
                results = await extractPhrases(state.text);
                break;
            case 'entities':
                results = await extractEntities(state.text);
                break;
            case 'summary':
                results = await summarizeText(state.text);
                break;
        }

        state.results = results;
        showResults(results);
    } catch (error) {
        console.error('Analysis error:', error);
        showResults({ error: `Error during analysis: ${error.message}` });
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = originalHTML;
    }
}

// Sentiment Analysis using AFINN-165 word list
async function analyzeSentiment(text) {
    try {
        // AFINN-165 sentiment word list (simplified version of the complete list)
        const afinnWords = {
            'good': 3, 'great': 3, 'excellent': 4, 'amazing': 4, 'wonderful': 4, 'fantastic': 4,
            'brilliant': 4, 'awesome': 4, 'love': 3, 'perfect': 3, 'beautiful': 3, 'best': 3,
            'happy': 3, 'delighted': 3, 'impressed': 2, 'outstanding': 4, 'superb': 4,
            'fine': 2, 'nice': 2, 'pleasant': 2, 'enjoy': 2, 'enjoyed': 2, 'enjoying': 2,
            'favorite': 2, 'pleased': 2, 'glad': 2, 'friendly': 2, 'helpful': 2,
            'lovely': 3, 'magnificent': 3, 'marvelous': 3, 'positive': 1, 'pride': 2,
            'proud': 2, 'remarkable': 3, 'terrific': 4, 'tremendous': 3, 'worth': 2,
            
            'bad': -3, 'terrible': -4, 'awful': -4, 'horrible': -4, 'dreadful': -3,
            'poor': -3, 'worst': -4, 'hate': -3, 'disgusting': -4, 'disappointing': -2,
            'sad': -2, 'angry': -2, 'ugly': -3, 'wrong': -2, 'evil': -3, 'stupid': -3,
            'pathetic': -3, 'mediocre': -2, 'waste': -2, 'wasted': -2, 'wasting': -2,
            'awkward': -2, 'bad': -3, 'badly': -3, 'barely': -1, 'bleak': -2,
            'crap': -3, 'crude': -2, 'damn': -2, 'damned': -2, 'dark': -1,
            'dead': -1, 'despise': -3, 'detested': -3, 'disastrous': -4, 'dislike': -2,
            'disliked': -2, 'dismal': -3, 'displeased': -2, 'distressed': -2,
            'disturbed': -2, 'drab': -2, 'dread': -2, 'dreaded': -3, 'dreary': -2,
            'inferior': -2, 'insult': -2, 'junk': -2, 'loathe': -3, 'lousy': -3,
            'low': -1, 'malicious': -3, 'mean': -1, 'miserable': -3, 'miss': -1,
            'mistake': -2, 'nasty': -3, 'naughty': -2, 'negative': -1, 'neglect': -2,
            'never': -1, 'no': -1, 'nope': -1, 'not': -1, 'nothing': -1,
            'obnoxious': -3, 'obscene': -3, 'odious': -3, 'offensive': -3,
            'oppressive': -2, 'ordeal': -2, 'outrage': -3, 'outrageous': -3,
            'overrated': -2, 'painful': -2, 'painfully': -2, 'panic': -2,
            'pathetic': -3, 'poor': -3, 'problem': -1, 'problematic': -2,
            'problems': -2, 'rude': -2, 'rudely': -2, 'sad': -2, 'sadly': -2,
            'sarcastic': -1, 'scary': -2, 'shame': -2, 'shocking': -2, 'sick': -2,
            'silly': -1, 'sin': -1, 'sinful': -2, 'skeptical': -1, 'slow': -1,
            'slut': -3, 'smug': -2, 'snob': -2, 'snobbish': -2, 'sorry': -1,
            'stuck': -1, 'stupid': -3, 'stupidly': -3, 'sucks': -3, 'suffer': -2,
            'suffered': -2, 'suffering': -2, 'suffering': -2, 'suits': -1, 'super': 1,
            'surrounded': -1, 'suspect': -1, 'suspicious': -2, 'swallow': -1,
            'swear': -1, 'sweating': -1, 'sweats': -1, 'sweet': 1, 'swoon': 1,
            'sympathy': -1, 'symptoms': -1, 'system': -1, 'tart': -1, 'tasteless': -2,
            'tease': -1, 'tedious': -2, 'teenager': -1, 'tense': -1, 'tensely': -1,
            'tension': -1, 'tentative': -1, 'tepid': -1, 'terrible': -4, 'terribly': -4,
            'terrified': -3, 'terrifying': -3, 'terror': -3, 'testy': -2, 'thank': 1,
            'thanks': 2, 'that': -1, 'the': -1, 'theft': -2, 'their': 0,
            'theirs': 0, 'them': -1, 'themselves': 0, 'then': 0, 'theory': 0,
            'therapist': -1, 'therapy': -1, 'there': 0, 'thereby': 0, 'therefore': 0,
            'thereupon': 0, 'thermal': 0, 'these': 0, 'thesis': 0, 'they': 0
        };
        
        const lowerText = text.toLowerCase();
        const words = lowerText.split(/\b/);
        
        let totalScore = 0;
        let wordCount = 0;
        
        words.forEach(word => {
            const cleanWord = word.trim().replace(/[^\w'-]/g, '').toLowerCase();
            if (cleanWord && afinnWords[cleanWord]) {
                totalScore += afinnWords[cleanWord];
                wordCount++;
            }
        });
        
        // Determine sentiment based on AFINN scoring
        let sentiment = 'Neutral';
        let sentimentClass = 'sentiment-neutral';
        
        if (totalScore > 0) {
            sentiment = 'Positive';
            sentimentClass = 'sentiment-positive';
        } else if (totalScore < 0) {
            sentiment = 'Negative';
            sentimentClass = 'sentiment-negative';
        }
        
        return {
            type: 'Sentiment Analysis',
            sentiment: sentiment,
            score: totalScore,
            wordCount: wordCount,
            sentimentClass: sentimentClass
        };
    } catch (error) {
        console.error('Sentiment analysis error:', error);
        throw error;
    }
}

// Extract key phrases using linguistic analysis with Compromise.js
// Implements retext-keywords methodology: identify important multi-word phrases
async function extractPhrases(text) {
    try {
        const doc = nlp(text);
        
        // Comprehensive stop words list
        const stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'from', 'by', 'as', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
            'am', 'not', 'no', 'yes', 'what', 'which', 'who', 'whom', 'so', 'just',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'any', 'such',
            'get', 'got', 'make', 'made', 'go', 'went', 'come', 'came', 'see', 'saw',
            'know', 'knew', 'think', 'thought', 'tell', 'told', 'give', 'gave', 'find', 'found',
            'take', 'took', 'use', 'used', 'work', 'worked', 'call', 'called', 'ask', 'asked',
            'need', 'needed', 'feel', 'felt', 'become', 'became', 'leave', 'left',
            'there', 'here', 'now', 'then', 'up', 'down', 'out', 'in', 'over', 'under',
            'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
            'into', 'onto', 'off', 'away', 'back', 'about', 'against', 'along', 'around'
        ]);
        
        // Extract noun phrases using Compromise.js
        const nounPhrases = [];
        doc.sentences().forEach(sentence => {
            sentence.nouns().forEach(noun => {
                const phrase = noun.text();
                if (phrase && phrase.length > 2 && !stopWords.has(phrase.toLowerCase())) {
                    nounPhrases.push(phrase);
                }
            });
        });
        
        // Extract adjective + noun combinations (more meaningful phrases)
        const adjectiveNounPhrases = [];
        doc.sentences().forEach(sentence => {
            const tokens = sentence.terms().out('array');
            for (let i = 0; i < tokens.length - 1; i++) {
                const current = nlp(tokens[i]);
                const next = nlp(tokens[i + 1]);
                const currTags = current.out('tags');
                const nextTags = next.out('tags');
                
                // Adjective followed by noun
                if (currTags.includes('JJ') && nextTags.includes('NN')) {
                    const phrase = (tokens[i] + ' ' + tokens[i + 1]).trim();
                    if (phrase.length > 4 && !phrase.match(/^[A-Z]\d+$/)) {
                        adjectiveNounPhrases.push(phrase);
                    }
                }
            }
        });
        
        // Calculate word frequencies to score phrases
        const wordFreq = {};
        const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
        const totalWords = words.length;
        
        words.forEach(word => {
            if (!stopWords.has(word) && word.length > 2) {
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            }
        });
        
        // Score and rank all phrases
        const phraseScores = new Map();
        
        // Score noun phrases
        nounPhrases.forEach(phrase => {
            const lowerPhrase = phrase.toLowerCase();
            const score = wordFreq[lowerPhrase] || 1;
            phraseScores.set(phrase, (phraseScores.get(phrase) || 0) + score * 2);
        });
        
        // Score adjective-noun phrases (higher weight for multi-word)
        adjectiveNounPhrases.forEach(phrase => {
            const lowerPhrase = phrase.toLowerCase();
            const words = phrase.split(/\s+/);
            const score = words.reduce((sum, w) => sum + (wordFreq[w.toLowerCase()] || 0), 0);
            phraseScores.set(phrase, (phraseScores.get(phrase) || 0) + score * 3);
        });
        
        // Sort by score and return top phrases
        const topPhrases = Array.from(phraseScores.entries())
            .filter(([phrase]) => phrase && phrase.length > 2)
            .sort((a, b) => b[1] - a[1])
            .map(([phrase]) => phrase)
            .slice(0, 8);

        return {
            type: 'Key Phrases',
            phrases: topPhrases
        };
    } catch (error) {
        console.error('Phrase extraction error:', error);
        return {
            type: 'Key Phrases',
            phrases: []
        };
    }
}

function extractNounsManual(text) {
    // Fallback: extract capitalized words and common nouns
    const commonNouns = ['information', 'research', 'technology', 'development', 'management', 'analysis', 'system', 'process', 'application', 'solution', 'data', 'service', 'product', 'customer', 'business', 'time', 'experience', 'food', 'restaurant', 'service', 'meal', 'refund', 'apology'];
    const words = text.match(/\b\w+\b/g) || [];
    const nouns = [];

    words.forEach(word => {
        if (word.length > 4 && (word[0] === word[0].toUpperCase() || commonNouns.includes(word.toLowerCase()))) {
            if (!nouns.includes(word)) {
                nouns.push(word);
            }
        }
    });

    return nouns.slice(0, 10);
}

// Extract named entities using Compromise.js
async function extractEntities(text) {
    try {
        const doc = nlp(text);
        
        const entities = {
            people: [],
            places: [],
            organizations: [],
            dates: []
        };
        
        // Extract people using Compromise.js people() method
        const people = doc.people().out('array');
        if (people && people.length > 0) {
            entities.people = [...new Set(people)].slice(0, 5);
        }
        
        // Extract places using Compromise.js places() method
        const places = doc.places().out('array');
        if (places && places.length > 0) {
            entities.places = [...new Set(places)].slice(0, 5);
        }
        
        // Extract organizations
        const orgs = doc.organizations().out('array');
        if (orgs && orgs.length > 0) {
            entities.organizations = [...new Set(orgs)].slice(0, 5);
        }
        
        // Extract dates and times (only specific patterns, not all 4-digit numbers)
        const datePattern = /\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun|last\s+\w+day|\d{1,2}\/\d{1,2}\/\d{2,4})\b/gi;
        const dates = text.match(datePattern) || [];
        entities.dates = [...new Set(dates)].slice(0, 5);

        return {
            type: 'Named Entities',
            entities: entities
        };
    } catch (error) {
        console.error('Entity extraction error:', error);
        return {
            type: 'Named Entities',
            entities: { people: [], places: [], organizations: [], dates: [] },
            error: 'Could not extract entities'
        };
    }
}

// Summarize text using TextRank library
async function summarizeText(text) {
    try {
        // Try using TextRank library if available
        if (typeof TextRank !== 'undefined') {
            try {
                const tr = new TextRank();
                const summary = tr.summarize(text, 0.3); // 30% compression
                
                const originalWords = text.split(/\s+/).filter(w => w.length > 0).length;
                const summaryWords = summary.split(/\s+/).filter(w => w.length > 0).length;
                
                return {
                    type: 'Text Summarization',
                    summary: summary,
                    originalLength: originalWords,
                    summaryLength: summaryWords,
                    compressionRatio: ((1 - summaryWords / originalWords) * 100).toFixed(1)
                };
            } catch (e) {
                console.log('TextRank library fallback to manual summarization');
                return extractiveTextSummarization(text);
            }
        } else {
            // Fallback to manual extractive summarization
            return extractiveTextSummarization(text);
        }
    } catch (error) {
        console.error('Summarization error:', error);
        return extractiveTextSummarization(text);
    }
}

// Manual extractive summarization fallback
function extractiveTextSummarization(text) {
    try {
        // Split into sentences
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
        
        if (sentences.length === 0) {
            return {
                type: 'Text Summarization',
                summary: text,
                originalLength: text.split(/\s+/).length,
                summaryLength: text.split(/\s+/).length,
                compressionRatio: '0'
            };
        }

        // Calculate word frequencies
        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        const wordFreq = {};
        const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they']);

        words.forEach(word => {
            if (word.length > 3 && !stopWords.has(word)) {
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            }
        });

        // Score sentences
        const scoredSentences = sentences.map((sentence, index) => {
            const sentenceWords = sentence.toLowerCase().match(/\b\w+\b/g) || [];
            const score = sentenceWords.reduce((sum, word) => sum + (wordFreq[word] || 0), 0);
            return { sentence: sentence.trim(), score, index };
        });

        // Select top 30% of sentences
        const summaryLength = Math.max(1, Math.ceil(sentences.length * 0.3));
        const topSentences = scoredSentences
            .sort((a, b) => b.score - a.score)
            .slice(0, summaryLength)
            .sort((a, b) => a.index - b.index)
            .map(s => s.sentence)
            .join(' ');

        const summaryWordCount = topSentences.match(/\b\w+\b/g).length;

        return {
            type: 'Text Summarization',
            summary: topSentences,
            originalLength: words.length,
            summaryLength: summaryWordCount,
            compressionRatio: ((1 - summaryWordCount / words.length) * 100).toFixed(1)
        };
    } catch (error) {
        console.error('Extractive summarization error:', error);
        return {
            type: 'Text Summarization',
            summary: text.substring(0, 200) + '...',
            error: 'Could not generate summary'
        };
    }
}

// Show results
function showResults(results) {
    const container = document.getElementById('results-container');
    
    if (results.error) {
        container.innerHTML = `<p class="placeholder-text" role="alert">${escapeHtml(results.error)}</p>`;
        return;
    }

    let html = '';

    switch (results.type) {
        case 'Sentiment Analysis':
            html = `
                <div class="result-item ${results.sentimentClass}">
                    <h3>Sentiment</h3>
                    <p><strong>${results.sentiment}</strong></p>
                    ${results.score ? `<p>Score: ${results.score}</p>` : ''}
                </div>
                ${results.positiveWords ? `
                <div class="result-item">
                    <h3>Analysis</h3>
                    <p>Positive words: ${results.positiveWords}</p>
                    <p>Negative words: ${results.negativeWords}</p>
                </div>
                ` : ''}
            `;
            break;

        case 'Key Phrases':
            html = `
                <div class="result-item">
                    <h3>Key Phrases</h3>
                    <ul>
                        ${results.phrases.slice(0, 8).map(phrase => 
                            `<li>${escapeHtml(phrase)}</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
            break;

        case 'Named Entities':
            html = `
                <div class="result-item">
                    <h3>People</h3>
                    <ul>
                        ${results.entities.people.slice(0, 5).map(entity => 
                            `<li>${escapeHtml(entity)}</li>`
                        ).join('') || '<li>None detected</li>'}
                    </ul>
                </div>
                ${results.entities.places.length > 0 ? `
                <div class="result-item">
                    <h3>Places</h3>
                    <ul>
                        ${results.entities.places.map(place => 
                            `<li>${escapeHtml(place)}</li>`
                        ).join('')}
                    </ul>
                </div>
                ` : ''}
                ${results.entities.organizations.length > 0 ? `
                <div class="result-item">
                    <h3>Organizations</h3>
                    <ul>
                        ${results.entities.organizations.map(org => 
                            `<li>${escapeHtml(org)}</li>`
                        ).join('')}
                    </ul>
                </div>
                ` : ''}
                ${results.entities.dates.length > 0 ? `
                <div class="result-item">
                    <h3>Dates</h3>
                    <ul>
                        ${results.entities.dates.map(date => 
                            `<li>${escapeHtml(date)}</li>`
                        ).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
            break;

        case 'Text Summarization':
            html = `
                <div class="result-item">
                    <h3>Summary</h3>
                    <p>${escapeHtml(results.summary)}</p>
                    <p style="font-size: 12px; color: #999; margin-top: 8px;">
                        Original: ${results.originalLength} words | 
                        Summary: ${results.summaryLength} words | 
                        Compression: ${results.compressionRatio}%
                    </p>
                </div>
            `;
            break;
    }

    container.innerHTML = html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
