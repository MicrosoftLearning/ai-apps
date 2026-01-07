import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm";

class AskAndrew {
    constructor() {
        this.engine = null;
        this.conversationHistory = [];
        this.isGenerating = false;
        this.indexData = null; // Contains the category structure from index.json
        this.stopRequested = false;
        this.currentStream = null;
        this.webGPUAvailable = false;
        this.simpleMode = false;
        this.currentModal = null;
        this.lastFocusedElement = null;
        this.modalFocusTrapHandler = null;
        
        this.elements = {
            progressSection: document.getElementById('progress-section'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            chatContainer: document.getElementById('chat-container'),
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            restartBtn: document.getElementById('restart-btn'),
            searchStatus: document.getElementById('search-status'),
            modeToggle: document.getElementById('mode-toggle'),
            aboutBtn: document.getElementById('about-btn'),
            aboutModal: document.getElementById('about-modal'),
            aboutModalClose: document.getElementById('about-modal-close'),
            aboutModalOk: document.getElementById('about-modal-ok'),
            aiModeModal: document.getElementById('ai-mode-modal'),
            modalClose: document.getElementById('modal-close'),
            modalOk: document.getElementById('modal-ok')
        };
        
        this.systemPrompt = `You are Andrew, a knowledgeable and friendly AI learning assistant who helps students understand AI concepts.

IMPORTANT: Follow these guidelines when responding:
- Explain concepts clearly and concisely in a single paragraph based only on the provided context.
- Keep responses short and focused on the question, with no headings.
- Use examples and analogies when helpful.
- Use simple language suitable for learners in a conversational, friendly tone.
- Provide a general descriptions and overviews, but do NOT provide explicit steps or instructions for developing AI solutions.
- If the context includes "Sorry, I couldn't find any specific information on that topic. Please try rephrasing your question or explore other AI concepts.", use that exact phrasing and no additional information.
- Do not start responses with "A:" or "Q:".
- Keep your responses concise and to the the point.
- Do NOT provide links for more information (these will be added automatically later).`;

        this.initialize();
    }

    async initialize() {
        try {
            // Load the index
            await this.loadIndex();
            
            // Initialize WebLLM
            await this.initializeWebLLM();
            
            // Setup event listeners
            this.setupEventListeners();
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize. Please refresh the page.');
        }
    }

    async loadIndex() {
        try {
            this.updateProgress(5, 'Loading knowledge base...');
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

    async initializeWebLLM() {
        try {
            this.updateProgress(15, 'Loading AI model...');
            
            const targetModelId = 'Phi-3-mini-4k-instruct-q4f16_1-MLC';
            
            this.engine = await webllm.CreateMLCEngine(
                targetModelId,
                {
                    initProgressCallback: (progress) => {
                        const percentage = Math.max(15, Math.round(progress.progress * 85) + 15);
                        this.updateProgress(
                            percentage, 
                            `Loading model: ${Math.round(progress.progress * 100)}%`
                        );
                    }
                }
            );
            
            this.updateProgress(100, 'Ready to chat!');
            console.log('WebLLM engine initialized successfully');
            this.webGPUAvailable = true;
            
            setTimeout(() => {
                this.showChatInterface();
            }, 500);
            
        } catch (error) {
            console.error('Failed to initialize WebLLM:', error);
            console.log('Falling back to simple mode');
            this.webGPUAvailable = false;
            this.simpleMode = true;
            this.updateProgress(100, 'Ready to chat! (Simple mode)');
            
            setTimeout(() => {
                this.showChatInterface();
            }, 500);
        }
    }

    updateProgress(percentage, text) {
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = text;
        
        // Update progress bar ARIA attributes
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.setAttribute('aria-label', text);
        }
    }

    showChatInterface() {
        this.elements.progressSection.style.display = 'none';
        this.elements.chatContainer.style.display = 'flex';
        this.updateModeToggle();
        this.elements.userInput.focus();
    }

    showError(message) {
        this.elements.progressText.textContent = message;
        this.elements.progressFill.style.backgroundColor = '#dc3545';
    }

    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => {
            if (this.isGenerating) {
                this.stopGeneration();
            } else {
                this.sendMessage();
            }
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
        
        // Keyboard navigation
        this.elements.userInput.addEventListener('keydown', (e) => {
            // Enter to send (without Shift)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.isGenerating) {
                    this.sendMessage();
                }
            }
            // Escape to stop generation
            if (e.key === 'Escape' && this.isGenerating) {
                this.stopGeneration();
            }
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
        
        // Restart button
        this.elements.restartBtn.addEventListener('click', () => {
            this.restartConversation();
        });
        
        // Mode toggle button
        this.elements.modeToggle.addEventListener('click', () => {
            this.toggleMode();
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
        
        // Modal handlers
        this.elements.modalClose.addEventListener('click', () => {
            this.hideAiModeModal();
        });
        
        this.elements.modalOk.addEventListener('click', () => {
            this.hideAiModeModal();
        });
        
        // Close modal on overlay click
        this.elements.aiModeModal.addEventListener('click', (e) => {
            if (e.target === this.elements.aiModeModal || e.target.classList.contains('modal-overlay')) {
                this.hideAiModeModal();
            }
        });
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.elements.aiModeModal.style.display === 'flex') {
                    this.hideAiModeModal();
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

        // Dynamic AI mode link keyboard handling (for links added to messages)
        this.elements.chatMessages.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.classList.contains('ai-mode-link')) {
                e.preventDefault();
                e.target.click();
            }
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
        
        // Clear input and reset height
        this.elements.userInput.value = '';
        this.elements.userInput.style.height = 'auto';
        
        // Add user message to chat
        this.addMessage('user', userMessage);
        
        // Check if this is an initial greeting (only if no messages yet)
        const messageCount = this.elements.chatMessages.querySelectorAll('.message').length;
        if (messageCount <= 1) { // Only user's message is in chat
            const greetingPattern = /^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\s!?]*$/i;
            if (greetingPattern.test(userMessage)) {
                // Respond with greeting without searching
                const greetingResponse = "Hello, I'm Andrew. I'm here to help you learn about AI concepts. What would you like to know?";
                this.addMessage('assistant', greetingResponse);
                return;
            }
        }
        
        // Search for relevant context
        const searchResult = this.searchContext(userMessage);
        
        // Generate response
        await this.generateResponse(userMessage, searchResult);
    }

    updateSendButton(isGenerating) {
        const sendIcon = this.elements.sendBtn.querySelector('.send-icon');
        if (isGenerating) {
            sendIcon.textContent = '■';
            this.elements.sendBtn.title = 'Stop generation';
            this.elements.sendBtn.setAttribute('aria-label', 'Stop generation');
        } else {
            sendIcon.textContent = '▶';
            this.elements.sendBtn.title = 'Send message';
            this.elements.sendBtn.setAttribute('aria-label', 'Send message');
        }
    }

    stopGeneration() {
        this.stopRequested = true;
        console.log('Stop requested');
    }

    toggleMode() {
        if (!this.webGPUAvailable) {
            alert('WebGPU mode is not available in this browser. Simple mode is the only option.');
            return;
        }
        
        this.simpleMode = !this.simpleMode;
        this.updateModeToggle();
        
        const mode = this.simpleMode ? 'Simple' : 'AI';
        console.log(`Switched to ${mode} mode`);
        
        // Add a system message to indicate mode change
        this.addSystemMessage(`Switched to ${mode} mode`);
    }

    updateModeToggle() {
        // Show current mode state, not the mode to switch to
        const modeText = this.simpleMode ? '📝 Simple Mode: ON' : '🤖 AI Mode: ON';
        const modeTitle = this.simpleMode ? 
            'Currently in Simple mode (search only). Click to switch to AI mode.' : 
            'Currently in AI mode (uses WebGPU). Click to switch to Simple mode.';
        const ariaLabel = this.simpleMode ?
            'Toggle chat mode. Currently in Simple mode. Click to switch to AI mode.' :
            'Toggle chat mode. Currently in AI mode. Click to switch to Simple mode.';
        
        this.elements.modeToggle.textContent = modeText;
        this.elements.modeToggle.title = modeTitle;
        this.elements.modeToggle.setAttribute('aria-label', ariaLabel);
        this.elements.modeToggle.setAttribute('aria-pressed', 'true');
        
        // Disable toggle if WebGPU not available
        if (!this.webGPUAvailable) {
            this.elements.modeToggle.textContent = '📝 Simple Mode: ON';
            this.elements.modeToggle.disabled = true;
            this.elements.modeToggle.title = 'WebGPU not available - Simple mode only';
            this.elements.modeToggle.setAttribute('aria-label', 'Chat mode set to Simple mode only. WebGPU not available.');
            this.elements.modeToggle.setAttribute('aria-disabled', 'true');
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

    async generateResponse(userMessage, searchResult) {
        // Use simple mode if explicitly enabled or if WebGPU not available
        if (this.simpleMode || !this.webGPUAvailable) {
            this.generateSimpleResponse(userMessage, searchResult);
            return;
        }
        
        const { context, categories, links } = searchResult;
        
        this.isGenerating = true;
        this.stopRequested = false;
        this.updateSendButton(true);
        
        // Add empty message that we'll stream into
        const responseMessage = this.addMessage('assistant', '', false);
        const messageTextDiv = responseMessage.querySelector('.message-text');
        messageTextDiv.innerHTML = '<span class="typing-indicator">●●●</span>';
        
        try {
            // Build a concise prompt with context
            let userPrompt = userMessage;
            if (context) {
                userPrompt = `${context}\n\nQ: ${userMessage}`;
            }
            
            // Keep only last 3 conversation turns to stay within context limits
            const recentHistory = this.conversationHistory.slice(-6); // 3 turns = 6 messages
            
            // Add to conversation history
            recentHistory.push({
                role: 'user',
                content: userPrompt
            });
            
            // Generate response with streaming
            const messages = [
                { role: 'system', content: this.systemPrompt },
                ...recentHistory
            ];
            
            const completion = await this.engine.chat.completions.create({
                messages: messages,
                temperature: 0.7,
                max_tokens: 500,
                stream: true // Enable streaming
            });
            
            this.currentStream = completion;
            let assistantMessage = '';
            
            // Stream the response
            for await (const chunk of completion) {
                if (this.stopRequested) {
                    console.log('Generation stopped by user');
                    break;
                }
                
                const delta = chunk.choices[0]?.delta?.content;
                if (delta) {
                    assistantMessage += delta;
                    // Update the message as we receive chunks
                    messageTextDiv.innerHTML = this.formatResponse(assistantMessage);
                    this.scrollToBottom();
                }
            }
            
            // Add learn more links
            if (links && links.length > 0 && categories && categories.length > 0) {
                // Store original message without learn more for conversation history
                const originalMessage = assistantMessage;
                
                // Add placeholder for learn more section for display only
                assistantMessage += '\n\n---\n\n**Learn more:** [[LEARN_MORE_LINKS]]';
                
                // Format the message
                let formattedMessage = this.formatResponse(assistantMessage);
                
                // Build HTML links with category names
                const linkHtml = links.map((link, index) => {
                    const categoryName = categories[Math.min(index, categories.length - 1)];
                    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
                }).join(' • ');
                formattedMessage = formattedMessage.replace(/\[\[LEARN_MORE_LINKS\]\]/g, linkHtml);
                
                messageTextDiv.innerHTML = formattedMessage;
                
                // Reset assistantMessage to original for conversation history
                assistantMessage = originalMessage;
            }
            
            // Add to full conversation history (keep complete history)
            this.conversationHistory.push({
                role: 'user',
                content: userMessage // Store original question, not the one with context
            });
            this.conversationHistory.push({
                role: 'assistant',
                content: assistantMessage
            });
            
        } catch (error) {
            console.error('Error generating response:', error);
            responseMessage.remove();
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again or switch to Simple mode.');
        } finally {
            this.isGenerating = false;
            this.stopRequested = false;
            this.currentStream = null;
            this.updateSendButton(false);
            
            // Clear search status after response is complete
            setTimeout(() => {
                this.elements.searchStatus.textContent = '';
            }, 2000);
        }
    }

    formatResponse(text) {
        // Split out the learn more section and note if they exist
        const learnMoreMatch = text.match(/([\s\S]*?)(---\s*\n\n\*\*Learn more:\*\*.*?)(\n\n\*Note:.*)?$/);
        
        if (learnMoreMatch) {
            const mainContent = learnMoreMatch[1];
            const learnMoreSection = learnMoreMatch[2];
            const noteSection = learnMoreMatch[3] || '';
            
            // Format main content (escape HTML)
            let formatted = this.escapeHtml(mainContent);
            
            // Convert **bold** to <strong>
            formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            
            // Convert *italic* to <em>
            formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
            
            // Convert line breaks to paragraphs
            formatted = formatted.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
            
            // Add learn more section - preserve placeholders and HTML structure
            const learnMoreFormatted = learnMoreSection
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/---\s*\n\n/g, '<hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;">\n\n');
            
            // Format note section - preserve placeholders
            let noteFormatted = '';
            if (noteSection) {
                // Extract the note text (remove leading \n\n*Note: and trailing *)
                let noteText = noteSection.replace(/^\n\n\*Note:\s*/g, '').replace(/\*$/g, '');
                // Wrap in styled paragraph - placeholders will be replaced by caller
                noteFormatted = `<p style="font-style: italic; color: #666; font-size: 0.9em; margin-top: 10px;">Note: ${noteText}</p>`;
            }
            
            return formatted + learnMoreFormatted + noteFormatted;
        }
        
        // No learn more section, process normally
        let formatted = this.escapeHtml(text);
        
        // Convert **bold** to <strong>
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Convert *italic* to <em>
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // Convert line breaks to paragraphs
        formatted = formatted.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
        
        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async animateTyping(element, htmlContent, speed = 5) {
        // Parse HTML to extract text while preserving structure
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;
        
        // For simple animation, just show the content progressively
        element.innerHTML = '';
        const words = htmlContent.split(' ');
        
        for (let i = 0; i < words.length; i++) {
            if (this.stopRequested) break;
            
            element.innerHTML = words.slice(0, i + 1).join(' ');
            this.scrollToBottom();
            
            // Small delay between words
            await new Promise(resolve => setTimeout(resolve, speed));
        }
        
        // Ensure final content is complete
        element.innerHTML = htmlContent;
        this.scrollToBottom();
    }

    async generateSimpleResponse(userMessage, searchResult) {
        const { context, categories, links, documents } = searchResult || { context: null, categories: [], links: [], documents: [] };
        
        // If no matches, use the fallback from searchContext (AI Concepts category)
        if (!documents || documents.length === 0) {
            let fallbackResponse = "I don't have specific information about that in my knowledge base. Could you try rephrasing your question or asking about a different AI topic?";
            
            // If we have fallback content from AI Concepts, use it
            if (context) {
                fallbackResponse = context;
            }
            
            // Build HTML response directly
            let formattedResponse = `<p>${this.escapeHtml(fallbackResponse).replace(/\n/g, '<br>')}</p>`;
            
            // Add learn more link if available
            if (links && links.length > 0 && categories && categories.length > 0) {
                const linkHtml = links.map((link, index) => {
                    const categoryName = categories[Math.min(index, categories.length - 1)];
                    return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
                }).join(' • ');
                formattedResponse += `<hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;"><p><strong>Learn more:</strong> ${linkHtml}</p>`;
            }
            
            // Add AI mode note
            formattedResponse += `<p style="font-style: italic; color: #666; font-size: 0.9em; margin-top: 10px;">Note: You're using Simple mode. Switch to <a href="#" class="ai-mode-link" onclick="window.askAndrew.lastFocusedElement = document.activeElement; window.askAndrew.showAiModeModal(); return false;" role="button" tabindex="0">AI mode</a> for more detailed explanations.</p>`;
            
            // Add message with typing animation
            const messageDiv = this.addMessage('assistant', '', true);
            const messageText = messageDiv.querySelector('.message-text');
            await this.animateTyping(messageText, formattedResponse);
            return;
        }
        
        console.log('Simple mode - returning full content for', documents.length, 'documents');
        
        // Build response from full document content (no summarization)
        let response = "";
        
        documents.forEach((doc, index) => {
            response += doc.content;
            
            // Add spacing between multiple documents
            if (index < documents.length - 1) {
                response += "\n\n";
            }
        });
        
        // Build the formatted response HTML directly (don't use formatResponse which escapes HTML)
        let formattedResponse = `<p>${this.escapeHtml(response).replace(/\n/g, '<br>')}</p>`;
        
        // Add learn more links
        if (links && links.length > 0 && categories && categories.length > 0) {
            const linkHtml = links.map((link, index) => {
                const categoryName = categories[Math.min(index, categories.length - 1)];
                return `<a href="${link}" target="_blank" rel="noopener noreferrer">${categoryName}</a>`;
            }).join(' • ');
            formattedResponse += `<hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;"><p><strong>Learn more:</strong> ${linkHtml}</p>`;
        }
        
        // Add note about AI mode
        formattedResponse += `<p style="font-style: italic; color: #666; font-size: 0.9em; margin-top: 10px;">Note: You're using Simple mode. Switch to <a href="#" class="ai-mode-link" onclick="window.askAndrew.lastFocusedElement = document.activeElement; window.askAndrew.showAiModeModal(); return false;" role="button" tabindex="0">AI mode</a> for more detailed explanations.</p>`;
        
        // Add message with typing animation
        const messageDiv = this.addMessage('assistant', '', true);
        const messageText = messageDiv.querySelector('.message-text');
        await this.animateTyping(messageText, formattedResponse);
        
        // Update search status
        if (categories.length > 0) {
            this.elements.searchStatus.textContent = `🔍 Found in: ${categories.join(', ')}`;
            setTimeout(() => {
                this.elements.searchStatus.textContent = '';
            }, 3000);
        }
    }

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    restartConversation() {
        if (confirm('Are you sure you want to start a new conversation? This will clear the chat history.')) {
            // Clear conversation history
            this.conversationHistory = [];
            
            // Clear chat messages (keep welcome message)
            const messages = this.elements.chatMessages.querySelectorAll('.message:not(.welcome-message)');
            messages.forEach(msg => msg.remove());
            
            // Clear search status
            this.elements.searchStatus.textContent = '';
            
            console.log('Conversation restarted');
        }
    }
    
    showAiModeModal() {
        this.elements.aiModeModal.style.display = 'flex';
        this.currentModal = this.elements.aiModeModal;
        // Announce modal to screen readers
        this.elements.aiModeModal.setAttribute('aria-hidden', 'false');
        // Set focus to close button
        setTimeout(() => {
            this.elements.modalClose.focus();
            this.setupModalFocusTrap(this.elements.aiModeModal);
        }, 100);
    }
    
    hideAiModeModal() {
        this.elements.aiModeModal.style.display = 'none';
        this.elements.aiModeModal.setAttribute('aria-hidden', 'true');
        this.removeModalFocusTrap();
        this.currentModal = null;
        // Restore focus to the element that opened the modal
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
