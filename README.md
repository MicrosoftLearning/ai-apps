# Microsoft Learning AI Apps

This repository contains source code and published web apps for educational use. The apps are designed to support training modules on [Microsoft Learn](https://learn.microsoft.com) and are <u>not</u> intended (or supported) for use in production solutions. They are not supported Microsoft services or products, and are provided as-is without warranty of any kind.

All of the apps are designed to run locally in-browser. No data is uploaded to Microsoft, though some apps make use of external web services such as Wikipedia to return query results based on user input. To run the apps successfully, you need a modern browser, such as Microsoft Edge. In some cases, the full app functionality is only available on computers that include a GPU (integrated or dedicated). When using Windows on ARM64 computers, you may need to enable WebGPU in your browser flag settings (for example at edge://flags or chrome://flags). The GPU-based apps are designed to use a "fallback" mode with some functionality restrictions when no GPU is available.

## Apps

- [AI Chat Playground](./chat-playground/)
- [Information Extractor](./info-extractor/)
- [Text Analyzer](./text-analyzer/)
- [Python ScriptBook](./scriptbook/)
- [ML Lite](./ml-lite/)
- [ML Lab](./ml-lab/)

## Transparency Notes

The AI functionality in these apps was developed with [Microsoft's principles for responsible AI](https://www.microsoft.com/en-us/ai/principles-and-approach) in mind. Models and prompts have been chosen to minimize the risk of harmful content generation, and ongoing automated code quality reviews are in place to mitigate potential abuse or accidental security issues. If you do encounter an issue, we encourage you to report it at [https://github.com/MicrosoftLearning/ai-apps/issues](https://github.com/MicrosoftLearning/ai-apps/issues).

### Data privacy

The apps, including AI models, run in your local browser and no data is shared with Microsoft. No data from your browser, such as cookies or configuration data, is collected by any of these apps.

In the **Chat Playground** app, depending on the app mode configuration, input to models (i.e. prompts) may be sent to third-party APIs. Specifically:

- When <u>not</u> using a generative AI model, keywords from prompts are sent to the [Wikipedia API](https://en.wikipedia.org/w/api.php) to retrieve relevant information from Wikipedia. Only text you explicitly enter into the chat API is sent to Wikpedia.
- When Speech-to-text is enabled, speech input is processed by the browser's native [Web Speech API](https://webaudio.github.io/web-speech-api/) implementation, which may send audio to a server for processing. Speech input must be explicitly enabled the first time you use it in the app. Only your audio spoken when the microphone is active (with audible and visual indicators) is processed.

### Generative AI

The following apps use the [Microsoft Phi-3-mini-4k-instruct](https://azure.microsoft.com/products/phi/) generative AI model (specifically *Microsoft Phi-3-mini-4k-instruct-q4f16_1-MLC*). No additional training or fine-tuning has been performed on the model. You can view the [model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) for this model for details, including considerations for responsible use.

The model is run in-browser using the [WebLLM](https://webllm.mlc.ai/) JavaScript module, with no server-side processing. Some PC and browser combinations may not support the underlying WebGPU framework on which WebLLM depends, in which case generative AI functionality is not used by the apps.

> **IMPORTANT**: Generative AI functionality in these apps is designed exclusively for *educational* use. Do <u>not</u> rely on the output from these apps for any real-world application, decision, or action.

#### Chat Playground

Chat Playground provides the option to use generative AI or a fallback mode that does not use a generative AI model (which you can explicitly enable by selecting "None" in the model list). The default system prompt is `You are an AI assistant that helps people find information`. Additionally, the app makes the following augmentations to prompts:

- When the user uploads a data file, the app appends `Use this data to answer questions:{text-from-uploaded-file}` to the system prompt.
- When the user enables text-to-speech, the app appends `Important: Always answer with a single, concise sentence.` to the system prompt.
- When the user enables image analysis and uploads an image, the app appends `({image-class-prediction})` to the user prompt (where the image class prediction is the text label predicted for the image by the MobileNetV3 image classification model).

#### Text Analyzer

Text Analyzer provides the option to use generative AI, or a fallback mode that uses statistical text analysis techniques (which you can explicitly activate by disabling the *Use Generative AI* toggle). The prompts used by the app depend on the specific analysis being performed:

- **Sentiment analysis**:
    - System prompt: `You are a sentiment analysis expert. Respond only with valid JSON`
    - User prompt: `{User-entered or uploaded text}`
- **Language detection**:
    - System prompt: `You are a language detection expert. Respond only with valid JSON.`
    - User prompt:

    ```
    Detect the language of the following text. Pay special attention to 
        non-Latin scripts like Chinese, Japanese, Korean, Arabic, etc. Respond in this exact JSON format:
    {
      "language": "full language name",
      "code": "ISO 639-1 two-letter code",
      "confidence": a number between 0.0 and 1.0
    }
    
    Examples:
    - Chinese text should return: {"language": "Chinese", "code": "zh", "confidence": 0.95}
    - Japanese text should return: {"language": "Japanese", "code": "ja", "confidence": 0.95}
    - English text should return: {"language": "English", "code": "en", "confidence": 0.95}
    
    Text to analyze:
    "{first 500 characters of the user-entered or uploaded text}"
    
    Respond only with the JSON object, no other text.
    ```

- **Key-phrase extraction**:
    - System prompt: `You are a key phrase extraction expert. Respond only with a valid JSON array of strings.`
    - User prompt:

    ```
    Extract 5-10 key phrases from the text below. Respond with a JSON array without any markdown formatting or code blocks:
    ["phrase 1", "phrase 2", "phrase 3"]
    
    Text: "{first 1000 characters of the user-entered or uploaded text}"
    
    Respond only with the JSON array, no markdown, no code blocks, no other text.
    ```

- **Entity recognition**
    - System prompt: `You are a named entity recognition expert. Respond only with a valid JSON array.`
    - User prompt:

    ```
    Extract named entities and categorize them. Respond with a JSON array without any markdown formatting or code blocks:
    [{"type": "Person", "value": "name"}, {"type": "Place", "value": "location"}]
    
    Types: Person, Place, Organization, Date, Money, Email, Phone, Product, Event
    
    Text: "{first 1000 characters of the user-entered or uploaded text}"
    
    Respond only with the JSON array, no markdown, no code blocks, no other text.
    ```

- **Text summarization**:
    - System prompt: `You are an expert text summarizer. Provide concise, accurate summaries.`
    - User prompt:

    ```
    Summarize the following text in 2-4 sentences, capturing the main points:
    
    "{first 2000 characters of the user-entered or uploaded text}"
    
    Provide only the summary, no other text.
    ```

#### Information Extractor

Information Extractor provides the option to use generative AI, or a fallback mode that uses heuristic-based field mapping techniques (which you can explicitly activate by disabling the *Use Generative AI* toggle)

The app uses the following prompts:

- System prompt: `You are a helpful assistant that extracts structured information from receipt text. Always respond with a clear list of field names and their values.`
- User prompt:
    ```
    The following text was extracted from a scanned receipt:
    ---
    {OCR text extracted from uploaded image}
    ---
    Please identify the most likely values for these fields:
    - Vendor
    - Vendor-Address
    - Vendor-Phone
    - Receipt-Date
    - Receipt-Time
    - Total-spent
    
    Date fields should be formatted as mm/dd/yyyy
    
    Respond as a list of fields with their values.
    ```

### Other AI models and technologies

In addition to WebLLM and the Microsoft Phi model described above for generative AI, the apps make use of the following models and technologies under the terms of their respective licenses:

- [MobileNet-V3](https://huggingface.co/docs/timm/en/models/mobilenet-v3) running in [Tensorflow.js](https://www.npmjs.com/package/@tensorflow/tfjs) used by Chat Playground to predict image classifications.
- [Wikipedia API](https://en.wikipedia.org/w/api.php) used by Chat Playground to retrieve relevant information from Wikipedia.
- [Web Speech API](https://webaudio.github.io/web-speech-api/) used by Chat Playground for speech recognition and synthesis.
- [NLP.js](https://www.npmjs.com/package/@nlpjs/nlp) used by Text Analyzer for statistical text analysis.
- [Compromise.js](https://www.npmjs.com/package/compromise) used by Text Analyzer for statistical text analysis.
- [Tesseract.js](https://github.com/naptha/tesseract.js/blob/master/README.md) used by Information Extractor to perform OCR analysis.
- [PyScript](https://pyscript.net/) used by ML Lab, ML Lite, and ScriptBook to provide an in-browser Python runtime. Imported libraries include numpy, pandas, matplotLib, and scikit-learn.