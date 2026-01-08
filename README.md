# Microsoft Learning AI Apps

This repository contains source code and published web apps for educational use. The apps are designed to support training modules on [Microsoft Learn](https://learn.microsoft.com) and are <u>not</u> intended (or supported) for use in production solutions. They are not supported Microsoft services or products, and are provided as-is without warranty of any kind.

All of the apps are designed to run locally in-browser. No data is uploaded to Microsoft, though some apps make use of external web services such as Wikipedia to return query results based on user input. To run the apps successfully, you need a modern browser, such as Microsoft Edge. In some cases, the full app functionality is only available on computers that include a GPU (integrated or dedicated). When using Windows on ARM64 computers, you may need to enable WebGPU in your browser flag settings (for example at [edge://flags](edge://flags) or [chrome://flags](chrome://flags)). The GPU-based apps are designed to use a "fallback" mode with some functionality restrictions when no GPU is available.

## Apps

- [Ask Andrew (sample AI agent)](./ask-andrew/)
- [Chat Playground](./chat-playground/)
- [Speech Playground](./speech-playground/)
- [Information Extractor](./info-extractor/)
- [Text Analyzer](./text-analyzer/)
- [Python ScriptBook](./scriptbook/)
- [ML Lite](./ml-lite/)
- [ML Lab](./ml-lab/)

## Transparency Notes

The AI functionality in these apps was developed with [Microsoft's principles for responsible AI](https://www.microsoft.com/en-us/ai/principles-and-approach) in mind. Models and prompts have been chosen to minimize the risk of harmful content generation, and ongoing automated code quality reviews are in place to mitigate potential abuse or accidental security issues. If you do encounter an issue, we encourage you to report it at [https://github.com/MicrosoftLearning/ai-apps/issues](https://github.com/MicrosoftLearning/ai-apps/issues).

### Data privacy

The apps, including AI models, run in your local browser and no data is shared with Microsoft. No data from your browser, such as cookies or configuration data, is collected by any of these apps.

In some cases, depending on the app mode configuration, input to models (i.e. prompts) may be sent to third-party APIs. Specifically:

- In **Chat Playground** and **Speech Playground**, when <u>not</u> using a generative AI model, keywords from prompts are sent to the [Wikipedia API](https://en.wikipedia.org/w/api.php) to retrieve relevant information from Wikipedia. Only text you explicitly enter into the chat API is sent to Wikipedia.
- In **Speech Playground**, speech input is processed by the browser's native [Web Speech API](https://webaudio.github.io/web-speech-api/) implementation, which may send audio to a server for processing. Speech input must be explicitly enabled the first time you use it in the app. Only your audio spoken when the microphone is active is processed.

### Generative AI

The following apps use the [Microsoft Phi-3-mini-4k-instruct](https://azure.microsoft.com/products/phi/) generative AI model (specifically *Microsoft Phi-3-mini-4k-instruct-q4f16_1-MLC*). No additional training or fine-tuning has been performed on the model. You can view the [model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) for this model for details, including considerations for responsible use.

The model is run in-browser using the [WebLLM](https://webllm.mlc.ai/) JavaScript module, with no server-side processing. Some PC and browser combinations may not support the underlying WebGPU framework on which WebLLM depends, in which case generative AI functionality is not used by the apps.

> **IMPORTANT**: Generative AI functionality in these apps is designed exclusively for *educational* use. Do <u>not</u> rely on the output from these apps for any real-world application, decision, or action.

#### Ask Andrew (sample AI agent)

Ask Andrew provides the option to use generative AI or a fallback mode that does not use a generative AI model (which you can explicitly enable by selecting "Simple Mode"). When using generative AI, the system prompt is:

```
You are Andrew, a knowledgeable and friendly AI learning assistant who helps students understand AI concepts.

IMPORTANT: Follow these guidelines when responding:
- Explain concepts clearly and concisely in a single paragraph based only on the provided context.
- Keep responses short and focused on the question, with no headings.
- Use examples and analogies when helpful.
- Use simple language suitable for learners in a conversational, friendly tone.
- Provide a general descriptions and overviews, but do NOT provide explicit steps or instructions for developing AI solutions.
- If the context includes "Sorry, I couldn't find any specific information on that topic. Please try rephrasing your question or explore other AI concepts.", use that exact phrasing and no additional information.
- Do not start responses with "A:" or "Q:".
- Keep your responses concise and to the the point.
- Do NOT provide links for more information (these will be added automatically later).
```

The index searched for context is in the [index.json](./ask-andrew/index.json) file and the search is performed locally in-browser. No prompt text or any other data is sent outside of the browser.

#### Chat Playground

Chat Playground provides the option to use generative AI or a fallback mode that does not use a generative AI model (which you can explicitly enable by selecting "None" in the model list). The default system prompt is `You are an AI assistant that helps people find information`. Additionally, the app makes the following augmentations to prompts:

- When the user uploads a data file, the app appends `Use this data to answer questions:{text-from-uploaded-file}` to the system prompt.
- When the user enables image analysis and uploads an image, the app appends `({image-class-prediction})` to the user prompt (where the image class prediction is the text label predicted for the image by the MobileNetV3 image classification model).

#### Speech Playground

Speech Playground provides the option to use generative AI or a fallback mode that does not use a generative AI model (which you can explicitly enable by selecting "None" in the model list). The default system prompt is `You are a helpful AI assistant that answers spoken questions with vocalized responses.` to which the additional instruction `IMPORTANT: Make your responses brief and to the point.` is appended. The user may change the system prompt in the UI, but the additional instruction to keep responses short is always appended.

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
- [Wikipedia API](https://en.wikipedia.org/w/api.php) used by Chat Playground and Speech Playground to retrieve relevant information from Wikipedia.
- [Web Speech API](https://webaudio.github.io/web-speech-api/) used by Chat Playground and Speech Playground for speech recognition and synthesis.
- [retext-keywords](https://github.com/retextjs/retext-keywords) used by Text Analyzer to extract key words and phrases.
- [Compromise.js](https://www.npmjs.com/package/compromise) used by Text Analyzer to support named entity recognition.
- [TextRank.js](https://www.jsdelivr.com/package/npm/textrank) used by Text Analyzer for text summarization.
- [Tesseract.js](https://github.com/naptha/tesseract.js/blob/master/README.md) used by Information Extractor to perform OCR analysis.
- [PyScript](https://pyscript.net/) used by ML Lab, ML Lite, and ScriptBook to provide an in-browser Python runtime. Imported libraries include numpy, pandas, matplotLib, and scikit-learn.
