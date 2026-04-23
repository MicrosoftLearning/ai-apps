## About this app

This app is designed as a learning aid for anyone seeking to become familiar with Generative AI apps and agents. It's based on the user interface in the Microsoft Foundry portal, but does not use any Azure cloud services.

### AI Models

The app supports two AI language models:

- **Phi-3-mini (GPU mode)**: Microsoft's Phi-3-mini small language model, hosted via WebLLM. Requires a modern browser with WebGPU API support and a GPU (integrated or dedicated). Provides the best performance and response quality. Maintains full conversation history. Uses 2048 token context window.

- **Phi-2 (CPU mode)**: Microsoft's Phi-2 model running via wllama (WebAssembly). Runs entirely on CPU without requiring GPU support. Uses a reduced context window for browser reliability and maintains a short recent conversation history. Works on any device, but CPU responses are slower than GPU mode.

If WebGPU is unavailable, the app automatically falls back to CPU mode using Phi-2. You can also manually switch between models using the Model dropdown.

The app also uses the **MobileNetV3** model (via TensorFlow.js) for image classification, and the **Web Speech API** for speech recognition and synthesis.

### Security

All user input is sanitized using HTML escaping to prevent injection attacks. Uploaded files are validated for type and size before processing.

### Known issues

- The initial download of the Phi-3 model may take a few minutes - particularly on low-bandwidth connections. Subsequent downloads should be quicker.
- Some GPU-enabled computers (particularly those with ARM-based processors) do not support WebGPU without enabling the **Unsafe WebGPU Support** browser flag. If your browser fails to load the Phi-3-mini model, you can enable this flag at edge://flags on Microsoft Edge or chrome://flags on Google Chrome. Disable it again when finished! Alternatively, switch to CPU mode (Phi-2).
- Microsoft Edge on ARM-based computers does not support Web Speech for speech recognition (speech to text), and returns a network error when attempting to capture input from the mic. Speech synthesis (text to speech) should still work.
- **Phi-2 (CPU mode)** runs through WebAssembly on the CPU and can be noticeably slower than GPU mode. It uses a smaller effective context window in this app to stay reliable in the browser, so long chats may retain less prior context than GPU mode.
