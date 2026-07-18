## About this app

This app is designed as a learning aid for anyone seeking to become familiar with Generative AI apps and agents. It's based on the user interface in the Microsoft Foundry portal, but does not use any Azure cloud services.

### AI Model

The app uses the **Microsoft Phi-3.5-mini** small language model, hosted via **wllama** (WebAssembly). Where available, it uses WebGPU to run the model on your computer's GPU; reverting to CPU when necessary. It Uses a reduced context window for browser reliability and maintains a short recent conversation history. Works on any device, but responses may be slow when using older or lower-spec hardware.

If your hardware does not meet the minimum required spec, or if wllama fails; A fallback mode with no model is used. You can also manually switch between modes using the Model dropdown.

The app also uses the **MobileNetV2** model (via TensorFlow.js) for image classification, and the **Web Speech API** for speech recognition and synthesis.

### Security

All user input is sanitized using HTML escaping to prevent injection attacks. Uploaded files are validated for type and size before processing.

### Known issues

- The initial download of the Phi model may take a few minutes - particularly on low-bandwidth connections. Subsequent downloads should be quicker.
- Microsoft Edge on ARM-based computers does not support Web Speech for speech recognition (speech to text), and returns a network error when attempting to capture input from the mic. Speech synthesis (text to speech) should still work.
