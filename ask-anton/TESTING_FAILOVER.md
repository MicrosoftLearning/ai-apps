# Testing Failover to Basic Mode

This guide explains how to test the failover mechanisms in the Ask Anton app to ensure it properly falls back to Basic mode when WebGPU or Wllama initialization fails.

## Failover Flow

The app attempts to initialize in this order:

1. **GPU Mode** (WebLLM with Phi-3-mini) - requires WebGPU support
2. **CPU Mode** (Wllama with SmolLM2) - fallback if GPU fails
3. **Basic Mode** - fallback if both GPU and CPU fail

## How Debug Testing Works

The debug flags inject **actual errors during initialization** to test the try-catch error handling:

- `forceWebGPUFail=true`: Starts WebGPU initialization (shows progress), then throws an error to trigger the catch block
- `forceWllamaFail=true`: Starts Wllama initialization (shows progress), then throws an error to trigger the catch block
- `forceBasicMode=true`: Bypasses all model loading and goes directly to Basic mode

This ensures the **actual error handling flow** is tested, not just alternate code paths.

## Testing Methods

### Method 1: URL Parameters (Recommended)

Add debug parameters to the URL when loading the page:

#### Test GPU → CPU → Basic Failover

```
index.html?debug=true&forceWebGPUFail=true&forceWllamaFail=true
```

#### Test GPU → CPU Failover Only

```
index.html?debug=true&forceWebGPUFail=true
```

#### Test CPU → Basic Failover Only

```
index.html?debug=true&forceWllamaFail=true
```

#### Force Basic Mode Directly

```
index.html?debug=true&forceBasicMode=true
```

### Method 2: Browser Console (Dynamic Testing)

After the page loads, you can dynamically change debug flags in the console:

```javascript
// Enable debug mode
window.askAnton.debugConfig.enabled = true;

// Force WebGPU failure
window.askAnton.debugConfig.forceWebGPUFail = true;

// Force Wllama failure
window.askAnton.debugConfig.forceWllamaFail = true;

// Force Basic mode
window.askAnton.debugConfig.forceBasicMode = true;

// Then reload or switch modes using the dropdown
```

**Note:** For mode switching after initial load, use the mode selector dropdown in the UI.

### Method 3: Browser DevTools Override (Advanced)

Paste this in the console **before** the page loads (or in a userscript):

```javascript
// Disable WebGPU
Object.defineProperty(navigator, 'gpu', {
    get: () => undefined,
    configurable: true
});

// Intercept Wllama constructor
const OriginalWllama = window.Wllama;
window.Wllama = class extends OriginalWllama {
    constructor(...args) {
        super(...args);
        throw new Error('Forced Wllama failure for testing');
    }
};
```

### Method 4: Network Throttling

Test CPU mode failure by blocking the model download:

1. Open DevTools → Network tab
2. Enable "Offline" mode or block the pattern `*.gguf`
3. Try to load CPU mode
4. Should fall back to Basic mode

## What to Verify

### ✅ GPU → CPU Failover (Error Handling Test)

- [ ] Progress bar shows "Loading AI model (WebGPU)..." - **proves initialization was attempted**
- [ ] Console shows: "🧪 DEBUG: Forcing WebGPU initialization to fail (testing error handling)"
- [ ] Console shows error: "Failed to initialize WebLLM:" with debug error
- [ ] Console shows: "WebLLM initialization failed, falling back to CPU mode"
- [ ] Progress bar transitions to "Loading AI model (CPU mode)..."
- [ ] App loads successfully in CPU mode
- [ ] Mode selector shows 🟠 CPU as active
- [ ] GPU mode (🟢 GPU) is disabled in selector

### ✅ CPU → Basic Failover (Error Handling Test)

- [ ] Progress bar shows "Loading AI model (CPU mode)..." - **proves initialization was attempted**
- [ ] Console shows: "🧪 DEBUG: Forcing Wllama initialization to fail (testing error handling)"
- [ ] Console shows error: "Failed to initialize wllama:" with debug error
- [ ] Console shows: "CPU model initialization failed, falling back to Basic mode"
- [ ] Progress bar shows "Ready to chat! (Basic mode)"
- [ ] System message: "Using Basic mode because the GPU and CPU models could not be loaded."
- [ ] Mode selector shows ⚪ Basic as active
- [ ] CPU and GPU modes are disabled in selector

### ✅ Complete Failover (GPU → CPU → Basic - Full Error Chain)

- [ ] Progress bar shows "Loading AI model (WebGPU)..." first
- [ ] Then shows "Loading AI model (CPU mode)..."
- [ ] Both debug error messages appear in console
- [ ] Both initialization failure messages appear in sequence
- [ ] App successfully loads in Basic mode
- [ ] All three mode indicators show correct availability
- [ ] Chat functionality works (returns direct knowledge base content)

### ✅ Debug Mode Indicators

When debug=true is in URL:

- [ ] Console shows: "🧪 Debug mode enabled: {config}"
- [ ] Console shows helpful tips for forcing failures
- [ ] Forced failures show "🧪 DEBUG:" prefix in console
- [ ] System messages show debug prefix when applicable

## Testing Basic Mode Functionality

Once in Basic mode, verify it works correctly:

1. **Test valid query:**
   - Ask: "What is machine learning?"
   - Should return content from knowledge base
   - Should include link to more information

2. **Test invalid query:**
   - Ask: "Tell me about cooking"
   - Should return: "Sorry, I couldn't find any specific information on that topic..."

3. **Test content moderation:**
   - Try inappropriate content
   - Should be blocked appropriately

## Automated Test Scenarios

### Scenario 1: Test GPU Error Handling → CPU Fallback

```
index.html?debug=true&forceWebGPUFail=true
```

**Expected:**

- Attempts WebGPU initialization (shows progress)
- Error thrown during initialization
- Catch block triggers fallback to CPU mode
- CPU mode loads successfully

### Scenario 2: Test Complete Error Chain (GPU → CPU → Basic)

```
index.html?debug=true&forceWebGPUFail=true&forceWllamaFail=true
```

**Expected:**

- Attempts WebGPU initialization → fails with error
- Attempts Wllama initialization → fails with error  
- Falls back to Basic mode
- Both error handling paths are exercised

### Scenario 3: Direct Basic Mode (No Error Testing)

```
index.html?debug=true&forceBasicMode=true
```

**Expected:**

- Bypasses all model loading
- Goes straight to Basic mode
- Useful for testing Basic mode functionality without waiting

## Debug Console Commands

Useful commands for testing:

```javascript
// Check current mode
window.askAnton.currentMode

// Check available modes
window.askAnton.availableModes

// View debug config
window.askAnton.debugConfig

// Check if engines are initialized
window.askAnton.engine  // WebLLM
window.askAnton.wllama  // Wllama

// Force a mode switch (if available)
window.askAnton.switchMode('basic')
```

## Troubleshooting

**Debug mode not working?**

- Ensure URL has `?debug=true`
- Check console for "🧪 Debug mode enabled" message

**Failover not triggering?**

- Verify debug flags are set before initialization
- Try hard refresh (Ctrl+Shift+R)
- Clear cache and reload

**Mode selector not updating?**

- Check `availableModes` object in console
- Verify `updateModeSelector()` was called
- Check for JavaScript errors

## Notes

- Debug flags only work when `debug=true` is in the URL
- URL parameters are parsed during construction
- Console changes to `debugConfig` work for subsequent operations
- Debug messages use 🧪 emoji prefix for easy identification
