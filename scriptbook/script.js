// Cell counter for unique IDs
let cellCounter = 0;
let uploadedFileContent = null;
let uploadedFileName = null;
let pyScriptReady = false;

// Track running cells and their script tags
const runningCells = new Map();

// Wait for PyScript to be ready
window.addEventListener('py:ready', () => {
    pyScriptReady = true;
    console.log('PyScript is ready');
});

// Initialize cell state based on its type
function initializeCellState(cellId) {
    const cell = document.getElementById(cellId);
    const cellType = cell.querySelector('.cell-type-selector').value;
    const runBtn = cell.querySelector('.btn-run');
    const stopBtn = cell.querySelector('.btn-stop');
    
    // Add toggle button event listener if not already added
    const toggleButton = cell.querySelector('.btn-toggle');
    if (toggleButton && !toggleButton.onclick) {
        toggleButton.onclick = (e) => {
            e.stopPropagation();
            toggleEditMode(cellId);
        };
    }
    
    if (cellType === 'markdown') {
        // Hide run/stop buttons for markdown cells
        runBtn.style.display = 'none';
        stopBtn.style.display = 'none';
    } else {
        // Show run button for code cells
        runBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
    }
    
    // Update tab order
    updateTabOrder();
}

// Initialize the notebook with one cell
document.addEventListener('DOMContentLoaded', () => {
    addCell();
    
    // Event listeners for toolbar buttons
    document.getElementById('runAllBtn').addEventListener('click', runAllCells);
    document.getElementById('clearAllBtn').addEventListener('click', clearAllCells);
    document.getElementById('fileUpload').addEventListener('change', handleFileUpload);
    document.getElementById('saveBtn').addEventListener('click', saveNotebook);
    document.getElementById('loadFile').addEventListener('change', loadNotebook);
    
    // Set initial tab order
    updateTabOrder();
});

// Add a new cell to the notebook
function addCell() {
    const cellId = `cell-${cellCounter++}`;
    const notebook = document.getElementById('notebook');
    
    const cellDiv = document.createElement('div');
    cellDiv.className = 'cell';
    cellDiv.id = cellId;
    cellDiv.setAttribute('role', 'article');
    cellDiv.setAttribute('aria-label', `Notebook cell ${cellCounter}`);
    
    cellDiv.innerHTML = `
        <div class="cell-header" role="toolbar" aria-label="Cell controls">
            <select class="cell-type-selector" aria-label="Select cell type" tabindex="0">
                <option value="python">Python</option>
                <option value="markdown">Markdown</option>
            </select>
            <button class="btn-run" onclick="runCell('${cellId}')" aria-label="Run this cell" title="Run this cell" tabindex="0">▶ Run</button>
            <button class="btn-stop" onclick="stopCell('${cellId}')" aria-label="Stop cell execution" title="Stop cell execution" style="display: none;" tabindex="0">⏹ Stop</button>
            <button class="btn-toggle" aria-label="Switch to view mode" title="Switch to view mode" tabindex="0"><span style="filter: brightness(0) invert(1); font-weight: bold;">👁</span></button>
            <button class="btn-delete" onclick="deleteCell('${cellId}')" aria-label="Delete this cell" title="Delete this cell" tabindex="0">✖</button>
        </div>
        <textarea class="cell-input" placeholder="Enter Python code or Markdown..." aria-label="Cell input code or text" tabindex="0" data-edit-mode="true"># Write your code here</textarea>
        <div class="cell-output" role="region" aria-label="Cell output" aria-live="polite"></div>
    `;
    
    // Add Tab key handler and edit mode handlers
    const textarea = cellDiv.querySelector('.cell-input');
    textarea.addEventListener('keydown', (e) => handleCellKeydown(e, cellId));
    
    // Add focus handler to select default text on first click
    textarea.addEventListener('focus', handleFirstFocus);
    
    // Add click handler to switch to edit mode
    cellDiv.addEventListener('click', (e) => handleCellClick(e, cellId));
    
    // Add cell type change handler
    const cellTypeSelector = cellDiv.querySelector('.cell-type-selector');
    cellTypeSelector.addEventListener('change', (e) => handleCellTypeChange(e, cellId));
    
    // Add hover zone before the cell
    const hoverZoneBefore = document.createElement('div');
    hoverZoneBefore.className = 'cell-hover-zone';
    hoverZoneBefore.innerHTML = '<button class="add-cell-button" tabindex="0" aria-label="Add cell before this cell">+ Add Cell</button>';
    const addButton = hoverZoneBefore.querySelector('.add-cell-button');
    addButton.onclick = () => {
        insertCellBefore(cellId);
    };
    addButton.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            insertCellBefore(cellId);
        }
    };
    
    notebook.appendChild(hoverZoneBefore);
    notebook.appendChild(cellDiv);
    
    // Add toggle button event listener
    const toggleButton = cellDiv.querySelector('.btn-toggle');
    if (toggleButton) {
        toggleButton.onclick = (e) => {
            e.stopPropagation();
            toggleEditMode(cellId);
        };
    }
    
    // Initialize cell state after adding to DOM
    initializeCellState(cellId);
    
    // Update hover zones
    updateHoverZones();
    
    // Update tab order for all elements
    updateTabOrder();
}

// Insert a cell before a specific cell
function insertCellBefore(beforeCellId) {
    const newCellId = `cell-${cellCounter++}`;
    const notebook = document.getElementById('notebook');
    const beforeCell = document.getElementById(beforeCellId);
    
    const cellDiv = document.createElement('div');
    cellDiv.className = 'cell';
    cellDiv.id = newCellId;
    cellDiv.setAttribute('role', 'article');
    cellDiv.setAttribute('aria-label', `Notebook cell ${cellCounter}`);
    
    cellDiv.innerHTML = `
        <div class="cell-header" role="toolbar" aria-label="Cell controls">
            <select class="cell-type-selector" aria-label="Select cell type" tabindex="0">
                <option value="python">Python</option>
                <option value="markdown">Markdown</option>
            </select>
            <button class="btn-run" onclick="runCell('${newCellId}')" aria-label="Run this cell" title="Run this cell" tabindex="0">▶ Run</button>
            <button class="btn-stop" onclick="stopCell('${newCellId}')" aria-label="Stop cell execution" title="Stop cell execution" style="display: none;" tabindex="0">⏹ Stop</button>
            <button class="btn-toggle" aria-label="Switch to view mode" title="Switch to view mode" tabindex="0"><span style="filter: brightness(0) invert(1); font-weight: bold;">👁</span></button>
            <button class="btn-delete" onclick="deleteCell('${newCellId}')" aria-label="Delete this cell" title="Delete this cell" tabindex="0">✖</button>
        </div>
        <textarea class="cell-input" placeholder="Enter Python code or Markdown..." aria-label="Cell input code or text" tabindex="0" data-edit-mode="true"># Write your code here</textarea>
        <div class="cell-output" role="region" aria-label="Cell output" aria-live="polite"></div>
    `;
    
    // Add Tab key handler and edit mode handlers
    const textarea = cellDiv.querySelector('.cell-input');
    textarea.addEventListener('keydown', (e) => handleCellKeydown(e, newCellId));
    
    // Add focus handler to select default text on first click
    textarea.addEventListener('focus', handleFirstFocus);
    
    // Add click handler to switch to edit mode
    cellDiv.addEventListener('click', (e) => handleCellClick(e, newCellId));
    
    // Add cell type change handler
    const cellTypeSelector = cellDiv.querySelector('.cell-type-selector');
    cellTypeSelector.addEventListener('change', (e) => handleCellTypeChange(e, newCellId));
    
    // Find the hover zone before the target cell
    const hoverZone = beforeCell.previousElementSibling;
    
    // Create new hover zone for the inserted cell
    const newHoverZone = document.createElement('div');
    newHoverZone.className = 'cell-hover-zone';
    newHoverZone.innerHTML = '<button class="add-cell-button" tabindex="0" aria-label="Add cell before this cell">+ Add Cell</button>';
    const addButton = newHoverZone.querySelector('.add-cell-button');
    addButton.onclick = () => {
        insertCellBefore(newCellId);
    };
    addButton.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            insertCellBefore(newCellId);
        }
    };
    
    // Insert before the hover zone
    notebook.insertBefore(newHoverZone, hoverZone);
    notebook.insertBefore(cellDiv, hoverZone);
    
    // Add toggle button event listener
    const toggleButton = cellDiv.querySelector('.btn-toggle');
    if (toggleButton) {
        toggleButton.onclick = (e) => {
            e.stopPropagation();
            toggleEditMode(newCellId);
        };
    }
    
    // Initialize cell state after adding to DOM
    initializeCellState(newCellId);
    
    updateHoverZones();
    updateTabOrder();
}

// Update hover zones (add one at the end)
function updateHoverZones() {
    const notebook = document.getElementById('notebook');
    
    // Remove existing end hover zone
    const existingEndZone = notebook.querySelector('.cell-hover-zone-end');
    if (existingEndZone) {
        existingEndZone.remove();
    }
    
    // Add hover zone at the end
    const hoverZoneEnd = document.createElement('div');
    hoverZoneEnd.className = 'cell-hover-zone cell-hover-zone-end';
    hoverZoneEnd.innerHTML = '<button class="add-cell-button" tabindex="0" aria-label="Add cell at end">+ Add Cell</button>';
    const addButton = hoverZoneEnd.querySelector('.add-cell-button');
    addButton.onclick = addCell;
    addButton.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            addCell();
        }
    };
    
    notebook.appendChild(hoverZoneEnd);
}

// Handle keyboard events in cell textareas
function handleCellKeydown(e, cellId) {
    const textarea = e.target;
    const cell = document.getElementById(cellId);
    const cellType = cell.querySelector('.cell-type-selector').value;
    const isEditMode = textarea.getAttribute('data-edit-mode') === 'true';
    
    if (e.key === 'Tab') {
        if (isEditMode) {
            // In edit mode, insert tab character
            e.preventDefault();
            
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            
            // Insert 4 spaces at cursor position
            const spaces = '    ';
            textarea.value = textarea.value.substring(0, start) + spaces + textarea.value.substring(end);
            
            // Move cursor after the inserted spaces
            textarea.selectionStart = textarea.selectionEnd = start + spaces.length;
        }
        // In view mode, let Tab work for navigation (don't preventDefault)
    } else if (e.key === 'Enter' && !e.shiftKey) {
        // ENTER key activates edit mode (when in view mode)
        if (!isEditMode) {
            e.preventDefault();
            toggleEditMode(cellId);
        }
        // When in edit mode, let Enter work normally
    } else if (e.key === 'Escape') {
        // ESC key deactivates edit mode
        if (isEditMode) {
            e.preventDefault();
            toggleEditMode(cellId);
            textarea.blur(); // Remove focus from textarea
        }
        // For markdown cells, ESC doesn't do anything special
    }
}

// Select default text on first focus
function handleFirstFocus(event) {
    const textarea = event.target;
    const defaultText = '# Write your code here';
    
    // Check if the textarea contains only the default text
    if (textarea.value === defaultText) {
        textarea.select();
        // Remove this event listener after first use
        textarea.removeEventListener('focus', handleFirstFocus);
    }
}

// Handle cell click to switch to edit mode
function handleCellClick(event, cellId) {
    const cell = document.getElementById(cellId);
    const textarea = cell.querySelector('.cell-input');
    const cellType = cell.querySelector('.cell-type-selector').value;
    const target = event.target;
    
    // Don't interfere if clicking on buttons or other controls
    if (target.tagName === 'BUTTON' || target.tagName === 'SELECT' || 
        target.classList.contains('cell-header') || 
        target.closest('.cell-header')) {
        return;
    }
    
    // Switch to edit mode if not already in edit mode
    const isEditMode = textarea.getAttribute('data-edit-mode') === 'true';
    if (!isEditMode) {
        toggleEditMode(cellId);
    }
    
    // Focus the textarea if it's editable
    if (!textarea.hasAttribute('readonly')) {
        textarea.focus();
    }
}

// Handle cell type change to update button state
function handleCellTypeChange(event, cellId) {
    const cell = document.getElementById(cellId);
    const textarea = cell.querySelector('.cell-input');
    const toggleBtn = cell.querySelector('.btn-toggle');
    const runBtn = cell.querySelector('.btn-run');
    const stopBtn = cell.querySelector('.btn-stop');
    const newCellType = event.target.value;
    
    if (newCellType === 'markdown') {
        // For markdown cells, start in edit mode like code cells
        textarea.classList.remove('view-mode', 'collapsed');
        textarea.removeAttribute('readonly');
        textarea.setAttribute('data-edit-mode', 'true');
        toggleBtn.innerHTML = '<span style="filter: brightness(0) invert(1); font-weight: bold;">👁</span>';
        toggleBtn.setAttribute('aria-label', 'Switch to view mode');
        toggleBtn.setAttribute('title', 'Switch to view mode');
        
        // Hide run/stop buttons for markdown cells
        runBtn.style.display = 'none';
        stopBtn.style.display = 'none';
    } else {
        // For code cells, start in edit mode
        textarea.classList.remove('view-mode', 'collapsed');
        textarea.removeAttribute('readonly');
        textarea.setAttribute('data-edit-mode', 'true');
        toggleBtn.innerHTML = '<span style="filter: brightness(0) invert(1); font-weight: bold;">👁</span>';
        toggleBtn.setAttribute('aria-label', 'Switch to view mode');
        toggleBtn.setAttribute('title', 'Switch to view mode');
        
        // Show run button for code cells
        runBtn.style.display = 'inline-block';
        // stopBtn stays hidden until needed
    }
    
    // Clear any existing output
    const output = cell.querySelector('.cell-output');
    output.innerHTML = '';
    output.classList.remove('markdown-output');
    
    // Update tab order
    updateTabOrder();
}

// Stop a running cell
function stopCell(cellId) {
    const cell = document.getElementById(cellId);
    const output = cell.querySelector('.cell-output');
    const runBtn = cell.querySelector('.btn-run');
    const stopBtn = cell.querySelector('.btn-stop');
    
    // Get the running cell info
    const runningInfo = runningCells.get(cellId);
    
    if (runningInfo) {
        // Clear the timeout
        if (runningInfo.timeoutId) {
            clearTimeout(runningInfo.timeoutId);
        }
        
        // Remove the script tag to stop execution
        if (runningInfo.scriptTag && runningInfo.scriptTag.parentNode) {
            runningInfo.scriptTag.parentNode.removeChild(runningInfo.scriptTag);
        }
        
        // Update output to show it was stopped
        output.innerHTML = '<div class="error">Execution stopped by user</div>';
        
        // Clean up tracking
        runningCells.delete(cellId);
    }
    
    // Restore buttons
    runBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
}

// Toggle code pane visibility
// Toggle edit mode for cells
function toggleEditMode(cellId) {
    const cell = document.getElementById(cellId);
    const input = cell.querySelector('.cell-input');
    const toggleBtn = cell.querySelector('.btn-toggle');
    const output = cell.querySelector('.cell-output');
    const cellType = cell.querySelector('.cell-type-selector').value;
    
    const isEditMode = input.getAttribute('data-edit-mode') === 'true';
    const hadFocus = document.activeElement === input;
    
    if (isEditMode) {
        // Switch to view mode
        input.setAttribute('data-edit-mode', 'false');
        toggleBtn.innerHTML = '<span style="filter: brightness(0) invert(1);">✏️</span>';
        toggleBtn.setAttribute('aria-label', 'Switch to edit mode');
        toggleBtn.setAttribute('title', 'Switch to edit mode');
        
        if (cellType === 'markdown') {
            // For markdown cells in view mode: hide source, show rendered output
            input.style.display = 'none';
            const code = input.value;
            if (code.trim()) {
                output.innerHTML = renderMarkdown(code);
                output.classList.add('markdown-output');
                output.style.display = 'block';
            } else {
                output.innerHTML = '';
                output.classList.remove('markdown-output');
                output.style.display = 'none';
            }
        } else {
            // For code cells in view mode: show source as read-only, keep output visible
            input.setAttribute('readonly', 'readonly');
            input.classList.add('view-mode');
            input.style.display = 'block';
            // Don't hide output for code cells - it should always be visible if present
            
            // Retain focus if cell had focus before
            if (hadFocus) {
                input.focus();
            }
        }
    } else {
        // Switch to edit mode
        input.setAttribute('data-edit-mode', 'true');
        toggleBtn.innerHTML = '<span style="filter: brightness(0) invert(1); font-weight: bold;">👁</span>';
        toggleBtn.setAttribute('aria-label', 'Switch to view mode');
        toggleBtn.setAttribute('title', 'Switch to view mode');
        
        if (cellType === 'markdown') {
            // For markdown cells in edit mode: show source, hide rendered output
            input.removeAttribute('readonly');
            input.classList.remove('view-mode');
            input.style.display = 'block';
            output.innerHTML = '';
            output.classList.remove('markdown-output');
            output.style.display = 'none';
        } else {
            // For code cells in edit mode: show source as editable, keep output visible
            input.removeAttribute('readonly');
            input.classList.remove('view-mode');
            input.style.display = 'block';
            // Don't hide output for code cells - it should always be visible if present
        }
        
        // Always focus when entering edit mode
        input.focus();
    }
    
    // Update tab order since textarea properties changed
    updateTabOrder();
}

// Delete a cell
function deleteCell(cellId) {
    const cell = document.getElementById(cellId);
    if (cell) {
        // Also remove the hover zone before this cell
        const hoverZone = cell.previousElementSibling;
        if (hoverZone && hoverZone.classList.contains('cell-hover-zone')) {
            hoverZone.remove();
        }
        cell.remove();
        updateHoverZones();
        updateTabOrder();
    }
}

// Run a specific cell
async function runCell(cellId) {
    const cell = document.getElementById(cellId);
    const input = cell.querySelector('.cell-input');
    const output = cell.querySelector('.cell-output');
    const runBtn = cell.querySelector('.btn-run');
    const stopBtn = cell.querySelector('.btn-stop');
    const cellType = cell.querySelector('.cell-type-selector').value;
    const code = input.value;
    
    // Skip markdown cells entirely - they don't get executed
    if (cellType === 'markdown') {
        return;
    }
    
    // Show stop button, hide run button IMMEDIATELY (before any async operations)
    runBtn.style.display = 'none';
    stopBtn.style.display = 'inline-block';
    
    // Update tab order since button visibility changed
    updateTabOrder();
    
    // Force a reflow to ensure UI updates
    stopBtn.offsetHeight;
    
    // Clear previous output
    output.innerHTML = '';
    output.className = 'cell-output';
    
    if (!code.trim()) {
        // Restore buttons if no code
        runBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
        return;
    }
    
    // Use setTimeout to ensure UI updates before starting execution
    await new Promise(resolve => setTimeout(resolve, 0));
    
    try {
        // Run Python code (we've already filtered out markdown cells above)
        output.innerHTML = '<div class="loading">Running...</div>';
        await runPythonCode(code, output, cellId);
        // Ensure Python cells stay expanded but switch to view mode after execution
        input.classList.remove('collapsed');
        input.setAttribute('data-edit-mode', 'false');
        input.setAttribute('readonly', 'readonly');
        input.classList.add('view-mode');
        
        // Update toggle button to show pencil icon for edit mode activation
        const toggleBtn = cell.querySelector('.btn-toggle');
        toggleBtn.innerHTML = '<span style="filter: brightness(0) invert(1);">✏️</span>';
        toggleBtn.setAttribute('aria-label', 'Switch to edit mode');
        toggleBtn.setAttribute('title', 'Switch to edit mode');
        
        // Update tab order since textarea properties changed
        updateTabOrder();
    } catch (error) {
        output.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        // Restore buttons for code cells only
        runBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
        // Update tab order since button visibility changed
        updateTabOrder();
    }
}

// Run Python code using PyScript
async function runPythonCode(code, outputElement, cellId) {
    try {
        // Wait for PyScript to be ready
        if (!pyScriptReady) {
            outputElement.innerHTML = '<div class="loading">Waiting for PyScript to initialize...</div>';
            await waitForPyScript();
        }
        
        outputElement.innerHTML = '<div class="loading">Running...</div>';
        
        // Generate unique IDs
        const outputId = 'output-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        outputElement.id = outputId;
        
        // Store reference globally so Python can access it
        window.currentOutputElement = outputElement;
        
        // Create inline script tag
        const scriptTag = document.createElement('script');
        scriptTag.type = 'py';
        // Note: Not using worker attribute because it requires secure context (HTTPS + proper headers)
        // Without workers, Stop button works for I/O-bound code but not pure compute loops
        
        // Add error handler
        scriptTag.addEventListener('error', (e) => {
            console.error('PyScript tag error:', e);
            outputElement.innerHTML = '<div class="error">PyScript failed to execute. Check console for details.</div>';
        });
        
        scriptTag.innerHTML = `
import js
import sys
from io import StringIO
import warnings

# Suppress all warnings including matplotlib backend warnings
warnings.filterwarnings('ignore')

output_id = "${outputId}"
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = StringIO()
sys.stderr = StringIO()

output_text = ""
stderr_text = ""
figures = []
error_msg = None

# User code as a string to handle syntax errors
user_code = """${code.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"""

try:
    # Try to compile first to catch syntax errors
    compiled_code = compile(user_code, '<cell>', 'exec')
    # Execute the compiled code
    exec(compiled_code)
    
    output_text = sys.stdout.getvalue()
    stderr_text = sys.stderr.getvalue()
    
    # Handle matplotlib figures
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        plt.ioff()
        
        for i in plt.get_fignums():
            fig = plt.figure(i)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            figures.append(img_str)
            plt.close(fig)
    except Exception:
        pass
    
except Exception as e:
    import traceback
    error_msg = traceback.format_exc()

finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Build HTML output
    html_output = ""
    if error_msg:
        import html
        html_output = f'<div class="error">{html.escape(error_msg)}</div>'
    else:
        if output_text:
            import html
            html_output += f'<pre class="text-output">{html.escape(output_text)}</pre>'
        
        if stderr_text and not error_msg:
            import html
            html_output += f'<pre class="text-output stderr">{html.escape(stderr_text)}</pre>'
        
        for img_data in figures:
            html_output += f'<img src="data:image/png;base64,{img_data}" class="plot-output" alt="Plot">'
        
        if not html_output:
            html_output = '<div class="no-output">Code executed successfully (no output)</div>'
    
    # Update the element
    element = js.document.getElementById(output_id)
    if element:
        element.innerHTML = html_output
        js.console.log(f"Updated element {output_id}")
    else:
        js.console.error(f"Could not find element {output_id}")
`;
        
        console.log(`Appended PyScript tag for cell ${outputId}`);
        document.body.appendChild(scriptTag);
        
        // Track this running cell
        runningCells.set(cellId, { scriptTag, outputId });
        
        // Add a timeout to check if output was updated (in case of error)
        const timeoutId = setTimeout(() => {
            const elem = document.getElementById(outputId);
            if (elem && elem.innerHTML.includes('Running...')) {
                // Output wasn't updated, likely due to an error
                console.error('Cell output still shows "Running..." after 5 seconds. PyScript may have failed silently.');
                elem.innerHTML = '<div class="error">An error occurred during execution. The Python code did not complete. Check the browser console for details.</div>';
                // Clean up tracking
                runningCells.delete(cellId);
            }
        }, 5000);  // Increased to 5 seconds
        
        // Store timeout ID so we can cancel it if stopped manually
        runningCells.get(cellId).timeoutId = timeoutId;
        
        // Clean up after a delay
        setTimeout(() => {
            if (scriptTag.parentNode) {
                scriptTag.parentNode.removeChild(scriptTag);
            }
            // Clean up tracking
            runningCells.delete(cellId);
        }, 6000);
        
    } catch (error) {
        console.error('Execution error:', error);
        outputElement.innerHTML = `<div class="error">Error: ${escapeHtml(error.toString())}</div>`;
    }
}

// Wait for PyScript to be ready
function waitForPyScript() {
    return new Promise((resolve) => {
        if (pyScriptReady) {
            resolve();
        } else {
            window.addEventListener('py:ready', () => resolve(), { once: true });
        }
    });
}

// Run all cells in order
async function runAllCells() {
    const cells = document.querySelectorAll('.cell');
    for (const cell of cells) {
        await runCell(cell.id);
        // Small delay to ensure sequential execution
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

// Clear all cells
function clearAllCells() {
    const notebook = document.getElementById('notebook');
    notebook.innerHTML = '';
    cellCounter = 0;
    addCell();
    uploadedFileContent = null;
    uploadedFileName = null;
    document.getElementById('uploadedFileName').textContent = '';
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    uploadedFileName = file.name;
    const reader = new FileReader();
    
    reader.onload = function(e) {
        uploadedFileContent = e.target.result;
        document.getElementById('uploadedFileName').textContent = `Uploaded: ${uploadedFileName}`;
        
        // Store file in PyScript environment
        const scriptElement = document.createElement('script');
        scriptElement.type = 'py';
        scriptElement.textContent = `
with open("${uploadedFileName}", 'w') as f:
    f.write("""${uploadedFileContent.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}""")
import js
js.console.log("File uploaded: ${uploadedFileName}")
        `;
        document.body.appendChild(scriptElement);
        
        setTimeout(() => {
            scriptElement.remove();
        }, 100);
    };
    
    reader.readAsText(file);
}

// Simple markdown renderer
function renderMarkdown(text) {
    let html = text;
    
    // Tables - handle pipe-separated values FIRST while line structure is completely intact
    html = html.replace(/^(\|[^|\n]*)+\|$/gim, function(match) {
        return '<table-row>' + match + '</table-row>';
    });
    
    // Process tables by finding groups of consecutive table-row elements
    html = html.replace(/(<table-row>[^<]*<\/table-row>(\s)*)+/g, function(match) {
        const rows = match.match(/<table-row>[^<]*<\/table-row>/g);
        if (!rows) return match;
        
        let tableHtml = '<table>';
        
        rows.forEach((row, index) => {
            const rowContent = row.replace(/<\/?table-row>/g, '');
            const cells = rowContent.split('|').filter(cell => cell.trim() !== '').map(cell => cell.trim());
            
            // Check if this is a separator row (contains only dashes and spaces)
            if (rowContent.match(/^\|[\s-|]+\|$/)) {
                return; // Skip separator rows
            }
            
            if (index === 0) {
                // First row - treat as header
                tableHtml += '<thead><tr>';
                cells.forEach(cell => {
                    tableHtml += '<th>' + cell + '</th>';
                });
                tableHtml += '</tr></thead><tbody>';
            } else {
                // Data row
                tableHtml += '<tr>';
                cells.forEach(cell => {
                    tableHtml += '<td>' + cell + '</td>';
                });
                tableHtml += '</tr>';
            }
        });
        
        tableHtml += '</tbody></table>';
        return tableHtml;
    });
    
    // Process blockquotes SECOND while line structure is mostly intact
    html = html.replace(/^> (.*)$/gim, '<blockquote-line>$1</blockquote-line>');
    // Use a simpler approach: replace all consecutive blockquote-line groups
    html = html.replace(/(<blockquote-line>[^<]*<\/blockquote-line>\s*)+/g, function(match) {
        // Extract all the content between blockquote-line tags
        const lines = match.match(/<blockquote-line>([^<]*)<\/blockquote-line>/g);
        if (lines) {
            const content = lines.map(line => line.replace(/<blockquote-line>([^<]*)<\/blockquote-line>/, '$1')).join('<br>');
            return '<blockquote>' + content + '</blockquote>';
        }
        return match;
    });
    
    // Process lists THIRD while line structure is still mostly intact
    // Unordered lists (bulleted) - handle both "- " and "* "
    html = html.replace(/^[-*] (.*)$/gim, '<ul-item>$1</ul-item>');
    html = html.replace(/(<ul-item>.*<\/ul-item>(\n)*)+/gs, function(match) {
        const items = match.replace(/<ul-item>/g, '<li>').replace(/<\/ul-item>/g, '</li>').replace(/(\n)+/g, '');
        return '<ul>' + items + '</ul>';
    });
    
    // Numbered lists (ordered) - handle "1. ", "2. ", etc.
    html = html.replace(/^(\d+)\. (.*)$/gim, function(match, num, content) {
        return '<ol-item>' + content + '</ol-item>';
    });
    html = html.replace(/(<ol-item>.*?<\/ol-item>(\n)*)+/gs, function(match) {
        const items = match.replace(/<ol-item>/g, '<li>').replace(/<\/ol-item>/g, '</li>').replace(/(\n)+/g, '');
        return '<ol>' + items + '</ol>';
    });
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Links with {:target="_blank"} syntax
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)\{:target="_blank"\}/g, '<a href="$2" target="_blank">$1</a>');
    
    // Regular links (without explicit target)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    
    // Code blocks
    html = html.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Line breaks (convert remaining newlines to <br>)
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Save notebook to .pysb file
function saveNotebook() {
    const cells = document.querySelectorAll('.cell');
    const notebookData = {
        version: '1.0',
        cells: []
    };
    
    cells.forEach(cell => {
        const cellType = cell.querySelector('.cell-type-selector').value;
        const cellContent = cell.querySelector('.cell-input').value;
        
        notebookData.cells.push({
            type: cellType,
            content: cellContent
        });
    });
    
    // Convert to JSON
    const json = JSON.stringify(notebookData, null, 2);
    
    // Create blob and download
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'notebook.pysb';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('Notebook saved successfully');
}

// Load notebook from .pysb file
function loadNotebook(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            const notebookData = JSON.parse(e.target.result);
            
            // Clear existing cells
            const notebook = document.getElementById('notebook');
            notebook.innerHTML = '';
            cellCounter = 0;
            
            // Load cells from file
            if (notebookData.cells && notebookData.cells.length > 0) {
                const markdownCellsToToggle = [];
                
                notebookData.cells.forEach((cellData, index) => {
                    addCell();
                    
                    // Get the newly added cell
                    const cells = document.querySelectorAll('.cell');
                    const cell = cells[cells.length - 1];
                    const cellId = cell.id;
                    
                    // Set cell type and trigger change event to update button visibility
                    const typeSelector = cell.querySelector('.cell-type-selector');
                    typeSelector.value = cellData.type || 'python';
                    typeSelector.dispatchEvent(new Event('change'));
                    
                    // Set cell content
                    const input = cell.querySelector('.cell-input');
                    input.value = cellData.content || '';
                    
                    // Track markdown cells with content for later view mode switch
                    if (cellData.type === 'markdown' && cellData.content) {
                        markdownCellsToToggle.push(cellId);
                    }
                });
                
                // Switch markdown cells to view mode after all cells are loaded
                setTimeout(() => {
                    markdownCellsToToggle.forEach(cellId => {
                        toggleEditMode(cellId);
                    });
                }, 50);
            } else {
                // If no cells in file, add one empty cell
                addCell();
            }
            
            console.log('Notebook loaded successfully');
        } catch (error) {
            alert('Error loading notebook: ' + error.message);
            console.error('Error loading notebook:', error);
        }
    };
    
    reader.readAsText(file);
    
    // Reset file input
    event.target.value = '';
}

// Update tab order for proper keyboard navigation
function updateTabOrder() {
    let tabIndex = 1;
    
    // Start with toolbar buttons
    const toolbarButtons = document.querySelectorAll('.toolbar button, .toolbar label');
    toolbarButtons.forEach(button => {
        button.setAttribute('tabindex', tabIndex++);
    });
    
    // Then process each cell in document order
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
        // Add cell button before each cell
        const hoverZone = cell.previousElementSibling;
        if (hoverZone && hoverZone.classList.contains('cell-hover-zone')) {
            const addButton = hoverZone.querySelector('.add-cell-button');
            if (addButton) {
                addButton.setAttribute('tabindex', tabIndex++);
            }
        }
        
        // Cell controls in left-to-right order
        const selector = cell.querySelector('.cell-type-selector');
        if (selector) selector.setAttribute('tabindex', tabIndex++);
        
        const runBtn = cell.querySelector('.btn-run');
        if (runBtn) runBtn.setAttribute('tabindex', tabIndex++);
        
        const stopBtn = cell.querySelector('.btn-stop');
        if (stopBtn) {
            // Only set tabindex if visible, otherwise remove from tab order
            if (stopBtn.style.display !== 'none') {
                stopBtn.setAttribute('tabindex', tabIndex++);
            } else {
                stopBtn.setAttribute('tabindex', '-1');
            }
        }
        
        const toggleBtn = cell.querySelector('.btn-toggle');
        if (toggleBtn) toggleBtn.setAttribute('tabindex', tabIndex++);
        
        const deleteBtn = cell.querySelector('.btn-delete');
        if (deleteBtn) deleteBtn.setAttribute('tabindex', tabIndex++);
        
        // Cell textarea
        const textarea = cell.querySelector('.cell-input');
        if (textarea) {
            // Only include in tab order if not collapsed
            if (!textarea.classList.contains('collapsed')) {
                textarea.setAttribute('tabindex', tabIndex++);
            } else {
                textarea.setAttribute('tabindex', '-1');
            }
        }
    });
    
    // Finally, the end add cell button
    const endHoverZone = document.querySelector('.cell-hover-zone-end');
    if (endHoverZone) {
        const addButton = endHoverZone.querySelector('.add-cell-button');
        if (addButton) {
            addButton.setAttribute('tabindex', tabIndex++);
        }
    }
}
