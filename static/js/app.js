/**
 * Neural Network Builder - Interactive Web Application
 * Allows creating, configuring, and exporting neural network architectures
 */

class NeuralNetworkBuilder {
    constructor() {
        // Tab management
        this.tabs = new Map();
        this.activeTabId = null;
        this.tabCounter = 0;
        
        // Current model state (references active tab's data)
        this.layers = new Map();
        this.connections = [];
        this.selectedLayerId = null;
        this.layerCounter = 0;
        this.layerShapes = {};
        
        // Drag and connection state
        this.isDragging = false;
        this.isConnecting = false;
        this.connectingFrom = null;
        this.tempLine = null;
        
        this.init();
    }
    
    init() {
        this.workspace = document.getElementById('workspace');
        this.layersContainer = document.getElementById('layers-container');
        this.connectionsSvg = document.getElementById('connections-svg');
        this.configPanel = document.getElementById('config-panel');
        this.configContent = document.getElementById('config-content');
        this.configTitle = document.getElementById('config-title');
        this.validationStatus = document.getElementById('validation-status');
        this.validationErrors = document.getElementById('validation-errors');
        
        // Tab elements
        this.tabsContainer = document.getElementById('tabs-container');
        
        // Modal elements
        this.codeModal = document.getElementById('code-preview-modal');
        this.codePreview = document.getElementById('code-preview').querySelector('code');
        this.generatedCode = '';
        
        this.setupEventListeners();
        
        // Create initial tab
        this.createNewTab();
    }
    
    setupEventListeners() {
        // Palette drag and drop
        document.querySelectorAll('.palette-item').forEach(item => {
            item.addEventListener('dragstart', (e) => this.onPaletteDragStart(e));
            item.addEventListener('dragend', (e) => this.onPaletteDragEnd(e));
        });
        
        // Workspace drop zone
        this.workspace.addEventListener('dragover', (e) => this.onWorkspaceDragOver(e));
        this.workspace.addEventListener('drop', (e) => this.onWorkspaceDrop(e));
        
        // Workspace click to deselect
        this.workspace.addEventListener('click', (e) => {
            if (e.target === this.workspace || e.target === this.layersContainer) {
                this.deselectAll();
            }
        });
        
        // Mouse move for connection drawing
        this.workspace.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.workspace.addEventListener('mouseup', (e) => this.onMouseUp(e));
        
        // Action buttons
        document.getElementById('validate-btn').addEventListener('click', () => this.validateNetwork());
        document.getElementById('export-btn').addEventListener('click', () => this.exportPyTorch());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearAll());
        
        // Config panel
        document.getElementById('close-config').addEventListener('click', () => this.closeConfigPanel());
        document.getElementById('delete-layer-btn').addEventListener('click', () => this.deleteSelectedLayer());
        
        // Modal controls
        this.codeModal.querySelector('.modal-overlay').addEventListener('click', () => this.closeCodeModal());
        this.codeModal.querySelector('.modal-close-btn').addEventListener('click', () => this.closeCodeModal());
        document.getElementById('copy-code-btn').addEventListener('click', () => this.copyCodeToClipboard());
        document.getElementById('download-code-btn').addEventListener('click', () => this.downloadCode());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Close modal on Escape
            if (e.key === 'Escape' && !this.codeModal.classList.contains('hidden')) {
                this.closeCodeModal();
            }
            
            // Ctrl+T: New tab
            if (e.ctrlKey && e.key === 't') {
                e.preventDefault();
                this.createNewTab();
            }
            
            // Ctrl+W: Close tab
            if (e.ctrlKey && e.key === 'w') {
                e.preventDefault();
                if (this.activeTabId) {
                    this.closeTab(this.activeTabId);
                }
            }
            
            // Ctrl+Tab: Next tab
            if (e.ctrlKey && e.key === 'Tab') {
                e.preventDefault();
                this.switchToNextTab(e.shiftKey ? -1 : 1);
            }
            
            // Ctrl+D: Duplicate selected layer
            if (e.ctrlKey && e.key === 'd') {
                e.preventDefault();
                if (this.selectedLayerId) {
                    this.duplicateLayer(this.selectedLayerId);
                }
            }
            
            // Delete/Backspace: Delete selected layer
            if ((e.key === 'Delete' || e.key === 'Backspace') && this.selectedLayerId) {
                // Only if not focused on an input
                if (!['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) {
                    e.preventDefault();
                    this.deleteLayer(this.selectedLayerId);
                }
            }
        });
        
        // Tab controls
        document.getElementById('new-tab-btn').addEventListener('click', () => this.createNewTab());
    }
    
    switchToNextTab(direction = 1) {
        const tabIds = Array.from(this.tabs.keys());
        if (tabIds.length <= 1) return;
        
        const currentIndex = tabIds.indexOf(this.activeTabId);
        let newIndex = (currentIndex + direction + tabIds.length) % tabIds.length;
        this.switchToTab(tabIds[newIndex]);
    }
    
    // ========== Tab Management ==========
    
    createNewTab(name = null) {
        const tabId = `tab_${++this.tabCounter}`;
        const tabName = name || `Model ${this.tabCounter}`;
        
        const tabData = {
            id: tabId,
            name: tabName,
            layers: new Map(),
            connections: [],
            layerCounter: 0,
            layerShapes: {},
            selectedLayerId: null
        };
        
        this.tabs.set(tabId, tabData);
        this.renderTab(tabData);
        this.switchToTab(tabId);
        
        return tabId;
    }
    
    renderTab(tabData) {
        const tab = document.createElement('div');
        tab.className = 'tab';
        tab.dataset.tabId = tabData.id;
        
        tab.innerHTML = `
            <span class="tab-name">${tabData.name}</span>
            <button class="tab-close" title="Close tab">×</button>
        `;
        
        tab.addEventListener('click', (e) => {
            if (!e.target.classList.contains('tab-close')) {
                this.switchToTab(tabData.id);
            }
        });
        
        tab.querySelector('.tab-close').addEventListener('click', (e) => {
            e.stopPropagation();
            this.closeTab(tabData.id);
        });
        
        // Double-click to rename
        tab.querySelector('.tab-name').addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.renameTab(tabData.id);
        });
        
        this.tabsContainer.appendChild(tab);
    }
    
    switchToTab(tabId) {
        if (this.activeTabId === tabId) return;
        
        // Save current tab state
        if (this.activeTabId) {
            this.saveCurrentTabState();
        }
        
        // Update active tab
        this.activeTabId = tabId;
        const tabData = this.tabs.get(tabId);
        
        if (!tabData) return;
        
        // Load tab state
        this.layers = tabData.layers;
        this.connections = tabData.connections;
        this.layerCounter = tabData.layerCounter;
        this.layerShapes = tabData.layerShapes;
        this.selectedLayerId = tabData.selectedLayerId;
        
        // Update UI
        this.updateTabUI();
        this.renderAllLayers();
        this.renderConnections();
        this.closeConfigPanel();
        this.updateValidationStatus(null);
    }
    
    saveCurrentTabState() {
        if (!this.activeTabId) return;
        
        const tabData = this.tabs.get(this.activeTabId);
        if (tabData) {
            tabData.layers = this.layers;
            tabData.connections = this.connections;
            tabData.layerCounter = this.layerCounter;
            tabData.layerShapes = this.layerShapes;
            tabData.selectedLayerId = this.selectedLayerId;
        }
    }
    
    updateTabUI() {
        // Update active tab styling
        this.tabsContainer.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tabId === this.activeTabId);
        });
    }
    
    closeTab(tabId) {
        if (this.tabs.size <= 1) {
            // Don't close the last tab, just clear it
            this.clearAll();
            return;
        }
        
        const tabElement = this.tabsContainer.querySelector(`[data-tab-id="${tabId}"]`);
        
        // If closing active tab, switch to another
        if (this.activeTabId === tabId) {
            const tabIds = Array.from(this.tabs.keys());
            const currentIndex = tabIds.indexOf(tabId);
            const newTabId = tabIds[currentIndex === 0 ? 1 : currentIndex - 1];
            this.switchToTab(newTabId);
        }
        
        // Remove tab
        this.tabs.delete(tabId);
        tabElement?.remove();
    }
    
    renameTab(tabId) {
        const tabData = this.tabs.get(tabId);
        if (!tabData) return;
        
        const tabElement = this.tabsContainer.querySelector(`[data-tab-id="${tabId}"]`);
        const nameSpan = tabElement?.querySelector('.tab-name');
        if (!nameSpan) return;
        
        const currentName = tabData.name;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentName;
        input.className = 'tab-name-input';
        input.style.cssText = 'width: 100%; background: var(--card-bg); border: 1px solid var(--primary-color); border-radius: 3px; padding: 2px 4px; color: var(--text-color); font-size: 0.85rem;';
        
        nameSpan.replaceWith(input);
        input.focus();
        input.select();
        
        const finishRename = () => {
            const newName = input.value.trim() || currentName;
            tabData.name = newName;
            
            const newSpan = document.createElement('span');
            newSpan.className = 'tab-name';
            newSpan.textContent = newName;
            newSpan.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                this.renameTab(tabId);
            });
            
            input.replaceWith(newSpan);
        };
        
        input.addEventListener('blur', finishRename);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                input.blur();
            } else if (e.key === 'Escape') {
                input.value = currentName;
                input.blur();
            }
        });
    }
    
    renderAllLayers() {
        // Clear workspace
        this.layersContainer.innerHTML = '';
        
        // Render all layers
        this.layers.forEach(layer => {
            this.renderLayer(layer);
        });
        
        // Restore selection
        if (this.selectedLayerId) {
            const el = document.getElementById(this.selectedLayerId);
            el?.classList.add('selected');
        }
    }
    
    // ========== Palette Drag and Drop ==========
    
    onPaletteDragStart(e) {
        const item = e.target.closest('.palette-item');
        if (!item) return;
        
        e.dataTransfer.setData('layerType', item.dataset.type);
        e.dataTransfer.effectAllowed = 'copy';
        
        // Create a custom drag image showing just the block
        const dragImage = item.cloneNode(true);
        dragImage.style.position = 'absolute';
        dragImage.style.top = '-1000px';
        dragImage.style.left = '-1000px';
        dragImage.style.width = item.offsetWidth + 'px';
        dragImage.style.opacity = '0.9';
        dragImage.style.transform = 'rotate(2deg)';
        dragImage.style.boxShadow = '0 8px 25px rgba(0,0,0,0.4)';
        document.body.appendChild(dragImage);
        
        // Set the drag image to our custom element
        e.dataTransfer.setDragImage(dragImage, item.offsetWidth / 2, item.offsetHeight / 2);
        
        // Clean up the drag image after a short delay
        setTimeout(() => {
            if (dragImage.parentNode) {
                dragImage.parentNode.removeChild(dragImage);
            }
        }, 0);
        
        item.style.opacity = '0.5';
    }
    
    onPaletteDragEnd(e) {
        const item = e.target.closest('.palette-item');
        if (item) {
            item.style.opacity = '1';
        }
    }
    
    onWorkspaceDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }
    
    onWorkspaceDrop(e) {
        e.preventDefault();
        const layerType = e.dataTransfer.getData('layerType');
        if (!layerType) return;
        
        const rect = this.workspace.getBoundingClientRect();
        const x = e.clientX - rect.left - 75; // Center the block
        const y = e.clientY - rect.top - 30;
        
        this.createLayer(layerType, x, y);
    }
    
    // ========== Layer Management ==========
    
    createLayer(type, x, y) {
        const id = `layer_${++this.layerCounter}`;
        const name = this.getDefaultLayerName(type);
        const config = this.getDefaultConfig(type);
        
        const layer = {
            id,
            type,
            name,
            config,
            x: Math.max(0, x),
            y: Math.max(0, y)
        };
        
        this.layers.set(id, layer);
        this.renderLayer(layer);
        this.selectLayer(id);
        this.markTabModified();
        
        return layer;
    }
    
    markTabModified() {
        if (!this.activeTabId) return;
        const tabElement = this.tabsContainer.querySelector(`[data-tab-id="${this.activeTabId}"]`);
        tabElement?.classList.add('modified');
    }
    
    getDefaultLayerName(type) {
        const names = {
            input: 'Input',
            output: 'Output',
            dense: 'Dense',
            conv2d: 'Conv2D',
            pooling: 'Pooling',
            flatten: 'Flatten',
            reshape: 'Reshape',
            concatenate: 'Concat',
            dropout: 'Dropout',
            batchnorm: 'BatchNorm',
            activation: 'Activation'
        };
        return names[type] || 'Layer';
    }
    
    getDefaultConfig(type) {
        const configs = {
            input: { shape: '784', shapePreset: '' },
            output: { neurons: 10, activation: 'softmax' },
            dense: { neurons: 128, activation: 'relu', useBias: true },
            conv2d: { filters: 32, kernel_size: 3, stride: 1, padding: 'valid', activation: 'relu' },
            pooling: { pool_type: 'max', pool_size: 2, stride: 2 },
            flatten: {},
            reshape: { target_shape: '28,28,1' },
            concatenate: { axis: -1 },
            dropout: { rate: 0.5 },
            batchnorm: { momentum: 0.1, epsilon: 0.00001 },
            activation: { activation: 'relu' }
        };
        return configs[type] || {};
    }
    
    getLayerIcon(type) {
        const icons = {
            input: '📥',
            output: '📤',
            dense: '⬛',
            conv2d: '🔲',
            pooling: '📊',
            flatten: '📏',
            reshape: '🔄',
            concatenate: '🔗',
            dropout: '💧',
            batchnorm: '📐',
            activation: '⚡'
        };
        return icons[type] || '📦';
    }
    
    getInputPortCount(type) {
        // Layers with multiple inputs
        const multiInputLayers = {
            concatenate: 2
        };
        return multiInputLayers[type] || 1;
    }
    
    getLayerInfo(layer) {
        const config = layer.config;
        const type = layer.type;
        
        switch (type) {
            case 'input':
                return `Shape: ${config.shape}`;
            case 'output':
                return `Units: ${config.neurons}<br>Activation: ${config.activation}`;
            case 'dense':
                return `Neurons: ${config.neurons}<br>Activation: ${config.activation}`;
            case 'conv2d':
                return `Filters: ${config.filters}<br>Kernel: ${config.kernel_size}×${config.kernel_size}`;
            case 'pooling':
                return `Type: ${config.pool_type}<br>Size: ${config.pool_size}×${config.pool_size}`;
            case 'flatten':
                return 'Flattens input';
            case 'reshape':
                return `Target: (${config.target_shape})`;
            case 'concatenate':
                return `Axis: ${config.axis}`;
            case 'dropout':
                return `Rate: ${config.rate}`;
            case 'batchnorm':
                return `Momentum: ${config.momentum}`;
            case 'activation':
                return `Function: ${config.activation}`;
            default:
                return '';
        }
    }
    
    renderLayer(layer) {
        const existingEl = document.getElementById(layer.id);
        if (existingEl) {
            existingEl.remove();
        }
        
        const el = document.createElement('div');
        el.id = layer.id;
        el.className = 'layer-block';
        el.dataset.type = layer.type;
        el.style.left = `${layer.x}px`;
        el.style.top = `${layer.y}px`;
        
        const shapeInfo = this.layerShapes[layer.id] 
            ? `<div class="layer-shape">Output: (${this.layerShapes[layer.id].join(', ')})</div>` 
            : '';
        
        // Determine which ports to show
        const hasInputPort = layer.type !== 'input';
        const hasOutputPort = layer.type !== 'output';
        const inputPortCount = this.getInputPortCount(layer.type);
        
        // Generate input ports HTML
        let inputPortsHtml = '';
        if (hasInputPort) {
            if (inputPortCount > 1) {
                // Multiple input ports
                let portsHtml = '';
                for (let i = 0; i < inputPortCount; i++) {
                    const portId = `input_${i}`;
                    const hasConnection = this.connections.some(conn => conn.to === layer.id && conn.toPort === portId);
                    portsHtml += `
                        <div class="input-port-wrapper${hasConnection ? ' has-connection' : ''}">
                            <div class="port input-port" data-port="${portId}"></div>
                            ${hasConnection ? '<span class="connection-delete-btn" title="Remove connection" data-port="' + portId + '">×</span>' : ''}
                            <span class="input-port-label">${String.fromCharCode(65 + i)}</span>
                        </div>
                    `;
                }
                inputPortsHtml = `<div class="multi-input-ports">${portsHtml}</div>`;
            } else {
                // Single input port
                const hasIncomingConnection = this.connections.some(conn => conn.to === layer.id);
                inputPortsHtml = `
                    <div class="input-port-wrapper${hasIncomingConnection ? ' has-connection' : ''}">
                        <div class="port input-port" data-port="input"></div>
                        ${hasIncomingConnection ? '<span class="connection-delete-btn" title="Remove connection">×</span>' : ''}
                    </div>
                `;
            }
        }
        
        el.innerHTML = `
            <div class="layer-actions">
                <button class="layer-action-btn layer-duplicate-btn" title="Duplicate layer">⧉</button>
                <button class="layer-action-btn layer-delete-btn" title="Delete layer">×</button>
            </div>
            <div class="layer-header">
                <span class="layer-icon">${this.getLayerIcon(layer.type)}</span>
                <span class="layer-name">${layer.name}</span>
            </div>
            <div class="layer-body">
                <div class="layer-info">${this.getLayerInfo(layer)}</div>
                ${shapeInfo}
            </div>
            ${inputPortsHtml}
            ${hasOutputPort ? '<div class="port output-port" data-port="output"></div>' : ''}
        `;
        
        // Delete button
        el.querySelector('.layer-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteLayer(layer.id);
        });
        
        // Duplicate button
        el.querySelector('.layer-duplicate-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.duplicateLayer(layer.id);
        });
        
        // Connection delete buttons
        el.querySelectorAll('.connection-delete-btn').forEach(btn => {
            btn.addEventListener('mousedown', (e) => {
                e.stopPropagation(); // Prevent port connection from starting
            });
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const portId = btn.dataset.port || 'input';
                this.removeConnectionTo(layer.id, portId);
            });
        });
        
        // Layer drag
        el.addEventListener('mousedown', (e) => this.onLayerMouseDown(e, layer.id));
        
        // Layer click to select
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectLayer(layer.id);
        });
        
        // Port interactions
        el.querySelectorAll('.port').forEach(port => {
            port.addEventListener('mousedown', (e) => this.onPortMouseDown(e, layer.id, port.dataset.port));
            port.addEventListener('mouseup', (e) => this.onPortMouseUp(e, layer.id, port.dataset.port));
        });
        
        this.layersContainer.appendChild(el);
    }
    
    removeConnectionTo(layerId, portId = 'input') {
        const index = this.connections.findIndex(conn => 
            conn.to === layerId && (conn.toPort || 'input') === portId
        );
        if (index !== -1) {
            this.connections.splice(index, 1);
            this.renderConnections();
            // Re-render the layer to update the delete button visibility
            const layer = this.layers.get(layerId);
            if (layer) {
                this.renderLayer(layer);
                if (this.selectedLayerId === layerId) {
                    document.getElementById(layerId)?.classList.add('selected');
                }
            }
        }
    }
    
    updateLayerDisplay(layerId) {
        const layer = this.layers.get(layerId);
        if (layer) {
            this.renderLayer(layer);
            if (this.selectedLayerId === layerId) {
                document.getElementById(layerId)?.classList.add('selected');
            }
        }
    }
    
    selectLayer(id) {
        this.deselectAll();
        this.selectedLayerId = id;
        
        const el = document.getElementById(id);
        if (el) {
            el.classList.add('selected');
        }
        
        this.openConfigPanel(id);
    }
    
    deselectAll() {
        document.querySelectorAll('.layer-block.selected').forEach(el => {
            el.classList.remove('selected');
        });
        this.selectedLayerId = null;
        this.closeConfigPanel();
    }
    
    deleteLayer(id) {
        // Remove connections
        this.connections = this.connections.filter(conn => 
            conn.from !== id && conn.to !== id
        );
        
        // Remove layer
        this.layers.delete(id);
        delete this.layerShapes[id];
        
        // Remove DOM element
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
        
        this.renderConnections();
        this.closeConfigPanel();
    }
    
    deleteSelectedLayer() {
        if (this.selectedLayerId) {
            this.deleteLayer(this.selectedLayerId);
        }
    }
    
    duplicateLayer(id) {
        const originalLayer = this.layers.get(id);
        if (!originalLayer) return;
        
        // Create a deep copy of the config
        const newConfig = JSON.parse(JSON.stringify(originalLayer.config));
        
        // Generate new ID and name
        const newId = `layer_${++this.layerCounter}`;
        const newName = `${originalLayer.name} (copy)`;
        
        // Create new layer with offset position
        const newLayer = {
            id: newId,
            type: originalLayer.type,
            name: newName,
            config: newConfig,
            x: originalLayer.x + 30,
            y: originalLayer.y + 30
        };
        
        this.layers.set(newId, newLayer);
        this.renderLayer(newLayer);
        this.selectLayer(newId);
        this.markTabModified();
        
        return newLayer;
    }
    
    clearAll() {
        if (confirm('Are you sure you want to clear all layers in this model?')) {
            this.layers.clear();
            this.connections = [];
            this.layerShapes = {};
            this.layerCounter = 0;
            this.selectedLayerId = null;
            this.layersContainer.innerHTML = '';
            this.connectionsSvg.innerHTML = '';
            this.closeConfigPanel();
            this.updateValidationStatus(null);
            
            // Update tab state
            this.saveCurrentTabState();
        }
    }
    
    // ========== Layer Dragging ==========
    
    onLayerMouseDown(e, layerId) {
        if (e.target.classList.contains('port')) return;
        
        e.preventDefault();
        this.isDragging = true;
        this.dragLayerId = layerId;
        
        const layer = this.layers.get(layerId);
        const rect = this.workspace.getBoundingClientRect();
        
        this.dragOffset = {
            x: e.clientX - rect.left - layer.x,
            y: e.clientY - rect.top - layer.y
        };
        
        document.getElementById(layerId).classList.add('dragging');
        
        const onMouseMove = (e) => {
            if (!this.isDragging) return;
            
            const rect = this.workspace.getBoundingClientRect();
            const x = Math.max(0, e.clientX - rect.left - this.dragOffset.x);
            const y = Math.max(0, e.clientY - rect.top - this.dragOffset.y);
            
            layer.x = x;
            layer.y = y;
            
            const el = document.getElementById(layerId);
            el.style.left = `${x}px`;
            el.style.top = `${y}px`;
            
            this.renderConnections();
        };
        
        const onMouseUp = () => {
            this.isDragging = false;
            document.getElementById(layerId)?.classList.remove('dragging');
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    }
    
    // ========== Connection Management ==========
    
    onPortMouseDown(e, layerId, portType) {
        e.stopPropagation();
        e.preventDefault();
        
        if (portType === 'output') {
            this.isConnecting = true;
            this.connectingFrom = { layerId, portType };
            
            const port = e.target;
            port.classList.add('connecting');
            
            // Create temporary line
            const rect = this.workspace.getBoundingClientRect();
            const portRect = port.getBoundingClientRect();
            
            this.connectStartX = portRect.left + portRect.width / 2 - rect.left;
            this.connectStartY = portRect.top + portRect.height / 2 - rect.top;
            
            this.tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            this.tempLine.classList.add('connection-line', 'temp');
            this.connectionsSvg.appendChild(this.tempLine);
        }
    }
    
    onPortMouseUp(e, layerId, portType) {
        e.stopPropagation();
        
        // Check if this is an input port (starts with 'input')
        const isInputPort = portType === 'input' || portType.startsWith('input_');
        
        if (this.isConnecting && isInputPort && this.connectingFrom) {
            // Check if not connecting to self
            if (this.connectingFrom.layerId !== layerId) {
                // Check if connection already exists to this specific port
                const exists = this.connections.some(conn => 
                    conn.from === this.connectingFrom.layerId && 
                    conn.to === layerId && 
                    (conn.toPort || 'input') === portType
                );
                
                if (!exists) {
                    // Remove existing connection to this specific input port
                    this.connections = this.connections.filter(conn => 
                        !(conn.to === layerId && (conn.toPort || 'input') === portType)
                    );
                    
                    // Add new connection with port info
                    this.connections.push({
                        from: this.connectingFrom.layerId,
                        to: layerId,
                        toPort: portType
                    });
                    
                    this.renderConnections();
                    this.markTabModified();
                    
                    // Re-render the target layer to show the delete button
                    const layer = this.layers.get(layerId);
                    if (layer) {
                        this.renderLayer(layer);
                        if (this.selectedLayerId === layerId) {
                            document.getElementById(layerId)?.classList.add('selected');
                        }
                    }
                }
            }
        }
        
        this.endConnection();
    }
    
    onMouseMove(e) {
        if (!this.isConnecting || !this.tempLine) return;
        
        const rect = this.workspace.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const path = this.createConnectionPath(
            this.connectStartX, this.connectStartY, x, y
        );
        this.tempLine.setAttribute('d', path);
    }
    
    onMouseUp(e) {
        if (this.isConnecting) {
            this.endConnection();
        }
    }
    
    endConnection() {
        this.isConnecting = false;
        this.connectingFrom = null;
        
        if (this.tempLine) {
            this.tempLine.remove();
            this.tempLine = null;
        }
        
        document.querySelectorAll('.port.connecting').forEach(p => {
            p.classList.remove('connecting');
        });
    }
    
    createConnectionPath(x1, y1, x2, y2) {
        const midX = (x1 + x2) / 2;
        const controlOffset = Math.min(Math.abs(x2 - x1) * 0.5, 100);
        
        return `M ${x1} ${y1} C ${x1 + controlOffset} ${y1}, ${x2 - controlOffset} ${y2}, ${x2} ${y2}`;
    }
    
    renderConnections() {
        // Clear existing connection lines (but not temp line)
        this.connectionsSvg.querySelectorAll('.connection-line:not(.temp)').forEach(el => el.remove());
        
        const workspaceRect = this.workspace.getBoundingClientRect();
        
        this.connections.forEach((conn, index) => {
            const fromEl = document.getElementById(conn.from);
            const toEl = document.getElementById(conn.to);
            
            if (!fromEl || !toEl) return;
            
            const fromPort = fromEl.querySelector('.output-port');
            // Find the correct input port based on toPort
            const toPortId = conn.toPort || 'input';
            const toPort = toEl.querySelector(`.input-port[data-port="${toPortId}"]`);
            
            if (!fromPort || !toPort) return;
            
            const fromRect = fromPort.getBoundingClientRect();
            const toRect = toPort.getBoundingClientRect();
            
            const x1 = fromRect.left + fromRect.width / 2 - workspaceRect.left;
            const y1 = fromRect.top + fromRect.height / 2 - workspaceRect.top;
            const x2 = toRect.left + toRect.width / 2 - workspaceRect.left;
            const y2 = toRect.top + toRect.height / 2 - workspaceRect.top;
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.classList.add('connection-line');
            path.setAttribute('d', this.createConnectionPath(x1, y1, x2, y2));
            path.dataset.connectionIndex = index;
            
            this.connectionsSvg.appendChild(path);
        });
    }
    
    // ========== Configuration Panel ==========
    
    openConfigPanel(layerId) {
        const layer = this.layers.get(layerId);
        if (!layer) return;
        
        const template = document.getElementById(`${layer.type}-config-template`);
        if (!template) return;
        
        this.configTitle.textContent = `${layer.name} Configuration`;
        this.configContent.innerHTML = '';
        this.configContent.appendChild(template.content.cloneNode(true));
        
        // Populate values
        this.populateConfigValues(layer);
        
        // Setup change listeners
        this.setupConfigListeners(layerId);
        
        this.configPanel.classList.remove('hidden');
    }
    
    populateConfigValues(layer) {
        const config = layer.config;
        
        // Common
        const nameInput = this.configContent.querySelector('#layer-name');
        if (nameInput) nameInput.value = layer.name;
        
        // Type specific
        switch (layer.type) {
            case 'input':
                const shapePreset = this.configContent.querySelector('#shape-preset');
                const shapeInput = this.configContent.querySelector('#input-shape');
                if (shapePreset) shapePreset.value = config.shapePreset || '';
                if (shapeInput) shapeInput.value = config.shape || '784';
                break;
                
            case 'output':
                const outputNeurons = this.configContent.querySelector('#neurons');
                const outputActivation = this.configContent.querySelector('#activation');
                if (outputNeurons) outputNeurons.value = config.neurons || 10;
                if (outputActivation) outputActivation.value = config.activation || 'softmax';
                break;
                
            case 'dense':
                const neuronsInput = this.configContent.querySelector('#neurons');
                const activationSelect = this.configContent.querySelector('#activation');
                const biasCheckbox = this.configContent.querySelector('#use-bias');
                if (neuronsInput) neuronsInput.value = config.neurons || 128;
                if (activationSelect) activationSelect.value = config.activation || 'relu';
                if (biasCheckbox) biasCheckbox.checked = config.useBias !== false;
                break;
                
            case 'conv2d':
                const filtersInput = this.configContent.querySelector('#filters');
                const kernelInput = this.configContent.querySelector('#kernel-size');
                const strideInput = this.configContent.querySelector('#stride');
                const paddingSelect = this.configContent.querySelector('#padding');
                const convActivation = this.configContent.querySelector('#activation');
                if (filtersInput) filtersInput.value = config.filters || 32;
                if (kernelInput) kernelInput.value = config.kernel_size || 3;
                if (strideInput) strideInput.value = config.stride || 1;
                if (paddingSelect) paddingSelect.value = config.padding || 'valid';
                if (convActivation) convActivation.value = config.activation || 'relu';
                break;
                
            case 'pooling':
                const poolTypeSelect = this.configContent.querySelector('#pool-type');
                const poolSizeInput = this.configContent.querySelector('#pool-size');
                const poolStrideInput = this.configContent.querySelector('#stride');
                if (poolTypeSelect) poolTypeSelect.value = config.pool_type || 'max';
                if (poolSizeInput) poolSizeInput.value = config.pool_size || 2;
                if (poolStrideInput) poolStrideInput.value = config.stride || 2;
                break;
                
            case 'reshape':
                const targetShapeInput = this.configContent.querySelector('#target-shape');
                if (targetShapeInput) targetShapeInput.value = config.target_shape || '28,28,1';
                break;
            
            case 'concatenate':
                const concatAxisSelect = this.configContent.querySelector('#concat-axis');
                if (concatAxisSelect) concatAxisSelect.value = config.axis !== undefined ? config.axis : -1;
                break;
                
            case 'dropout':
                const rateInput = this.configContent.querySelector('#dropout-rate');
                if (rateInput) rateInput.value = config.rate || 0.5;
                break;
                
            case 'batchnorm':
                const momentumInput = this.configContent.querySelector('#momentum');
                const epsilonInput = this.configContent.querySelector('#epsilon');
                if (momentumInput) momentumInput.value = config.momentum || 0.1;
                if (epsilonInput) epsilonInput.value = config.epsilon || 0.00001;
                break;
                
            case 'activation':
                const actSelect = this.configContent.querySelector('#activation');
                if (actSelect) actSelect.value = config.activation || 'relu';
                break;
        }
    }
    
    setupConfigListeners(layerId) {
        const layer = this.layers.get(layerId);
        if (!layer) return;
        
        const updateLayer = () => {
            // Update name
            const nameInput = this.configContent.querySelector('#layer-name');
            if (nameInput && nameInput.value) {
                layer.name = nameInput.value;
            }
            
            // Update type-specific config
            switch (layer.type) {
                case 'input':
                    const shapeInput = this.configContent.querySelector('#input-shape');
                    const shapePreset = this.configContent.querySelector('#shape-preset');
                    if (shapeInput) layer.config.shape = shapeInput.value;
                    if (shapePreset) layer.config.shapePreset = shapePreset.value;
                    break;
                    
                case 'output':
                    const outputNeurons = this.configContent.querySelector('#neurons');
                    const outputActivation = this.configContent.querySelector('#activation');
                    if (outputNeurons) layer.config.neurons = parseInt(outputNeurons.value) || 10;
                    if (outputActivation) layer.config.activation = outputActivation.value;
                    break;
                    
                case 'dense':
                    const neuronsInput = this.configContent.querySelector('#neurons');
                    const activationSelect = this.configContent.querySelector('#activation');
                    const biasCheckbox = this.configContent.querySelector('#use-bias');
                    if (neuronsInput) layer.config.neurons = parseInt(neuronsInput.value) || 128;
                    if (activationSelect) layer.config.activation = activationSelect.value;
                    if (biasCheckbox) layer.config.useBias = biasCheckbox.checked;
                    break;
                    
                case 'conv2d':
                    const filtersInput = this.configContent.querySelector('#filters');
                    const kernelInput = this.configContent.querySelector('#kernel-size');
                    const strideInput = this.configContent.querySelector('#stride');
                    const paddingSelect = this.configContent.querySelector('#padding');
                    const convActivation = this.configContent.querySelector('#activation');
                    if (filtersInput) layer.config.filters = parseInt(filtersInput.value) || 32;
                    if (kernelInput) layer.config.kernel_size = parseInt(kernelInput.value) || 3;
                    if (strideInput) layer.config.stride = parseInt(strideInput.value) || 1;
                    if (paddingSelect) layer.config.padding = paddingSelect.value;
                    if (convActivation) layer.config.activation = convActivation.value;
                    break;
                    
                case 'pooling':
                    const poolTypeSelect = this.configContent.querySelector('#pool-type');
                    const poolSizeInput = this.configContent.querySelector('#pool-size');
                    const poolStrideInput = this.configContent.querySelector('#stride');
                    if (poolTypeSelect) layer.config.pool_type = poolTypeSelect.value;
                    if (poolSizeInput) layer.config.pool_size = parseInt(poolSizeInput.value) || 2;
                    if (poolStrideInput) layer.config.stride = parseInt(poolStrideInput.value) || 2;
                    break;
                    
                case 'reshape':
                    const targetShapeInput = this.configContent.querySelector('#target-shape');
                    if (targetShapeInput) layer.config.target_shape = targetShapeInput.value;
                    break;
                
                case 'concatenate':
                    const concatAxisSelect = this.configContent.querySelector('#concat-axis');
                    if (concatAxisSelect) layer.config.axis = parseInt(concatAxisSelect.value);
                    break;
                    
                case 'dropout':
                    const rateInput = this.configContent.querySelector('#dropout-rate');
                    if (rateInput) layer.config.rate = parseFloat(rateInput.value) || 0.5;
                    break;
                    
                case 'batchnorm':
                    const momentumInput = this.configContent.querySelector('#momentum');
                    const epsilonInput = this.configContent.querySelector('#epsilon');
                    if (momentumInput) layer.config.momentum = parseFloat(momentumInput.value) || 0.1;
                    if (epsilonInput) layer.config.epsilon = parseFloat(epsilonInput.value) || 0.00001;
                    break;
                    
                case 'activation':
                    const actSelect = this.configContent.querySelector('#activation');
                    if (actSelect) layer.config.activation = actSelect.value;
                    break;
            }
            
            this.updateLayerDisplay(layerId);
            this.configTitle.textContent = `${layer.name} Configuration`;
        };
        
        // Special handling for shape preset dropdown
        if (layer.type === 'input') {
            const shapePreset = this.configContent.querySelector('#shape-preset');
            const shapeInput = this.configContent.querySelector('#input-shape');
            
            if (shapePreset && shapeInput) {
                shapePreset.addEventListener('change', () => {
                    if (shapePreset.value) {
                        shapeInput.value = shapePreset.value;
                        layer.config.shape = shapePreset.value;
                        layer.config.shapePreset = shapePreset.value;
                        this.updateLayerDisplay(layerId);
                    }
                });
            }
        }
        
        // Add listeners to all inputs
        this.configContent.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('change', updateLayer);
            input.addEventListener('input', updateLayer);
        });
    }
    
    closeConfigPanel() {
        this.configPanel.classList.add('hidden');
    }
    
    // ========== Validation ==========
    
    async validateNetwork() {
        const layers = Array.from(this.layers.values());
        const connections = this.connections;
        
        if (layers.length === 0) {
            this.updateValidationStatus(false, [{ type: 'error', message: 'No layers in network' }]);
            return false;
        }
        
        try {
            const response = await fetch('/api/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layers, connections })
            });
            
            const result = await response.json();
            
            this.layerShapes = result.shapes || {};
            this.updateValidationStatus(result.valid, result.errors);
            
            // Update layer displays with shapes
            layers.forEach(layer => this.updateLayerDisplay(layer.id));
            
            // Highlight error layers
            document.querySelectorAll('.layer-block').forEach(el => {
                el.classList.remove('error');
            });
            
            result.errors.forEach(error => {
                if (error.layer_id) {
                    const el = document.getElementById(error.layer_id);
                    if (el && error.type === 'error') {
                        el.classList.add('error');
                    }
                }
            });
            
            return result.valid;
        } catch (error) {
            console.error('Validation error:', error);
            this.updateValidationStatus(false, [{ type: 'error', message: 'Validation request failed' }]);
            return false;
        }
    }
    
    updateValidationStatus(valid, errors = []) {
        if (valid === null) {
            this.validationStatus.innerHTML = '<span class="status-pending">Not validated</span>';
            this.validationErrors.innerHTML = '';
            return;
        }
        
        if (valid) {
            this.validationStatus.innerHTML = '<span class="status-valid">✓ Network is valid</span>';
        } else {
            this.validationStatus.innerHTML = '<span class="status-invalid">✗ Network has errors</span>';
        }
        
        this.validationErrors.innerHTML = errors.map(error => `
            <div class="error-item ${error.type}">${error.message}</div>
        `).join('');
    }
    
    // ========== Export ==========
    
    async exportPyTorch() {
        // Validate first
        const isValid = await this.validateNetwork();
        
        if (!isValid) {
            alert('Please fix validation errors before exporting.');
            return;
        }
        
        const layers = Array.from(this.layers.values());
        const connections = this.connections;
        
        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layers, connections })
            });
            
            if (!response.ok) {
                const error = await response.json();
                alert('Export failed: ' + (error.errors?.[0]?.message || 'Unknown error'));
                return;
            }
            
            const code = await response.text();
            this.generatedCode = code;
            
            // Show code preview modal
            this.showCodeModal(code);
            
        } catch (error) {
            console.error('Export error:', error);
            alert('Export request failed');
        }
    }
    
    // ========== Code Preview Modal ==========
    
    showCodeModal(code) {
        // Apply syntax highlighting
        const highlightedCode = this.highlightPythonSyntax(code);
        this.codePreview.innerHTML = highlightedCode;
        this.codeModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
    
    closeCodeModal() {
        this.codeModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    
    highlightPythonSyntax(code) {
        // Escape HTML characters first
        const escapeHtml = (str) => {
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        };
        
        // Tokenize and highlight to avoid regex conflicts
        const lines = code.split('\n');
        const highlightedLines = lines.map(line => {
            let result = '';
            let i = 0;
            
            while (i < line.length) {
                // Check for comment
                if (line[i] === '#') {
                    result += '<span class="code-comment">' + escapeHtml(line.slice(i)) + '</span>';
                    break;
                }
                
                // Check for triple-quoted string start (handle in simple way)
                if (line.slice(i, i + 3) === '"""' || line.slice(i, i + 3) === "'''") {
                    const quote = line.slice(i, i + 3);
                    const endIdx = line.indexOf(quote, i + 3);
                    if (endIdx !== -1) {
                        result += '<span class="code-string">' + escapeHtml(line.slice(i, endIdx + 3)) + '</span>';
                        i = endIdx + 3;
                        continue;
                    }
                }
                
                // Check for string
                if (line[i] === '"' || line[i] === "'") {
                    const quote = line[i];
                    let j = i + 1;
                    while (j < line.length && line[j] !== quote) {
                        if (line[j] === '\\') j++; // Skip escaped char
                        j++;
                    }
                    result += '<span class="code-string">' + escapeHtml(line.slice(i, j + 1)) + '</span>';
                    i = j + 1;
                    continue;
                }
                
                // Check for keywords and identifiers
                if (/[a-zA-Z_]/.test(line[i])) {
                    let j = i;
                    while (j < line.length && /[a-zA-Z0-9_]/.test(line[j])) {
                        j++;
                    }
                    const word = line.slice(i, j);
                    const keywords = ['import', 'from', 'class', 'def', 'return', 'self', 'super', 'if', 'else', 'elif', 'for', 'while', 'in', 'and', 'or', 'not', 'True', 'False', 'None', 'as', 'with', 'try', 'except', 'finally', 'raise', 'pass', 'break', 'continue'];
                    const builtins = ['print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set', 'type'];
                    
                    if (keywords.includes(word)) {
                        result += '<span class="code-keyword">' + word + '</span>';
                    } else if (builtins.includes(word)) {
                        result += '<span class="code-function">' + word + '</span>';
                    } else {
                        result += escapeHtml(word);
                    }
                    i = j;
                    continue;
                }
                
                // Check for numbers
                if (/[0-9]/.test(line[i])) {
                    let j = i;
                    while (j < line.length && /[0-9.]/.test(line[j])) {
                        j++;
                    }
                    result += '<span class="code-number">' + line.slice(i, j) + '</span>';
                    i = j;
                    continue;
                }
                
                // Default: just add the character
                result += escapeHtml(line[i]);
                i++;
            }
            
            return result;
        });
        
        return highlightedLines.join('\n');
    }
    
    async copyCodeToClipboard() {
        const btn = document.getElementById('copy-code-btn');
        
        try {
            await navigator.clipboard.writeText(this.generatedCode);
            
            // Show success feedback
            const originalText = btn.innerHTML;
            btn.innerHTML = '✓ Copied!';
            btn.classList.add('copied');
            
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.classList.remove('copied');
            }, 2000);
        } catch (err) {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = this.generatedCode;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            
            const originalText = btn.innerHTML;
            btn.innerHTML = '✓ Copied!';
            btn.classList.add('copied');
            
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.classList.remove('copied');
            }, 2000);
        }
    }
    
    downloadCode() {
        const blob = new Blob([this.generatedCode], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'neural_network.py';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.nnBuilder = new NeuralNetworkBuilder();
});
