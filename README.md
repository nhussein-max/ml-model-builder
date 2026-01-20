# Neural Network Builder

An interactive web application for creating and visualizing feed-forward neural networks. Build your network visually by dragging and connecting layer blocks, then export the model as PyTorch code.

![Neural Network Builder](https://via.placeholder.com/800x400?text=Neural+Network+Builder)

## Features

### 1. Draggable Layer Blocks

**Input / Output:**
- **Input Layer**: Define the input shape with preset options (MNIST, CIFAR, ImageNet, etc.) or custom shapes
- **Output Layer**: Configure output units and activation (Softmax, Sigmoid, None for regression)

**Core Layers:**
- **Dense (Fully Connected)**: Configure neurons, activation functions, and bias
- **Conv2D**: Set filters, kernel size, stride, padding, and activation
- **Pooling**: Choose between Max and Average pooling with configurable size and stride

**Utility:**
- **Flatten**: Convert multi-dimensional tensors to 1D
- **Reshape**: Reshape tensors to any compatible shape (supports -1 for auto-dimension)
- **Activation**: Standalone activation layers (ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU)

**Regularization:**
- **Dropout**: Add regularization with configurable dropout rate
- **BatchNorm**: Batch normalization with momentum and epsilon settings

### 2. Visual Network Building
- Drag layers from the sidebar onto the workspace
- Connect layers by clicking and dragging from output ports to input ports
- **Delete layers**: Hover over a layer and click the × button in the corner
- **Delete connections**: Hover over a connection to reveal the delete button
- Move layers around to organize your network visually

### 3. Layer Configuration
- Click on any layer to open its configuration panel
- Configure layer-specific parameters:
  - **Input**: Shape presets (MNIST, CIFAR, ImageNet) or custom shape
  - **Output**: Number of units, output activation
  - **Dense**: Number of neurons, activation function, use bias
  - **Conv2D**: Number of filters, kernel size, stride, padding, activation
  - **Pooling**: Pool type (max/avg), pool size, stride
  - **Reshape**: Target shape with optional auto-dimension (-1)
  - **Dropout**: Dropout rate
  - **BatchNorm**: Momentum, epsilon
  - **Activation**: Activation function type

### 4. Network Validation
- Real-time validation of network architecture
- Checks for:
  - Proper layer connections
  - Shape compatibility between layers
  - Cycle detection
  - Disconnected layers
- Visual feedback with error highlighting on problematic layers
- Output shape calculation and display for each layer

### 5. PyTorch Code Export
- Export your network as a ready-to-use PyTorch Python script
- Generated code includes:
  - Proper layer definitions in `__init__`
  - Forward pass implementation
  - Correct input/output shape handling
  - Activation functions
  - Example usage comments

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Creating a Simple Neural Network

1. **Add an Input Layer**
   - Drag the "Input" block from the sidebar to the workspace
   - Click on it to configure the input shape (e.g., "784" for MNIST or "28,28,1" for 2D input)

2. **Add Hidden Layers**
   - Drag "Dense" or other layer types to the workspace
   - Connect layers by clicking on the output port (right side) of one layer and dragging to the input port (left side) of another

3. **Configure Layers**
   - Click on any layer to open the configuration panel
   - Adjust parameters like neuron count, activation functions, etc.

4. **Validate Your Network**
   - Click "Validate Network" to check for errors
   - The validation panel will show any issues
   - Layers with errors will be highlighted in red

5. **Export to PyTorch**
   - Once validation passes, click "Export PyTorch"
   - A Python file will be downloaded containing your model

### Example: Simple Classifier

1. Add Input layer (select "MNIST Flat (784)" preset)
2. Add Dense layer (128 neurons, ReLU activation)
3. Add Dropout layer (rate: 0.5)
4. Add Output layer (10 units, Softmax activation)
5. Connect: Input → Dense → Dropout → Output
6. Validate and Export

### Example: Using Reshape

1. Add Input layer (shape: 784)
2. Add Reshape layer (target: 28,28,1 or use -1,28,1 for auto)
3. Add Conv2D layer (32 filters, 3x3 kernel)
4. Continue building your CNN...
5. The Reshape layer converts flat input to 2D for convolution

### Example: CNN for Image Classification

1. Add Input layer (shape: 28,28,1)
2. Add Conv2D layer (32 filters, 3x3 kernel)
3. Add Pooling layer (Max, 2x2)
4. Add Conv2D layer (64 filters, 3x3 kernel)
5. Add Pooling layer (Max, 2x2)
6. Add Flatten layer
7. Add Dense layer (128 neurons, ReLU)
8. Add Dense layer (10 neurons, Softmax)
9. Connect all layers in sequence
10. Validate and Export

## Project Structure

```
zed-base/
├── app.py                 # Flask backend with validation and code generation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main HTML page with layer templates
└── static/
    ├── css/
    │   └── style.css     # Application styling
    └── js/
        └── app.js        # Frontend JavaScript (drag-drop, connections, etc.)
```

## API Endpoints

### POST /api/validate
Validates the network configuration.

**Request Body:**
```json
{
  "layers": [...],
  "connections": [...]
}
```

**Response:**
```json
{
  "valid": true/false,
  "errors": [...],
  "shapes": {...}
}
```

### POST /api/export
Exports the network as PyTorch code.

**Request Body:** Same as validate

**Response:** Python source code file download

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Visualization**: SVG for connection lines
- **Code Generation**: Custom PyTorch code generator

## License

MIT License