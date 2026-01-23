"""
Neural Network Builder Web Application
A Flask-based web app for creating and visualizing feed-forward neural networks interactively.
"""

from flask import Flask, render_template, request, jsonify, Response
import json
from typing import Dict, List, Any, Optional, Tuple

app = Flask(__name__)


class LayerValidator:
    """Validates neural network layer configurations and connections."""
    
    @staticmethod
    def get_output_shape(layer: Dict, input_shape: Optional[Tuple]) -> Optional[Tuple]:
        """Calculate output shape for a layer given its input shape."""
        layer_type = layer.get('type')
        config = layer.get('config', {})
        
        if layer_type == 'input':
            shape_str = config.get('shape', '784')
            try:
                # Parse shape string like "28,28,1" or "784"
                shape = tuple(int(x.strip()) for x in shape_str.split(','))
                return shape
            except:
                return None
                
        if input_shape is None:
            return None
            
        if layer_type == 'dense':
            neurons = config.get('neurons', 128)
            return (neurons,)
            
        elif layer_type == 'output':
            neurons = config.get('neurons', 10)
            return (neurons,)
            
        elif layer_type == 'conv2d':
            if len(input_shape) < 2:
                return None
            
            # Input shape: (H, W, C) or (H, W)
            h, w = input_shape[0], input_shape[1]
            filters = config.get('filters', 32)
            kernel_size = config.get('kernel_size', 3)
            stride = config.get('stride', 1)
            padding = config.get('padding', 'valid')
            
            if padding == 'same':
                out_h = (h + stride - 1) // stride
                out_w = (w + stride - 1) // stride
            else:  # valid
                out_h = (h - kernel_size) // stride + 1
                out_w = (w - kernel_size) // stride + 1
                
            if out_h <= 0 or out_w <= 0:
                return None
            return (out_h, out_w, filters)
            
        elif layer_type == 'pooling':
            if len(input_shape) < 2:
                return None
                
            h, w = input_shape[0], input_shape[1]
            channels = input_shape[2] if len(input_shape) > 2 else 1
            pool_size = config.get('pool_size', 2)
            stride = config.get('stride', 2)
            
            out_h = (h - pool_size) // stride + 1
            out_w = (w - pool_size) // stride + 1
            
            if out_h <= 0 or out_w <= 0:
                return None
            return (out_h, out_w, channels)
            
        elif layer_type == 'flatten':
            # Flatten all dimensions
            total = 1
            for dim in input_shape:
                total *= dim
            return (total,)
            
        elif layer_type == 'reshape':
            # Parse target shape
            target_shape_str = config.get('target_shape', '')
            try:
                target_shape = [int(x.strip()) if x.strip() != '-1' else -1 
                               for x in target_shape_str.split(',')]
            except:
                return None
            
            # Calculate total elements in input
            input_total = 1
            for dim in input_shape:
                input_total *= dim
            
            # Handle -1 (automatic dimension)
            neg_one_idx = None
            known_total = 1
            for i, dim in enumerate(target_shape):
                if dim == -1:
                    if neg_one_idx is not None:
                        return None  # Only one -1 allowed
                    neg_one_idx = i
                else:
                    known_total *= dim
            
            if neg_one_idx is not None:
                if input_total % known_total != 0:
                    return None
                target_shape[neg_one_idx] = input_total // known_total
            
            # Validate total elements match
            output_total = 1
            for dim in target_shape:
                output_total *= dim
            
            if output_total != input_total:
                return None
                
            return tuple(target_shape)
        
        elif layer_type == 'concatenate':
            # Concatenate is handled specially with multiple inputs
            # This method handles single input; multi-input handled in validate()
            return input_shape
            
        elif layer_type == 'dropout':
            return input_shape
            
        elif layer_type == 'batchnorm':
            return input_shape
            
        elif layer_type == 'activation':
            return input_shape
            
        return None
    
    @staticmethod
    def validate_connection(from_layer: Dict, to_layer: Dict, from_shape: Tuple) -> Tuple[bool, str]:
        """Validate if two layers can be connected."""
        to_type = to_layer.get('type')
        
        if to_type == 'input':
            return False, "Cannot connect to an input layer"
            
        if to_type == 'dense':
            # Dense layers can accept any shape (will be flattened)
            return True, ""
            
        if to_type == 'output':
            # Output layers can accept any shape (will be flattened)
            return True, ""
            
        if to_type == 'conv2d':
            if len(from_shape) < 2:
                return False, f"Conv2D requires 2D+ input, got shape {from_shape}"
            return True, ""
            
        if to_type == 'pooling':
            if len(from_shape) < 2:
                return False, f"Pooling requires 2D+ input, got shape {from_shape}"
            return True, ""
            
        if to_type == 'flatten':
            return True, ""
            
        if to_type == 'reshape':
            return True, ""
        
        if to_type == 'concatenate':
            return True, ""
            
        if to_type == 'dropout':
            return True, ""
            
        if to_type == 'batchnorm':
            return True, ""
            
        if to_type == 'activation':
            return True, ""
            
        return True, ""


class NetworkValidator:
    """Validates entire neural network structure."""

    def expand_custom_layers(self, layers: List[Dict], connections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Expand custom layers into their sub-layers and connections."""
        expanded_layers = []
        expanded_connections = []
        layer_id_mapping = {}
        global_counter = 0

        for layer in layers:
            if layer.get('type') == 'custom':
                # Expand custom layer
                config = layer.get('config', {})
                sub_layers = config.get('subLayers', {})
                sub_connections = config.get('subConnections', [])
                custom_id = layer['id']

                # Create mapping for sub-layer IDs
                sub_mapping = {}
                for sub_id, sub_layer in sub_layers.items():
                    new_id = f"{custom_id}_{sub_id}_{global_counter}"
                    global_counter += 1
                    sub_mapping[sub_id] = new_id
                    expanded_layer = dict(sub_layer)
                    expanded_layer['id'] = new_id
                    expanded_layers.append(expanded_layer)
                    layer_id_mapping[new_id] = custom_id  # Track which custom block it came from

                # Add sub-connections
                for conn in sub_connections:
                    expanded_connections.append({
                        'from': sub_mapping[conn['from']],
                        'to': sub_mapping[conn['to']],
                        'fromPort': conn.get('fromPort', 'output'),
                        'toPort': conn.get('toPort', 'input')
                    })

                # Connect external connections to the custom block's ports
                input_ports = config.get('inputPorts', 0)
                output_ports = config.get('outputPorts', 0)

                # Find the input layer in sub-layers if hasInputLayer
                if config.get('hasInputLayer'):
                    input_layer_id = None
                    for sub_id, sub_layer in sub_layers.items():
                        if sub_layer['type'] == 'input':
                            input_layer_id = sub_mapping[sub_id]
                            break
                    # Connect external incoming to this input layer
                    # But since it's expanded, the connections will be handled below

                # For external connections to the custom block
                # We need to map them to the appropriate sub-layers
                # This is tricky, but for now, assume single input/output

            else:
                expanded_layers.append(layer)

        # Now handle external connections
        for conn in connections:
            from_layer = next((l for l in expanded_layers if l['id'] == conn['from']), None)
            to_layer = next((l for l in expanded_layers if l['id'] == conn['to']), None)

            if from_layer and to_layer:
                # Both are regular layers
                expanded_connections.append(conn)
            elif to_layer and not from_layer:
                # Connection to expanded custom block
                # Find the custom layer it was connected to
                custom_id = conn['to']
                custom_layer = next((l for l in layers if l['id'] == custom_id), None)
                if custom_layer and custom_layer.get('type') == 'custom':
                    config = custom_layer.get('config', {})
                    if config.get('hasInputLayer'):
                        # Find input sub-layer
                        for exp_layer in expanded_layers:
                            if exp_layer['id'].startswith(f"{custom_id}_") and exp_layer['type'] == 'input':
                                expanded_connections.append({
                                    'from': conn['from'],
                                    'to': exp_layer['id'],
                                    'fromPort': conn.get('fromPort', 'output'),
                                    'toPort': conn.get('toPort', 'input')
                                })
                                break
                    else:
                        # Map to port number
                        port_num = int(conn.get('toPort', 'input_0').split('_')[1]) if 'input_' in conn.get('toPort', '') else 0
                        # Find the sub-layer that should receive this input
                        # This is simplified - assume first non-input layer or something
                        # Actually, need better mapping
                        pass  # TODO: implement proper port mapping
            elif from_layer and not to_layer:
                # Connection from expanded custom block
                custom_id = conn['from']
                custom_layer = next((l for l in layers if l['id'] == custom_id), None)
                if custom_layer and custom_layer.get('type') == 'custom':
                    config = custom_layer.get('config', {})
                    if config.get('hasOutputLayer'):
                        # Find output sub-layer
                        for exp_layer in expanded_layers:
                            if exp_layer['id'].startswith(f"{custom_id}_") and exp_layer['type'] == 'output':
                                expanded_connections.append({
                                    'from': exp_layer['id'],
                                    'to': conn['to'],
                                    'fromPort': conn.get('fromPort', 'output'),
                                    'toPort': conn.get('toPort', 'input')
                                })
                                break
                    else:
                        # Map from port number
                        pass  # TODO

        return expanded_layers, expanded_connections

    def __init__(self, layers: List[Dict], connections: List[Dict]):
        self.layers = {layer['id']: layer for layer in layers}
        self.layers_list = layers
        self.connections = connections
        self.layer_validator = LayerValidator()
        
    def build_graph(self) -> Dict[str, List[str]]:
        """Build adjacency list from connections."""
        graph = {layer_id: [] for layer_id in self.layers}
        for conn in self.connections:
            from_id = conn['from']
            to_id = conn['to']
            if from_id in graph:
                graph[from_id].append(to_id)
        return graph
    
    def get_incoming(self) -> Dict[str, List[Dict]]:
        """Get incoming connections for each layer with port info."""
        incoming = {layer_id: [] for layer_id in self.layers}
        for conn in self.connections:
            from_id = conn['from']
            to_id = conn['to']
            to_port = conn.get('toPort', 'input')
            if to_id in incoming:
                incoming[to_id].append({'from': from_id, 'toPort': to_port})
        return incoming
    
    def get_incoming_simple(self) -> Dict[str, List[str]]:
        """Get incoming layer IDs for each layer (for backward compatibility)."""
        incoming = {layer_id: [] for layer_id in self.layers}
        for conn in self.connections:
            from_id = conn['from']
            to_id = conn['to']
            if to_id in incoming:
                incoming[to_id].append(from_id)
        return incoming
    
    def topological_sort(self) -> Tuple[bool, List[str], str]:
        """Perform topological sort to detect cycles and get execution order."""
        graph = self.build_graph()
        incoming_simple = self.get_incoming_simple()
        
        # Find layers with no incoming connections
        queue = [layer_id for layer_id, inc in incoming_simple.items() 
                 if len(inc) == 0]
        
        result = []
        visited = set()
        
        # Track in-degree for proper topological sort
        in_degree = {layer_id: len(inc) for layer_id, inc in incoming_simple.items()}
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in visited:
                    queue.append(neighbor)
        
        if len(result) != len(self.layers):
            return False, [], "Network contains a cycle"
            
        return True, result, ""
    
    def validate(self) -> Tuple[bool, List[Dict], Dict[str, Tuple]]:
        """
        Validate the entire network.
        Returns: (is_valid, errors, layer_shapes)
        """
        errors = []
        layer_shapes = {}

        # Check if there are custom layers
        has_custom = any(layer.get('type') == 'custom' for layer in self.layers_list)

        if has_custom:
            # Expand custom layers
            expanded_layers, expanded_connections = self.expand_custom_layers(self.layers_list, self.connections)
            # Create a temporary validator for expanded network
            temp_validator = NetworkValidator(expanded_layers, expanded_connections)
            return temp_validator.validate()
        else:
            # Normal validation without expansion
            if not self.layers:
                errors.append({'type': 'error', 'message': 'No layers in network'})
                return False, errors, {}

            # Check for cycles and get topological order
            is_acyclic, order, cycle_error = self.topological_sort()
            if not is_acyclic:
                errors.append({'type': 'error', 'message': cycle_error})
                return False, errors, {}

            # Find input layers
            incoming = self.get_incoming()
            input_layers = [lid for lid, layer in self.layers.items()
                           if layer.get('type') == 'input']

            if not input_layers:
                # Check for layers with no incoming connections
                root_layers = [lid for lid, inc in incoming.items() if len(inc) == 0]
                if not root_layers:
                    errors.append({'type': 'error', 'message': 'No input layer found'})
                    return False, errors, {}

            # Validate each layer in topological order
            for layer_id in order:
                layer = self.layers[layer_id]
                inc = incoming.get(layer_id, [])

                if layer.get('type') == 'input':
                    shape = self.layer_validator.get_output_shape(layer, None)
                    if shape is None:
                        errors.append({
                            'type': 'error',
                            'message': f"Invalid input shape for layer {layer.get('name', layer_id)}",
                            'layer_id': layer_id
                        })
                    else:
                        layer_shapes[layer_id] = shape
                elif len(inc) == 0:
                    errors.append({
                        'type': 'warning',
                        'message': f"Layer {layer.get('name', layer_id)} has no input connection",
                        'layer_id': layer_id
                    })
                elif layer.get('type') == 'concatenate':
                    # Handle concatenate layer with multiple inputs
                    config = layer.get('config', {})
                    axis = config.get('axis', -1)

                    # Get shapes from all input connections
                    input_shapes = []
                    for conn_info in inc:
                        input_layer_id = conn_info['from']
                        if input_layer_id in layer_shapes:
                            input_shapes.append(layer_shapes[input_layer_id])

                    if len(input_shapes) < 2:
                        errors.append({
                            'type': 'warning',
                            'message': f"Concatenate layer {layer.get('name', layer_id)} needs 2 inputs, has {len(input_shapes)}",
                            'layer_id': layer_id
                        })
                        if len(input_shapes) == 1:
                            layer_shapes[layer_id] = input_shapes[0]
                    else:
                        # Validate shapes are compatible for concatenation
                        shape1 = list(input_shapes[0])
                        shape2 = list(input_shapes[1])

                        # Normalize axis
                        concat_axis = axis if axis >= 0 else len(shape1) + axis

                        if len(shape1) != len(shape2):
                            errors.append({
                                'type': 'error',
                                'message': f"Concatenate inputs must have same number of dimensions",
                                'layer_id': layer_id
                            })
                        elif concat_axis < 0 or concat_axis >= len(shape1):
                            errors.append({
                                'type': 'error',
                                'message': f"Invalid concatenation axis {axis}",
                                'layer_id': layer_id
                            })
                        else:
                            # Check all dims except concat axis match
                            valid = True
                            for i in range(len(shape1)):
                                if i != concat_axis and shape1[i] != shape2[i]:
                                    errors.append({
                                        'type': 'error',
                                        'message': f"Shapes {tuple(shape1)} and {tuple(shape2)} not compatible for concat on axis {axis}",
                                        'layer_id': layer_id
                                    })
                                    valid = False
                                    break

                            if valid:
                                # Calculate output shape
                                output_shape = list(shape1)
                                output_shape[concat_axis] = shape1[concat_axis] + shape2[concat_axis]
                                layer_shapes[layer_id] = tuple(output_shape)
                else:
                    # Get input shape from connected layer
                    input_layer_id = inc[0]['from']  # Take first connection
                    if input_layer_id in layer_shapes:
                        input_shape = layer_shapes[input_layer_id]

                        # Validate connection
                        is_valid, error = self.layer_validator.validate_connection(
                            self.layers[input_layer_id], layer, input_shape
                        )
                        if not is_valid:
                            errors.append({
                                'type': 'error',
                                'message': error,
                                'layer_id': layer_id
                            })
                            continue

                        # Calculate output shape
                        output_shape = self.layer_validator.get_output_shape(layer, input_shape)
                        if output_shape is None:
                            errors.append({
                                'type': 'error',
                                'message': f"Invalid configuration for layer {layer.get('name', layer_id)}",
                                'layer_id': layer_id
                            })
                        else:
                            layer_shapes[layer_id] = output_shape

            # Check for disconnected layers
            graph = self.build_graph()
            incoming_simple = self.get_incoming_simple()
            for layer_id, layer in self.layers.items():
                if layer.get('type') != 'input':
                    if len(incoming_simple.get(layer_id, [])) == 0:
                        if layer_id not in [e.get('layer_id') for e in errors]:
                            errors.append({
                                'type': 'warning',
                                'message': f"Layer {layer.get('name', layer_id)} is disconnected",
                                'layer_id': layer_id
                            })

            has_errors = any(e['type'] == 'error' for e in errors)
            return not has_errors, errors, layer_shapes


class PyTorchCodeGenerator:
    """Generates PyTorch code from network definition."""

    def expand_custom_layers(self, layers: List[Dict], connections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Expand custom layers into their sub-layers and connections."""
        expanded_layers = []
        expanded_connections = []
        global_counter = 0

        for layer in layers:
            if layer.get('type') == 'custom':
                config = layer.get('config', {})
                sub_layers = config.get('subLayers', {})
                sub_connections = config.get('subConnections', [])
                custom_id = layer['id']

                sub_mapping = {}
                for sub_id, sub_layer in sub_layers.items():
                    new_id = f"{custom_id}_{sub_id}_{global_counter}"
                    global_counter += 1
                    sub_mapping[sub_id] = new_id
                    expanded_layer = dict(sub_layer)
                    expanded_layer['id'] = new_id
                    expanded_layers.append(expanded_layer)

                for conn in sub_connections:
                    expanded_connections.append({
                        'from': sub_mapping[conn['from']],
                        'to': sub_mapping[conn['to']],
                        'fromPort': conn.get('fromPort', 'output'),
                        'toPort': conn.get('toPort', 'input')
                    })

                # Handle external connections
                input_ports = config.get('inputPorts', 0)
                output_ports = config.get('outputPorts', 0)

                # Find external connections to/from this custom block
                for conn in connections:
                    if conn['to'] == custom_id:
                        if config.get('hasInputLayer'):
                            # Connect to input sub-layer
                            input_sub_id = next((sid for sid, sl in sub_layers.items() if sl['type'] == 'input'), None)
                            if input_sub_id:
                                expanded_connections.append({
                                    'from': conn['from'],
                                    'to': sub_mapping[input_sub_id],
                                    'fromPort': conn.get('fromPort', 'output'),
                                    'toPort': conn.get('toPort', 'input')
                                })
                        else:
                            # Map port number to first layer that needs input
                            port_num = 0
                            if 'input_' in str(conn.get('toPort', '')):
                                port_num = int(conn['toPort'].split('_')[1])
                            # Assume the first non-input layer
                            non_input_subs = [sid for sid, sl in sub_layers.items() if sl['type'] != 'input']
                            if port_num < len(non_input_subs):
                                target_sub = non_input_subs[port_num]
                                expanded_connections.append({
                                    'from': conn['from'],
                                    'to': sub_mapping[target_sub],
                                    'fromPort': conn.get('fromPort', 'output'),
                                    'toPort': 'input'
                                })

                    elif conn['from'] == custom_id:
                        if config.get('hasOutputLayer'):
                            # Connect from output sub-layer
                            output_sub_id = next((sid for sid, sl in sub_layers.items() if sl['type'] == 'output'), None)
                            if output_sub_id:
                                expanded_connections.append({
                                    'from': sub_mapping[output_sub_id],
                                    'to': conn['to'],
                                    'fromPort': 'output',
                                    'toPort': conn.get('toPort', 'input')
                                })
                        else:
                            # Map from port number
                            port_num = 0
                            if 'output_' in str(conn.get('fromPort', '')):
                                port_num = int(conn['fromPort'].split('_')[1])
                            # Assume from the last layer or something
                            # Simplified: from the layer that has no outgoing internal connections
                            candidates = []
                            for sid in sub_layers:
                                has_outgoing = any(c['from'] == sid for c in sub_connections)
                                if not has_outgoing:
                                    candidates.append(sid)
                            if port_num < len(candidates):
                                source_sub = candidates[port_num]
                                expanded_connections.append({
                                    'from': sub_mapping[source_sub],
                                    'to': conn['to'],
                                    'fromPort': 'output',
                                    'toPort': conn.get('toPort', 'input')
                                })
            else:
                expanded_layers.append(layer)

        # Add non-custom connections
        for conn in connections:
            from_custom = any(l['id'] == conn['from'] and l.get('type') == 'custom' for l in layers)
            to_custom = any(l['id'] == conn['to'] and l.get('type') == 'custom' for l in layers)
            if not from_custom and not to_custom:
                expanded_connections.append(conn)

        return expanded_layers, expanded_connections

    def __init__(self, layers: List[Dict], connections: List[Dict], layer_shapes: Dict[str, Tuple]):
        # Expand custom layers
        expanded_layers, expanded_connections = self.expand_custom_layers(layers, connections)

        self.layers = {layer['id']: layer for layer in expanded_layers}
        self.layers_list = expanded_layers
        self.connections = expanded_connections
        self.layer_shapes = layer_shapes  # Note: shapes may need updating for expanded layers
        
    def generate(self) -> str:
        """Generate PyTorch model code."""
        lines = [
            '"""',
            'PyTorch Neural Network Model',
            'Generated by Neural Network Builder',
            '"""',
            '',
            'import torch',
            'import torch.nn as nn',
            'import torch.nn.functional as F',
            '',
            '',
            'class NeuralNetwork(nn.Module):',
            '    def __init__(self):',
            '        super(NeuralNetwork, self).__init__()',
            ''
        ]
        
        # Build execution order
        validator = NetworkValidator(self.layers_list, self.connections)
        is_acyclic, order, _ = validator.topological_sort()
        incoming = validator.get_incoming()
        
        if not is_acyclic:
            return "# Error: Network contains cycles"
        
        # Identify layers that need intermediate variables:
        # 1. Layers that feed into concatenate
        # 2. Layers that have multiple outgoing connections (branching)
        needs_save = set()
        
        # Layers feeding into concatenate
        for layer_id, layer in self.layers.items():
            if layer.get('type') == 'concatenate':
                for conn_info in incoming.get(layer_id, []):
                    needs_save.add(conn_info['from'])
        
        # Layers with multiple outgoing connections (branching)
        outgoing_count = {}
        for conn in self.connections:
            from_id = conn['from']
            outgoing_count[from_id] = outgoing_count.get(from_id, 0) + 1
        for layer_id, count in outgoing_count.items():
            if count > 1:
                needs_save.add(layer_id)
        
        concat_inputs = needs_save  # For backward compatibility
        
        # Generate layer definitions
        layer_vars = {}
        var_counter = {'dense': 0, 'conv': 0, 'pool': 0, 'bn': 0, 'dropout': 0, 'output': 0}
        
        prev_shape = None
        for layer_id in order:
            layer = self.layers[layer_id]
            layer_type = layer.get('type')
            config = layer.get('config', {})
            
            if layer_type == 'input':
                prev_shape = self.layer_shapes.get(layer_id)
                layer_vars[layer_id] = 'x'
                continue
            
            # Get input shape
            inc = incoming.get(layer_id, [])
            if inc:
                input_shape = self.layer_shapes.get(inc[0]['from'])
            else:
                input_shape = prev_shape
                
            if layer_type == 'dense':
                var_name = f'fc{var_counter["dense"]}'
                var_counter['dense'] += 1
                
                in_features = 1
                if input_shape:
                    for dim in input_shape:
                        in_features *= dim
                out_features = config.get('neurons', 128)
                
                lines.append(f'        self.{var_name} = nn.Linear({in_features}, {out_features})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'output':
                var_name = f'output{var_counter["output"]}'
                var_counter['output'] += 1
                
                in_features = 1
                if input_shape:
                    for dim in input_shape:
                        in_features *= dim
                out_features = config.get('neurons', 10)
                
                lines.append(f'        self.{var_name} = nn.Linear({in_features}, {out_features})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'conv2d':
                var_name = f'conv{var_counter["conv"]}'
                var_counter['conv'] += 1
                
                in_channels = input_shape[2] if input_shape and len(input_shape) > 2 else 1
                out_channels = config.get('filters', 32)
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 1)
                padding = 0 if config.get('padding', 'valid') == 'valid' else kernel_size // 2
                
                lines.append(f'        self.{var_name} = nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'pooling':
                var_name = f'pool{var_counter["pool"]}'
                var_counter['pool'] += 1
                
                pool_type = config.get('pool_type', 'max')
                pool_size = config.get('pool_size', 2)
                stride = config.get('stride', 2)
                
                if pool_type == 'max':
                    lines.append(f'        self.{var_name} = nn.MaxPool2d(kernel_size={pool_size}, stride={stride})')
                else:
                    lines.append(f'        self.{var_name} = nn.AvgPool2d(kernel_size={pool_size}, stride={stride})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'batchnorm':
                var_name = f'bn{var_counter["bn"]}'
                var_counter['bn'] += 1
                
                if input_shape and len(input_shape) > 2:
                    num_features = input_shape[2]
                    lines.append(f'        self.{var_name} = nn.BatchNorm2d({num_features})')
                else:
                    num_features = input_shape[0] if input_shape else 128
                    lines.append(f'        self.{var_name} = nn.BatchNorm1d({num_features})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'dropout':
                var_name = f'dropout{var_counter["dropout"]}'
                var_counter['dropout'] += 1
                
                rate = config.get('rate', 0.5)
                lines.append(f'        self.{var_name} = nn.Dropout(p={rate})')
                layer_vars[layer_id] = var_name
                
            elif layer_type == 'flatten':
                layer_vars[layer_id] = 'flatten'
                
            elif layer_type == 'reshape':
                layer_vars[layer_id] = 'reshape'
            
            elif layer_type == 'concatenate':
                layer_vars[layer_id] = 'concatenate'
                
            elif layer_type == 'activation':
                layer_vars[layer_id] = 'activation'
            
            prev_shape = self.layer_shapes.get(layer_id, input_shape)
        
        # Generate forward method
        lines.extend([
            '',
            '    def forward(self, x):',
        ])
        
        # Track which saved variable to use for each layer
        layer_input_var = {}  # Maps layer_id to the variable name it should read from
        
        for layer_id in order:
            layer = self.layers[layer_id]
            layer_type = layer.get('type')
            config = layer.get('config', {})
            var_name = layer_vars.get(layer_id, '')
            
            if layer_type == 'input':
                # Reshape input if needed - convert from (H, W, C) to PyTorch (C, H, W)
                shape = self.layer_shapes.get(layer_id)
                if shape and len(shape) > 1:
                    if len(shape) == 2:
                        lines.append(f'        # Input shape: {shape} (H, W) -> (1, H, W) for PyTorch')
                        lines.append(f'        x = x.view(-1, 1, {shape[0]}, {shape[1]})')
                    elif len(shape) == 3:
                        lines.append(f'        # Input shape: {shape} (H, W, C) -> (C, H, W) for PyTorch')
                        lines.append(f'        x = x.view(-1, {shape[2]}, {shape[0]}, {shape[1]})')
                # Save input if it branches
                if layer_id in needs_save:
                    var_suffix = layer_id.replace('layer_', '')
                    lines.append(f'        x_{var_suffix} = x')
                continue
                
            # Check if we need to restore from a saved variable
            inc = incoming.get(layer_id, [])
            if inc and layer_type != 'concatenate':
                input_layer_id = inc[0]['from']
                if input_layer_id in needs_save:
                    var_suffix = input_layer_id.replace('layer_', '')
                    lines.append(f'        x = x_{var_suffix}')
            
            # Save intermediate variable if this layer feeds into concatenate or branches
            save_var = layer_id in needs_save
            
            if layer_type == 'dense':
                # Check if we need to flatten first
                inc = incoming.get(layer_id, [])
                if inc:
                    input_shape = self.layer_shapes.get(inc[0]['from'])
                    if input_shape and len(input_shape) > 1:
                        lines.append('        x = x.view(x.size(0), -1)')
                
                activation = config.get('activation', 'relu')
                lines.append(f'        x = self.{var_name}(x)')
                if activation and activation != 'none':
                    if activation == 'relu':
                        lines.append('        x = F.relu(x)')
                    elif activation == 'sigmoid':
                        lines.append('        x = torch.sigmoid(x)')
                    elif activation == 'tanh':
                        lines.append('        x = torch.tanh(x)')
                    elif activation == 'softmax':
                        lines.append('        x = F.softmax(x, dim=1)')
                    elif activation == 'leaky_relu':
                        lines.append('        x = F.leaky_relu(x)')
                        
            elif layer_type == 'output':
                # Check if we need to flatten first
                inc = incoming.get(layer_id, [])
                if inc:
                    input_shape = self.layer_shapes.get(inc[0]['from'])
                    if input_shape and len(input_shape) > 1:
                        lines.append('        x = x.view(x.size(0), -1)')
                
                activation = config.get('activation', 'softmax')
                lines.append(f'        x = self.{var_name}(x)')
                if activation and activation != 'none':
                    if activation == 'relu':
                        lines.append('        x = F.relu(x)')
                    elif activation == 'sigmoid':
                        lines.append('        x = torch.sigmoid(x)')
                    elif activation == 'tanh':
                        lines.append('        x = torch.tanh(x)')
                    elif activation == 'softmax':
                        lines.append('        x = F.softmax(x, dim=1)')
                    elif activation == 'leaky_relu':
                        lines.append('        x = F.leaky_relu(x)')
                        
            elif layer_type == 'conv2d':
                activation = config.get('activation', 'relu')
                lines.append(f'        x = self.{var_name}(x)')
                if activation and activation != 'none':
                    if activation == 'relu':
                        lines.append('        x = F.relu(x)')
                    elif activation == 'sigmoid':
                        lines.append('        x = torch.sigmoid(x)')
                    elif activation == 'tanh':
                        lines.append('        x = torch.tanh(x)')
                    elif activation == 'leaky_relu':
                        lines.append('        x = F.leaky_relu(x)')
                        
            elif layer_type == 'pooling':
                lines.append(f'        x = self.{var_name}(x)')
                
            elif layer_type == 'batchnorm':
                lines.append(f'        x = self.{var_name}(x)')
                
            elif layer_type == 'dropout':
                lines.append(f'        x = self.{var_name}(x)')
                
            elif layer_type == 'flatten':
                lines.append('        x = x.view(x.size(0), -1)')
                
            elif layer_type == 'reshape':
                target_shape = self.layer_shapes.get(layer_id)
                if target_shape:
                    # For 3D shapes (H, W, C), convert to PyTorch format (C, H, W)
                    if len(target_shape) == 3:
                        h, w, c = target_shape
                        lines.append(f'        x = x.view(-1, {c}, {h}, {w})')
                    elif len(target_shape) == 2:
                        lines.append(f'        x = x.view(-1, 1, {target_shape[0]}, {target_shape[1]})')
                    else:
                        shape_str = ', '.join(str(d) for d in target_shape)
                        lines.append(f'        x = x.view(-1, {shape_str})')
            
            elif layer_type == 'concatenate':
                # Get the two input layers
                inc = incoming.get(layer_id, [])
                if len(inc) >= 2:
                    # Sort by port to get consistent ordering
                    sorted_inc = sorted(inc, key=lambda x: x.get('toPort', 'input'))
                    input_vars = []
                    for conn_info in sorted_inc[:2]:
                        input_id = conn_info['from']
                        var_suffix = input_id.replace('layer_', '')
                        input_vars.append(f'x_{var_suffix}')
                    
                    axis = config.get('axis', -1)
                    # PyTorch dim is different - for (N, C, H, W), dim 1 is channels
                    # We'll use dim=1 for concatenation (channel dimension for 4D, feature dim for 2D)
                    dim = 1 if axis == -1 else axis + 1  # +1 for batch dimension
                    lines.append(f'        x = torch.cat([{", ".join(input_vars)}], dim={dim})')
                
            elif layer_type == 'activation':
                activation = config.get('activation', 'relu')
                if activation == 'relu':
                    lines.append('        x = F.relu(x)')
                elif activation == 'sigmoid':
                    lines.append('        x = torch.sigmoid(x)')
                elif activation == 'tanh':
                    lines.append('        x = torch.tanh(x)')
                elif activation == 'softmax':
                    lines.append('        x = F.softmax(x, dim=1)')
                elif activation == 'leaky_relu':
                    lines.append('        x = F.leaky_relu(x)')
            
            # Save intermediate variable if this layer feeds into concatenate
            if save_var and layer_type != 'input':
                var_suffix = layer_id.replace('layer_', '')
                lines.append(f'        x_{var_suffix} = x')
        
        lines.extend([
            '        return x',
            '',
            '',
            '# Create model instance',
            'model = NeuralNetwork()',
            'print(model)',
            '',
            '# Example usage:',
            '# input_tensor = torch.randn(batch_size, input_features)',
            '# output = model(input_tensor)',
        ])
        
        return '\n'.join(lines)


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/api/validate', methods=['POST'])
def validate_network():
    """Validate the neural network configuration."""
    data = request.get_json()
    layers = data.get('layers', [])
    connections = data.get('connections', [])

    validator = NetworkValidator(layers, connections)
    is_valid, errors, layer_shapes = validator.validate()

    # Map errors and shapes back to original layer ids for custom blocks
    for error in errors:
        if 'layer_id' in error:
            lid = error['layer_id']
            if '_sub_' in lid:
                custom_id = lid.split('_sub_')[0]
                error['layer_id'] = custom_id
                # Update message to indicate it's from a custom block
                if 'has no input connection' in error['message']:
                    error['message'] = f"Custom block '{layers_dict.get(custom_id, {}).get('name', custom_id)}' contains unconnected layers"

    # Map shapes back to custom block ids
    mapped_shapes = {}
    for lid, shape in layer_shapes.items():
        if '_sub_' in lid:
            custom_id = lid.split('_sub_')[0]
            mapped_shapes[custom_id] = list(shape)
        else:
            mapped_shapes[lid] = list(shape)

    return jsonify({
        'valid': is_valid,
        'errors': errors,
        'shapes': mapped_shapes
    })


@app.route('/api/export', methods=['POST'])
def export_pytorch():
    """Export the network as PyTorch code."""
    data = request.get_json()
    layers = data.get('layers', [])
    connections = data.get('connections', [])
    
    # Validate first
    validator = NetworkValidator(layers, connections)
    is_valid, errors, layer_shapes = validator.validate()
    
    if not is_valid:
        return jsonify({
            'success': False,
            'errors': errors
        }), 400
    
    # Generate code
    generator = PyTorchCodeGenerator(layers, connections, layer_shapes)
    code = generator.generate()
    
    return Response(
        code,
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment; filename=neural_network.py'}
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
