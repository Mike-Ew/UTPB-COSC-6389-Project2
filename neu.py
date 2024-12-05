import math
import random

num_inputs = 10
num_hidden_layers = 3
hidden_layer_width = 12
num_outputs = 4
learning_rate = 0.1
training_data_size = 100


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, x, y, input_idx=-1, bias=None):
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []
        self.index = input_idx
        if bias is None:
            self.bias = random.uniform(-0.1, 0.1)
        else:
            self.bias = bias
        self.result = None
        self.error = 0.0

    def connect_input(self, in_n):
        in_axon = Axon(in_n, self)
        self.inputs.append(in_axon)

    def connect_output(self, out_n):
        out_axon = Axon(self, out_n)
        self.outputs.append(out_axon)

    def forward_prop(self, inputs):
        if self.result is not None:
            return self.result
        if self.index >= 0:
            # Input neuron
            self.result = inputs[self.index]
        else:
            total = self.bias
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_val = in_n.forward_prop(inputs) * in_axon.weight
                total += in_val
            self.result = sigmoid(total)
        return self.result

    def back_prop(self):
        gradient = self.result * (1.0 - self.result)
        delta = self.error * gradient
        # Update bias
        self.bias -= learning_rate * delta
        # Update weights and propagate error backward
        for in_axon in self.inputs:
            in_n = in_axon.input
            # Accumulate error for the input neuron
            in_n.error += delta * in_axon.weight
            # Update the weight
            in_axon.weight -= learning_rate * delta * in_n.result
        # Reset this neuron's error
        self.error = 0.0

    def reset(self):
        self.result = None
        self.error = 0.0


class Axon:
    def __init__(self, in_n, out_n, weight=None):
        self.input = in_n
        self.output = out_n
        if weight is None:
            self.weight = random.uniform(-0.1, 0.1)
        else:
            self.weight = weight


class Network:
    def __init__(self):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []

        # Initialize input neurons
        for idx in range(num_inputs):
            in_n = Neuron(0, 0, idx)
            self.inputs.append(in_n)

        # Initialize hidden layers
        prev_layer = self.inputs
        for _ in range(num_hidden_layers):
            current_layer = []
            for _ in range(hidden_layer_width):
                neuron = Neuron(0, 0)
                for prev_neuron in prev_layer:
                    neuron.connect_input(prev_neuron)
                    prev_neuron.connect_output(neuron)
                current_layer.append(neuron)
            self.hidden_layers.append(current_layer)
            prev_layer = current_layer

        # Initialize output neurons
        for _ in range(num_outputs):
            out_n = Neuron(0, 0)
            for prev_neuron in prev_layer:
                out_n.connect_input(prev_neuron)
                prev_neuron.connect_output(out_n)
            self.outputs.append(out_n)

    def get_all_neurons(self):
        # Helper method to get all neurons in the network
        neurons = self.inputs.copy()
        for layer in self.hidden_layers:
            neurons.extend(layer)
        neurons.extend(self.outputs)
        return neurons

    def forward_prop(self, inputs):
        # Reset neuron results
        for neuron in self.get_all_neurons():
            neuron.reset()
        # Perform forward propagation
        for out_n in self.outputs:
            out_n.forward_prop(inputs)

    def back_prop(self, target_outputs):
        # Calculate output errors
        for idx, out_n in enumerate(self.outputs):
            out_n.error = out_n.result - target_outputs[idx]
        # Start backpropagation from output neurons
        for out_n in self.outputs:
            out_n.back_prop()

    def train(self, data):
        self.forward_prop(data.inputs)
        self.back_prop(data.outputs)

    def test(self, data):
        self.forward_prop(data.inputs)
        return [out_n.result for out_n in self.outputs]


class RandData:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def generate(self):
        # Generate random inputs
        self.inputs = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        # For a meaningful learning task, let's set the outputs to be a function of inputs
        # For example, the sum of inputs, normalized between 0 and 1 using sigmoid
        total = sum(self.inputs)
        output_value = sigmoid(total)
        self.outputs = [output_value for _ in range(num_outputs)]


def train():
    global training_data_size
    network = Network()
    for i in range(training_data_size):
        dat = RandData()
        dat.generate()
        network.train(dat)
        if (i + 1) % 10 == 0:
            print(f"Training iteration {i + 1}/{training_data_size}")

    # Test the network after training
    test_data = RandData()
    test_data.generate()
    outputs = network.test(test_data)
    print("Test Inputs:", test_data.inputs)
    print("Expected Outputs:", test_data.outputs)
    print("Network Outputs:", outputs)


if __name__ == "__main__":
    train()
