import numpy as np
import configparser
from functions import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    mse,
    mse_derivative,
    cross_entropy,
    cross_entropy_derivative,
)
from data_loader import load_wisconsin_breast_cancer_data


class Neuron:
    def __init__(self, index, activation, activation_deriv):
        self.index = index
        self.inputs = []
        self.outputs = []
        self.bias = 0.0
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.result = 0.0
        self.error = 0.0

    def forward_prop(self):
        if not self.inputs:
            return self.result
        total = 0.0
        for in_axon in self.inputs:
            total += in_axon.input.result * in_axon.weight
        total += self.bias
        self.result = self.activation(total)
        return self.result

    def back_prop(self, learning_rate):
        delta = self.error * self.activation_deriv(self.result)
        for in_axon in self.inputs:
            grad = delta * in_axon.input.result
            in_axon.weight -= learning_rate * grad
            in_axon.input.error += delta * in_axon.weight
        self.bias -= learning_rate * delta
        self.error = 0.0


class Axon:
    def __init__(self, in_neuron, out_neuron):
        self.input = in_neuron
        self.output = out_neuron
        self.weight = np.random.randn() * 0.01


class Network:
    def __init__(
        self,
        layer_sizes,
        activation="sigmoid",
        output_activation="sigmoid",
        cost="mse",
        learning_rate=0.1,
        seed=42,
    ):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # Select activation for hidden layers
        if activation == "sigmoid":
            self.hidden_activation = sigmoid
            self.hidden_activation_deriv = sigmoid_derivative
        elif activation == "tanh":
            self.hidden_activation = tanh
            self.hidden_activation_deriv = tanh_derivative
        elif activation == "relu":
            self.hidden_activation = relu
            self.hidden_activation_deriv = relu_derivative
        else:
            raise ValueError("Unsupported hidden activation function")

        # Select activation for output layer
        if output_activation == "sigmoid":
            self.output_activation = sigmoid
            self.output_activation_deriv = sigmoid_derivative
        elif output_activation == "tanh":
            self.output_activation = tanh
            self.output_activation_deriv = tanh_derivative
        elif output_activation == "relu":
            self.output_activation = relu
            self.output_activation_deriv = relu_derivative
        else:
            raise ValueError("Unsupported output activation function")

        # Select cost functions
        if cost == "mse":
            self.cost = mse
            self.cost_deriv = mse_derivative
        elif cost == "cross_entropy":
            self.cost = cross_entropy
            self.cost_deriv = cross_entropy_derivative
        else:
            raise ValueError("Unsupported cost function")

        # Create network layers
        self.layers = []
        for layer_idx, size in enumerate(layer_sizes):
            layer = []
            if layer_idx == len(layer_sizes) - 1:
                # output layer
                act = self.output_activation
                act_deriv = self.output_activation_deriv
            elif layer_idx == 0:
                # input layer
                act = self.hidden_activation
                act_deriv = self.hidden_activation_deriv
            else:
                # hidden layer
                act = self.hidden_activation
                act_deriv = self.hidden_activation_deriv

            for i in range(size):
                neuron = Neuron(i, act, act_deriv)
                neuron.bias = 0.0
                layer.append(neuron)
            self.layers.append(layer)

        # Connect neurons with axons
        for i in range(len(self.layers) - 1):
            for n_out in self.layers[i + 1]:
                for n_in in self.layers[i]:
                    ax = Axon(n_in, n_out)
                    n_out.inputs.append(ax)
                    n_in.outputs.append(ax)

    def forward_prop(self, X):
        outputs = []
        for sample in X:
            for i, neuron in enumerate(self.layers[0]):
                neuron.result = sample[i]

            for layer_idx in range(1, len(self.layers)):
                for neuron in self.layers[layer_idx]:
                    neuron.forward_prop()

            out_layer = self.layers[-1]
            outputs.append([n.result for n in out_layer])
        return np.array(outputs)

    def back_prop(self, X, y):
        n_samples = X.shape[0]
        for layer in self.layers:
            for neuron in layer:
                neuron.error = 0.0

        # Accumulate errors
        for i in range(n_samples):
            # forward single sample
            for idx, neuron in enumerate(self.layers[0]):
                neuron.result = X[i, idx]
            for layer_idx in range(1, len(self.layers)):
                for neuron in self.layers[layer_idx]:
                    neuron.forward_prop()

            outputs = np.array([n.result for n in self.layers[-1]])
            targets = y[i]
            if targets.ndim == 0:
                targets = np.array([targets])
            dC_dOut = self.cost_deriv(outputs, targets)

            for n_idx, neuron in enumerate(self.layers[-1]):
                neuron.error += dC_dOut[n_idx]

        # Update weights
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[layer_idx]:
                neuron.back_prop(self.learning_rate)

    def train(self, X, y, epochs=100, print_interval=10, callback=None):
        for epoch in range(1, epochs + 1):
            outputs = self.forward_prop(X)
            c = self.cost(outputs, y)
            self.back_prop(X, y)
            if epoch % print_interval == 0:
                if callback is not None:
                    callback(epoch, c)
                else:
                    print(f"Epoch {epoch}: Cost = {c:.6f}")

    def predict(self, X):
        outputs = self.forward_prop(X)
        return (outputs > 0.5).astype(np.float32)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


if __name__ == "__main__":
    # If run directly, use config and run from terminal
    config = configparser.ConfigParser()
    config.read("config.ini")
    section = "DEFAULT"
    mode = config[section].get("mode", "dummy")
    data_path = config[section].get("data_path", "data.csv")
    test_size = config[section].getfloat("test_size", 0.2)
    learning_rate = config[section].getfloat("learning_rate", 0.1)
    epochs = config[section].getint("epochs", 100)
    print_interval = config[section].getint("print_interval", 20)

    hidden_layers_str = config[section].get(
        "hidden_layers", "10"
    )  # default one hidden layer of size 10
    hidden_layer_sizes = [
        int(x.strip()) for x in hidden_layers_str.split(",") if x.strip().isdigit()
    ]

    activation = "sigmoid"
    output_activation = "sigmoid"
    cost = "mse"

    if mode == "real":
        X_train, X_test, y_train, y_test, feature_names = (
            load_wisconsin_breast_cancer_data(
                csv_path=data_path, test_size=test_size, random_state=42
            )
        )
        input_dim = X_train.shape[1]
        layer_sizes = [input_dim, 10, 1]
        net = Network(
            layer_sizes=layer_sizes,
            activation=activation,
            output_activation=output_activation,
            cost=cost,
            learning_rate=learning_rate,
            seed=42,
        )
        print("Running in real mode...")
        print("Initial accuracy (test):", net.accuracy(X_test, y_test))
        net.train(
            X_train,
            y_train.reshape(-1, 1),
            epochs=epochs,
            print_interval=print_interval,
        )
        print("Final accuracy (test):", net.accuracy(X_test, y_test))
    else:
        # Dummy mode
        np.random.seed(42)
        X_dummy = np.random.randn(100, 5)
        y_dummy = (np.sum(X_dummy, axis=1) > 0).astype(np.float32).reshape(-1, 1)
        net = Network(
            layer_sizes=[5, 10, 1],
            activation=activation,
            output_activation=output_activation,
            cost=cost,
            learning_rate=learning_rate,
            seed=42,
        )
        print("Running in dummy mode...")
        print("Initial accuracy:", net.accuracy(X_dummy, y_dummy))
        net.train(X_dummy, y_dummy, epochs=epochs, print_interval=print_interval)
        print("Final accuracy:", net.accuracy(X_dummy, y_dummy))
