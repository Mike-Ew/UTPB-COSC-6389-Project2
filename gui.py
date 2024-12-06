# gui.py
import tkinter as tk
from tkinter import ttk
import numpy as np
from data_loader import load_wisconsin_breast_cancer_data
from network import Network

ACTIVATIONS = [("Sigmoid", "sigmoid"), ("Tanh", "tanh"), ("ReLU", "relu")]
COSTS = [("MSE", "mse"), ("Cross Entropy", "cross_entropy")]


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network GUI")

        self.mode = tk.StringVar(value="dummy")
        self.hidden_activation = tk.StringVar(value="sigmoid")
        self.output_activation = tk.StringVar(value="sigmoid")
        self.cost_function = tk.StringVar(value="mse")

        self.learning_rate = 0.1
        self.epochs = 50
        self.print_interval = 10

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.net = None

        self.build_controls()
        self.build_canvas()
        self.build_log_area()

    def build_controls(self):
        frame = ttk.Frame(self.master)
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Mode selection
        tk.Label(frame, text="Mode:").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(frame, text="Dummy", variable=self.mode, value="dummy").grid(
            row=0, column=1
        )
        tk.Radiobutton(frame, text="Real", variable=self.mode, value="real").grid(
            row=0, column=2
        )

        # Hidden activation selection
        tk.Label(frame, text="Hidden Activation:").grid(row=1, column=0, sticky="w")
        col_idx = 1
        for name, val in ACTIVATIONS:
            tk.Radiobutton(
                frame, text=name, variable=self.hidden_activation, value=val
            ).grid(row=1, column=col_idx)
            col_idx += 1

        # Output activation selection
        tk.Label(frame, text="Output Activation:").grid(row=2, column=0, sticky="w")
        col_idx = 1
        for name, val in ACTIVATIONS:
            tk.Radiobutton(
                frame, text=name, variable=self.output_activation, value=val
            ).grid(row=2, column=col_idx)
            col_idx += 1

        # Cost function selection
        tk.Label(frame, text="Cost Function:").grid(row=3, column=0, sticky="w")
        col_idx = 1
        for name, val in COSTS:
            tk.Radiobutton(
                frame, text=name, variable=self.cost_function, value=val
            ).grid(row=3, column=col_idx)
            col_idx += 1

        # Hidden layers
        tk.Label(frame, text="Hidden Layers (comma-separated):").grid(
            row=5, column=0, sticky="w"
        )
        self.hidden_layers_entry = tk.Entry(frame)
        self.hidden_layers_entry.insert(
            0, "10"
        )  # default one hidden layer with 10 neurons
        self.hidden_layers_entry.grid(row=5, column=1, pady=5)

        # Epochs
        tk.Label(frame, text="Epochs:").grid(row=6, column=0, sticky="w")
        self.epochs_entry = tk.Entry(frame)
        self.epochs_entry.insert(0, str(self.epochs))  # put current default epochs
        self.epochs_entry.grid(row=6, column=1, pady=5)

        # Buttons for actions
        self.load_button = tk.Button(frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=7, column=0, pady=5)

        self.init_net_button = tk.Button(
            frame, text="Init Network", command=self.init_network
        )
        self.init_net_button.grid(row=7, column=1, pady=5)

        self.train_button = tk.Button(
            frame, text="Train Network", command=self.train_network
        )
        self.train_button.grid(row=7, column=2, pady=5)

    def build_canvas(self):
        self.canvas_frame = ttk.Frame(self.master)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=400, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def build_log_area(self):
        self.log_frame = ttk.Frame(self.master)
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(self.log_frame, wrap="word", height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Welcome to the Neural Network GUI.\n")

    def load_data(self):
        if self.mode.get() == "real":
            self.log("Loading real dataset...")
            X_train, X_test, y_train, y_test, feature_names = (
                load_wisconsin_breast_cancer_data(
                    csv_path="data.csv", test_size=0.2, random_state=42
                )
            )
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.feature_names = feature_names
            self.log(
                f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}"
            )
        else:
            self.log(
                "Dummy mode selected, no data to load. Will use dummy data during training."
            )

    def init_network(self):
        if self.mode.get() == "real" and self.X_train is not None:
            input_dim = self.X_train.shape[1]
        else:
            input_dim = 5

        # Parse the hidden layers from the entry field
        # e.g., if user types "10,20" then hidden_layer_sizes = [10, 20]
        hidden_layers_str = self.hidden_layers_entry.get().strip()
        if hidden_layers_str == "":
            hidden_layer_sizes = []
        else:
            hidden_layer_sizes = [
                int(x.strip())
                for x in hidden_layers_str.split(",")
                if x.strip().isdigit()
            ]

        # Construct the full layer_sizes list: input layer, hidden layers, output layer
        layer_sizes = [input_dim] + hidden_layer_sizes + [1]

        self.net = Network(
            layer_sizes=layer_sizes,
            activation=self.hidden_activation.get(),
            output_activation=self.output_activation.get(),
            cost=self.cost_function.get(),
            learning_rate=self.learning_rate,
            seed=42,
        )

        self.log(f"Network initialized with layers: {layer_sizes}")
        self.draw_network()

    def train_network(self):
        if self.net is None:
            self.log("Please initialize the network first.")
            return

        # Get the number of epochs from the GUI
        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            self.log("Invalid epochs value, defaulting to 50.")
            epochs = 50

        if self.mode.get() == "dummy":
            np.random.seed(42)
            X_dummy = np.random.randn(100, 5)
            y_dummy = (np.sum(X_dummy, axis=1) > 0).astype(np.float32).reshape(-1, 1)
            X_train, y_train = X_dummy, y_dummy
            self.log("Training on dummy data...")
        else:
            if self.X_train is None:
                self.log("Please load data first.")
                return
            X_train, y_train = self.X_train, self.y_train.reshape(-1, 1)
            self.log("Training on real data...")

        # Use callback for real-time logging
        self.net.train(
            X_train,
            y_train,
            epochs=epochs,
            print_interval=self.print_interval,
            callback=self.train_callback,
        )

        if self.mode.get() == "dummy":
            acc = self.net.accuracy(X_train, y_train)
            self.log(f"Final Accuracy (dummy): {acc:.4f}")
            y_targets = y_train
            y_outputs = self.net.forward_prop(X_train)
        else:
            acc = self.net.accuracy(self.X_test, self.y_test)
            self.log(f"Final Accuracy (test set): {acc:.4f}")
            y_targets = self.y_test.reshape(-1, 1)
            y_outputs = self.net.forward_prop(self.X_test)

        self.log("Sample predictions:")
        for i in range(min(5, len(y_targets))):
            self.log(f"Target: {y_targets[i]}, Output: {y_outputs[i]}")

        self.draw_network()

    def train_callback(self, epoch, cost):
        self.log(f"Epoch {epoch}: Cost = {cost:.6f}")
        self.master.update_idletasks()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def draw_network(self):
        if self.net is None:
            return

        self.canvas.delete("all")
        width = 400
        height = 300
        num_layers = len(self.net.layers)
        layer_spacing = width / (num_layers + 1)
        max_neurons = max(len(layer) for layer in self.net.layers)
        neuron_radius = 10
        vertical_spacing = height / (max_neurons + 1)
        positions = []
        for l_idx, layer in enumerate(self.net.layers):
            x = (l_idx + 1) * layer_spacing
            layer_positions = []
            for n_idx, neuron in enumerate(layer):
                y = (n_idx + 1) * vertical_spacing
                layer_positions.append((x, y))
                self.canvas.create_oval(
                    x - neuron_radius,
                    y - neuron_radius,
                    x + neuron_radius,
                    y + neuron_radius,
                    fill="skyblue",
                )
            positions.append(layer_positions)

        # Draw axons
        for l_idx in range(len(self.net.layers) - 1):
            for n_out_idx, n_out in enumerate(self.net.layers[l_idx + 1]):
                for in_axon in n_out.inputs:
                    in_n = in_axon.input
                    n_in_idx = self.net.layers[l_idx].index(in_n)
                    x1, y1 = positions[l_idx][n_in_idx]
                    x2, y2 = positions[l_idx + 1][n_out_idx]
                    weight = in_axon.weight
                    width_line = max(1, min(5, abs(weight) * 50))
                    color = "red" if weight < 0 else "black"
                    self.canvas.create_line(
                        x1, y1, x2, y2, fill=color, width=width_line
                    )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
