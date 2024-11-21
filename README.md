# UTPB-COSC-6389-Project2

This repo contains the assignment and provided code base for Project 2 of the graduate Computational Biomimicry class.

The goals of this project are:

1. Learn the basics of how neural networks work, and how to build and use them.
2. Understand the kinds of problems that neural networks are best at solving, and how to apply them to find solutions.

Description:
The provided Python code base includes an implementation of a basic neural network. Right now, it uses completely random input values and attempts to optimize or "learn" to produce a given output from those inputs.

Your task is twofold: first, you're going to find a problem that you can solve using a neural network that only handles binary or real-valued inputs and makes some sort of choice based on those inputs; second, you're going to implement additional activation/cost functions and provide means within the UI of configuring the network to use the function desired by the user during runtime.

This site (https://www.kaggle.com/datasets) seems to be a database of datasets that you will likely find useful for both projects 2 and 3.

As with Project 1, your application must generate the neural networks, display them on a canvas, and update them in real time as the weight values change.

You are not allowed to make use of any libraries related to neural networks in the code for this project. The implementation of the network construction, operation, forward and backward propagation, training, and testing must all be your own.

Grading criteria:

1. If the code submitted via your pull request does not compile, the grade is zero.
2. If the code crashes due to forseeable unhandled exceptions, the grade is zero.
3. For full points, the code must correctly perform the relevant algorithms and display the network in real time, via the UI.

Deliverables:
A Python application which provides a correct implementation of a neural network generation and training system, and allows the user to define whether the hidden neurons use the sigmoid, tanh, or RelU function for their activation. The selection of activation function for the output neurons will depend upon the choice of problem, but should be matched to the problem correctly.
