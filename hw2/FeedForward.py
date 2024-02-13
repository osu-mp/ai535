"""
Matthew Pacey
AI 535 - HW2
"""

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################


class SigmoidCrossEntropy:

    # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
    #
    # TODO: Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
    def forward(self, logits, labels):
        # TODO store forward pass for use in backward
        raise Exception('Student error: You haven\'t implemented the forward pass for SigmoidCrossEntropy yet.')

    # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        # TODO : use forward pass stored results
        # TODO result = sigmoid(x) * (1 - sigmoid)
        raise Exception('Student error: You haven\'t implemented the backward pass for SigmoidCrossEntropy yet.')


class ReLU:

    # TODO: Compute ReLU(input) element-wise
    def forward(self, input):
        self.input = input
        raise Exception('Student error: You haven\'t implemented the forward pass for ReLU yet.')

    # TODO: Given dL/doutput, return dL/dinput
    def backward(self, grad):
        raise Exception('Student error: You haven\'t implemented the backward pass for ReLU yet.')

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size, momentum=0, weight_decay=0):
        # TODO : see slide 24 of L3.2
        return


class LinearLayer:

    # TODO: Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        # save dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        print(f"{self.input_dim=}, {self.output_dim=}")

        # init weights with random values (input_dim, output_dim)
        self.weights = np.random.randn(input_dim, output_dim)
        # init bias vector (1, output_dim)
        self.bias = np.zeros((1, output_dim))

        # TODO:
        self.grad_dl_dz = None
        self.grad_weights = None
        self.grad_bias = None

        # momentum related
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    # TODO: During the forward pass, we simply compute XW+b
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    # TODO: Backward pass inputs:
    #
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)

    def backward(self, labels):
        n = len(labels)
        # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
        #                       of the loss with respect to the weights of this layer.
        #                       This is an summation over the gradient of the loss of
        #                       each example with respect to the weights.
        #
        self.grad_weights = np.dot(self.input.T, labels)
        assert self.grad_weights.shape == (self.input_dim, self.output_dim)

        # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
        #                       of the loss with respect to the bias of this layer.
        #                       This is an summation over the gradient of the loss of
        #                       each example with respect to the bias.
        self.grad_bias = np.sum(labels, axis=0, keepdims=True)
        assert self.grad_bias.shape == (1, self.output_dim)

        # Return Value:
        #
        # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
        #               the i'th row is the gradient of the loss of example i with respect
        #               to x_i (the input of this layer for example i)
        grad_input = np.dot(labels, self.weights.T)
        assert (grad_input.shape == (n, self.input_dim))
        # assert grad_input.shape == (self.input_dim)
        return grad_input

    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        # update the weights
        self.weights -= step_size * (self.grad_weights + weight_decay * self.weights)
        self.bias -= step_size * self.grad_bias

        # implement momentum
        if momentum > 0.0:
            self.velocity_weights = momentum * self.velocity_weights - step_size * (self.grad_weights + weight_decay * self.weights)
            self.velocity_bias = momentum * self.velocity_bias - step_size * self.grad_bias

            self.weights += self.velocity_weights
            self.bias += self.velocity_bias

        # Clear gradients for next iteration
        self.grad_weights = None
        self.grad_bias = None


######################################################
# Q4 Implement Evaluation for Monitoring Training
###################################################### 

# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    raise Exception('Student error: You haven\'t implemented the step for evaluate function.')


def create_mini_batches(inputs, labels, batch_size):
    sample_count = len(inputs)
    batches = []

    indices = np.arange(sample_count)
    np.random.shuffle(indices)
    shuffled_inputs = inputs[indices]
    shuffled_labels = labels[indices]

    batch_count = sample_count // batch_size
    for i in range(batch_count):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size

        batch_inputs = shuffled_inputs[batch_start:batch_end]
        batch_labels = shuffled_labels[batch_start:batch_end]
        batches.append((batch_inputs, batch_labels))

    if sample_count % batch_size != 0:
        batch_start = batch_count * batch_size
        batch_inputs = shuffled_inputs[batch_start:]
        batch_labels = shuffled_labels[batch_start:]
        batches.append((batch_inputs, batch_labels))

    return batches

def normalize_data(train_data, test_data):
    # convert to float first
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # compute mean and stdev on training data
    mean = np.mean(train_data, axis=0)
    stdev = np.std(train_data, axis=0)

    # normalize the data around 0 with stdev of 1 (use train mean/std for both)
    norm_train_data = (train_data - mean) / stdev
    norm_test_data = (test_data - mean) / stdev

    return norm_train_data, norm_test_data


def binary_cross_entropy_loss(predictions, labels):
    # Ensure numerical stability by clipping predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

    # Compute binary cross-entropy loss
    loss = - (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    # Take the mean across the batch
    mean_loss = np.mean(loss)

    return mean_loss


def compute_loss_gradients(predictions, labels):
    # Compute gradients of binary cross-entropy loss with respect to predictions

    # Number of examples in the batch
    num_examples = predictions.shape[0]

    # Compute gradients for each example
    grad_loss = predictions - labels

    # Normalize gradients by the number of examples in the batch
    grad_loss /= num_examples

    return grad_loss


def main():
    # TODO: Set optimization parameters (NEED TO SUPPLY THESE)
    batch_size = 256  # TODO: lower? 32, 64
    max_epochs = 5 # 300  # TODO: lower? 50, 100
    step_size = 0.1

    number_of_layers = 1  # TODO : update
    width_of_layers = 64
    weight_decay = 0.001
    momentum = 0.8


    # Load data
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']


    X_train, X_test = normalize_data(X_train, X_test)

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 2  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # training epoch loop
    for i in range(max_epochs):
        print(f"Training loop #{i + 1}")
        batches = create_mini_batches(X_train, Y_train, batch_size)
        for inputs, labels in batches:
            # Compute forward pass
            predictions = net.forward(inputs)
            # print(f"{predictions=}")
            # print(f"     {labels=}")

            # Compute loss
            loss = binary_cross_entropy_loss(predictions, labels)

            # Compute gradients of loss with respect to predictions
            grad_loss = compute_loss_gradients(predictions, labels)

            # Backward loss and networks
            gradients = net.backward(grad_loss)

            # Take optimizer step
            net.step(step_size, momentum, weight_decay)

            # Book-keeping for loss / accuracy

            # Evaluate performance on test.
        # _, tacc = evaluate(net, X_test, Y_test, batch_size)
        # print(tacc)

        ###############################################################
        # Print some stats about the optimization process after each epoch
        ###############################################################
        # epoch_avg_loss -- average training loss across batches this epoch
        # epoch_avg_acc -- average accuracy across batches this epoch
        # vacc -- testing accuracy this epoch
        ###############################################################

        # logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,epoch_avg_loss, epoch_avg_acc, vacc*100))

    ###############################################################
    # Code for producing output plot requires
    ###############################################################
    # losses -- a list of average loss per batch in training
    # accs -- a list of accuracies per batch in training
    # val_losses -- a list of average testing loss at each epoch
    # val_acc -- a list of testing accuracy at each epoch
    # batch_size -- the batch size
    ################################################################

    # Plot training and testing curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            raise Exception("TODO")
        # TODO: Please create a network with hidden layers based on the parameters

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size, momentum, weight_decay):
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)


def displayExample(x):
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)

    plt.imshow(np.stack([r, g, b], axis=2))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
