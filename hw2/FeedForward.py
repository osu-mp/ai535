"""
Matthew Pacey
AI 535 - HW2
Neural Nets and Backpropagation
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


batch_plot = "batches.png"
learn_plot = "learning_rate.png"
hidden_plot = "hidden_plot.png"
NUM_TRIALS = 10                     # run this many trials per config and take average test accuracy
OMIT_SINGLE_PLOT = True             # for each config, just return highest test acc, do not plot individual run
RUN_PARALLEL = True                 # run trials of batch size, learning rates, and hidden unit configs in parallel
DEBUG = False                       # turn on to display debug prints

######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################


class SigmoidCrossEntropy:

    # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
    # Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
    def forward(self, logits, labels):
        self.labels = labels
        self.y_pred = 1 / (1 + np.exp(-logits))

        # Compute cross-entropy loss
        loss = - (labels * np.log(self.y_pred) + (1 - labels) * np.log(1 - self.y_pred))

        # Take the mean across the batch
        self.output = np.mean(loss)

        # error check
        assert self.output >= 0, "output should be a positive scalar"
        return self.output


    # Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        # use forward pass stored results
        # Compute gradient of loss with respect to the sigmoid logits
        dx = self.y_pred - self.labels
        return np.mean(dx)


class ReLU():

    # Compute ReLU(input) element-wise
    def forward(self, input):
        self.input = input
        self.deriv = np.where(input > 0, 1, 0)      # precompute deriv of i
        return np.maximum(0, input)

    # Given dL/doutput, return dL/dinput
    def backward(self, grad):
        grad_input = grad * self.deriv
        return grad_input

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size, momentum=0, weight_decay=0):
        return


class LinearLayer:

    # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        # save dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # randomize weights between +/- sqrt k
        k = 1 / input_dim
        self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), (input_dim, output_dim))
        assert self.weights.shape == (input_dim, output_dim)

        # init bias vector (1, output_dim)
        self.bias = np.zeros((1, output_dim))

        # momentum related
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)

    # During the forward pass, we simply compute XW+b
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    # Backward pass inputs:
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)
    def backward(self, labels):
        n = len(labels)
        # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
        #                       of the loss with respect to the weights of this layer.
        #                       This is an summation over the gradient of the loss of
        #                       each example with respect to the weights.
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
        return grad_input

    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        assert self.grad_weights.shape == self.weights.shape
        assert self.velocity_weights.shape == self.grad_weights.shape

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

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    num_examples = X_val.shape[0]
    total_loss = 0
    correct_predictions = 0

    # Create mini-batches
    batches = create_mini_batches(X_val, Y_val, batch_size)

    for inputs, labels in batches:
        # Forward pass
        predictions = model.forward(inputs)

        # Compute loss
        loss = binary_cross_entropy_loss(predictions, labels)
        total_loss += loss * inputs.shape[0]

        # Compute accuracy
        predicted_labels = (predictions > 0.5).astype(np.int8)
        correct_predictions += np.sum(predicted_labels == labels)

    # Compute average loss and accuracy
    average_loss = total_loss / num_examples
    accuracy = correct_predictions / num_examples

    return average_loss, accuracy


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


def main(batch_size, step_size, hidden_units):
    # batch_size = 64
    max_epochs = 10
    # step_size = 0.01

    width_of_layers = hidden_units
    number_of_layers = len(hidden_units)

    weight_decay = 0.01
    momentum = 0.8

    if DEBUG:
        print(f"RUN TEST {batch_size=}, {step_size=}, {hidden_units=}")

    # Load data
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']


    X_train, X_test = normalize_data(X_train, X_test)

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 1  # number of class labels

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
        if DEBUG:
            print(f"Training loop #{i + 1}")
        batches = create_mini_batches(X_train, Y_train, batch_size)
        train_losses = []
        for inputs, labels in batches:
            # Compute forward pass
            predictions = net.forward(inputs)

            # Compute loss
            loss = binary_cross_entropy_loss(predictions, labels)

            # Compute gradients of loss with respect to predictions
            grad_loss = compute_loss_gradients(predictions, labels)

            # Backward loss and networks
            gradients = net.backward(grad_loss)

            # Take optimizer step
            net.step(step_size, momentum, weight_decay)

            # Book-keeping for loss / accuracy
            train_losses.append(loss)

        average_loss, accuracy = evaluate(net, inputs, labels, batch_size)
        losses.append(average_loss)
        accs.append(accuracy)


        # Evaluate performance on test.
        val_loss, vacc = evaluate(net, X_test, Y_test, batch_size)
        val_accs.append(vacc)
        val_losses.append(val_loss)


        ###############################################################
        # Print some stats about the optimization process after each epoch
        ###############################################################
        # epoch_avg_loss -- average training loss across batches this epoch
        # epoch_avg_acc -- average accuracy across batches this epoch
        # vacc -- testing accuracy this epoch
        ###############################################################
        epoch_avg_loss = average_loss # np.mean(losses)
        epoch_avg_acc = np.mean(accs)
        if DEBUG:
            logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,epoch_avg_loss, epoch_avg_acc*100, vacc*100))

    ###############################################################
    # Code for producing output plot requires
    ###############################################################
    # losses -- a list of average loss per batch in training
    # accs -- a list of accuracies per batch in training
    # val_losses -- a list of average testing loss at each epoch
    # val_acc -- a list of testing accuracy at each epoch
    # batch_size -- the batch size
    ################################################################

    if OMIT_SINGLE_PLOT:
        best_acc = np.max(val_accs)                      # report the highest test accuracy
        print(f"Trial: {batch_size=}, {step_size=}, {hidden_units=}, {best_acc=}")
        return best_acc
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


# Define a function to run a single test
def run_test(batch_size, learning_rate, hidden_units):
    # run the given config
    accs = []
    for i in range(NUM_TRIALS):
        accs.append(main(batch_size, learning_rate, hidden_units))
    avg_acc = np.average(accs) * 100
    print(f"AVG RESULT: {batch_size=}, {learning_rate=}, {hidden_units=}, {avg_acc}")
    return avg_acc

# Define a function to run tests with different batch sizes and generate plot
def plot_batch_sizes(batch_sizes, learning_rate, hidden_units):
    results = []
    for batch_size in batch_sizes:
        test_accuracy = run_test(batch_size, learning_rate, hidden_units)
        results.append((batch_size, test_accuracy))

    title = f'Test Accuracy vs. Batch Size (LR: {learning_rate}, Hidden Units: {hidden_units})'
    xlabel = 'Batch Size'
    plot_results(results, title, xlabel, fname=batch_plot)


# Define a function to run tests with different learning rates and generate plot
def plot_learning_rates(batch_size, learning_rates, hidden_units):
    results = []
    for lr in learning_rates:
        test_accuracy = run_test(batch_size, lr, hidden_units)
        results.append((lr, test_accuracy))

    title = f'Test Accuracy vs. Learning Rate (Batch Size: {batch_size}, Hidden Units: {hidden_units})'
    xlabel = 'Learning Rate'
    plot_results(results, title, xlabel, fname=learn_plot)


def plot_hidden_units(batch_size, learning_rate, unit_cfgs):
    results = []
    for units in unit_cfgs[0]:
        test_accuracy = run_test(batch_size, learning_rate, units)
        unit_str = ", ".join(str(i) for i in units)
        results.append((unit_str, test_accuracy))

    title = f'Test Accuracy vs. Hidden Units (Batch Size: {batch_size}, Learning Rate: {learning_rate})'
    xlabel = 'Number of Hidden Units'
    plot_results(results, title, xlabel, fname=hidden_plot, font_size=10)

def plot_results(results, title, xlabel, fname, font_size=None):
    # Plot the results
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    # Extract x and y values from results
    x_values, y_values = zip(*results)

    # Plot data points with specified x values
    plt.plot(x_values, y_values, marker='o')

    plt.ylabel('Test Accuracy (%)')
    plt.xlabel(xlabel)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    # adjust xlabel for hidden units plots
    if font_size:
        num_ticks = len(x_values)
        plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to 20%
        plt.xticks(rotation=-90)
        plt.xticks(range(num_ticks), x_values, fontsize=font_size)

    plt.savefig(fname)
    print(f"Generated {fname}")


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dims, num_layers):
        self.layers = [
            LinearLayer(input_dim, hidden_dims[0]),
            ReLU(),
        ]

        for i in range(1, len(hidden_dims)):
            self.layers.extend([
                LinearLayer(hidden_dims[i - 1], hidden_dims[i]),
                ReLU(),
            ])

        self.layers.append(LinearLayer(hidden_dims[-1], output_dim))
        # layers produce logits (any number), sigmoid reduces those to probs [0,1)

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

# Define a function to execute each plot function
def execute_plot_function(plot_function, *args):
    plot_function(*args)


if __name__ == "__main__":
    # Define batch sizes, learning rates, and number of hidden units
    best_batch = 512
    best_lr = 0.01
    best_hidden_units = [128, 64]


    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
    hidden_units_cfgs = [
        [64],
        [128],
        [128, 64],
        [512, 64],
        [1024, 256],
        [256, 128, 64],
        [1024, 128, 64],
        [512, 256, 128, 64],
        [1024, 512, 256, 128, 64],
    ],

    if RUN_PARALLEL:
        # Create a pool of processes
        pool = multiprocessing.Pool(processes=3)

        # Submit each plot function to the pool for parallel execution
        pool.apply_async(execute_plot_function, (plot_batch_sizes, batch_sizes, best_lr, best_hidden_units))
        pool.apply_async(execute_plot_function, (plot_learning_rates, best_batch, learning_rates, best_hidden_units))
        pool.apply_async(execute_plot_function, (plot_hidden_units, best_batch, best_lr, hidden_units_cfgs))

        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()
    else:
        # Plot test accuracy vs. batch sizes
        plot_batch_sizes(batch_sizes, best_lr, best_hidden_units)

        # Plot test accuracy vs. learning rates
        plot_learning_rates(best_batch, learning_rates, best_hidden_units)

        # Plot test accuracy vs. learning rates
        plot_hidden_units(best_batch, best_lr, hidden_units_cfgs)

