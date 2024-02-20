"""
Matthew Pacey
AI 539 - HW3
Empirical Risk Analysis of Plug-In Classifier
"""

import os.path
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
from scipy.io import loadmat

RUN_TRIALS = True
DEBUG = True
# there are two ways to implement the classifier when the positive/negative label counts
# match in a given cell (or both equal 0):
#   DYNAMIC_COIN_FLIP = False: flip a coin once and use this label (pos or negative) against all test examples
#   DYNAMIC_COIN_FLIP = True: flip a coin for each training example
DYNAMIC_COIN_FLIP = True

# load data from matlab
mat_contents = loadmat('hw2_data.mat')
results_pkl = f'results_dynamic_{DYNAMIC_COIN_FLIP}.pkl'
buckets_pkl = 'buckets.pkl'

X_train = mat_contents['X_train']
Y_train = mat_contents['y_train'][0]
X_test = mat_contents['X_test']
Y_test = mat_contents['y_test'][0]

Rmin = mat_contents['Rmin'][0][0]

n_values = np.array([10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
m_values = np.array([2, 4, 8, 16])
num_trials = 100

plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)


def get_i_j(x, y, m):
    """
    Return the cell index for a given point/grid configuration

    :param x: x-coordinate of the point (between 0 and 1)
    :param y: y-coordinate of the point (between 0 and 1)
    :param m: number of cells in each dimension
    :return: tuple (i, j) representing the cell indices
    """
    i = min(int(x * m), m - 1)
    j = min(int(y * m), m - 1)
    return i, j


def build_classifier(points, labels, m):
    """
    Using the test points/labels, build a plug-in classifier with a grid of m by m cells
    Take the majority class in each cell and set that as the predicted value.
    If the counts are equal (or zero), flip a coin to decide the class
    :param points:
    :param labels:
    :param m:
    :return:
    """
    # cumulative sum of label values in each cell (pos and neg will offset each other)
    grid_label_sums = np.zeros((m, m), dtype=float)
    # total number of labels in each cell (can be used to create probability heatmap)
    grid_label_counts = np.zeros((m, m), dtype=float)

    # put the training points in their respective cells
    for point, label in zip(points.T, labels):
        x, y = point
        i, j = get_i_j(x, y, m)
        grid_label_sums[i, j] += label
        grid_label_counts[i, j] += 1

    # use majority class in each cell to build a deterministic classifier
    classifier = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            count = grid_label_sums[i, j]
            if count > 0:                   # positive labels are majority
                classifier[i, j] = 1
            elif count < 0:                 # negative lavels are majority
                classifier[i, j] = -1
            else:                           # equal numbers/both are zero
                if DYNAMIC_COIN_FLIP:
                    classifier[i, j] = 0
                else:
                    classifier[i, j] = np.random.choice([-1, 1])

    # not strictly necessary, but useful for debug
    # create a heatmap of probabilities for each cell
    mask = grid_label_counts == 0       # use this mask to avoid divide by zero if there are no points in the given cell
    masked_grid_label_counts = np.ma.array(grid_label_counts, mask=mask)
    grid_probs = np.where(mask, 0, grid_label_sums / masked_grid_label_counts)

    return classifier, grid_probs


def compute_empirical_risk(classifier, test_bucket, m):
    """
    Calculate empirical risk of the classifier by running all test points through it
    The risk is calculated as the number of incorrectly classified test points
    divided by the total number of test points.
    :param classifier:
    :param test_bucket:
    :param m:
    :return:
    """
    miss_count = 0
    total_points = 0

    for i in range(m):
        for j in range(m):
            neg_count = test_bucket[i, j, 0]
            pos_count = test_bucket[i, j, 1]
            total_points += neg_count + pos_count

            predicted_label = classifier[i, j]
            if predicted_label == 0:  # this will only be hit when DYNAMIC_COIN_FLIP is True
                predicted_label = np.random.choice([1, -1])

            # take the points that do not match the given cell's label and increment the miss coun
            if predicted_label == -1:
                miss_count += pos_count
            else:
                miss_count += neg_count

    # empirical risk is total mis-classifications divided by total number of points
    # subtract the given minimum risk
    empirical_risk = miss_count / total_points
    return empirical_risk - Rmin


def build_test_buckets(points, labels):
    """
    Do a one time build of the test points into the m x m cells
    Count positive and negative labels in each m value
    This will eliminate redundant calculations for each classifier
    Can be done since we will evaluate with all test points and those points are unchanged
    :param points:
    :param labels:
    :return:
    """
    if os.path.exists(buckets_pkl):
        with open(buckets_pkl, 'rb') as file:
            test_buckets = pickle.load(file)
            print(f"{len(test_buckets)} Test Buckets Loaded From Pickle")
            return test_buckets

    test_buckets = {}
    for m in m_values:
        m_bucket = np.zeros((m, m, 2), dtype=int)  # i x j cells with a negative/positive count for each
        for point, label in zip(points.T, labels):
            x, y = point
            i, j = get_i_j(x, y, m)
            if label == -1:
                m_bucket[i, j, 0] += 1          # increment negative count
            else:
                m_bucket[i, j, 1] += 1          # increment positive count
        test_buckets[m] = m_bucket

    print(f"{len(test_buckets)} Test Buckets Built")
    # Save the results to a pickle file
    with open(buckets_pkl, 'wb') as file:
        pickle.dump(test_buckets, file)
    return test_buckets


def run_config(params):
    # individual run: run num_trials Monte Carlo runs of:
    #   build classifier using n training points
    #   calculate empirical risk using test data (of each trial and avg across all runs)
    m, n, test_buckets = params
    all_risks = np.zeros(num_trials)
    average_classifier = np.zeros((m, m), dtype=int)
    average_probs = np.zeros((m, m), dtype=float)
    for trial in range(num_trials):
        sampled_indices = np.random.choice(X_train.shape[1], size=n, replace=False)
        sampled_points = X_train[:, sampled_indices]
        sampled_labels = Y_train[sampled_indices]
        classifier, grid_probs = build_classifier(sampled_points, sampled_labels, m)
        emp_risk = compute_empirical_risk(classifier, test_buckets[m], m)
        all_risks[trial] = emp_risk
        # if DEBUG:
        #     print(f"{m=}, {n=}, Trial {trial+1}, Empirical Risk: {emp_risk}")

        average_classifier += classifier
        average_probs += grid_probs / num_trials

    average_risk = np.mean(all_risks)
    print(f"{m=}, {n=}, Average Risk: {average_risk}")

    # Create heatmap of averages
    plt.figure()
    plt.imshow(average_probs, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    formatted_n = f"{n:.0e}".split('e')[1]
    plt.title(f'Average Probability Heatmap with m={m}, n=$10^{formatted_n}$')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.xticks([])
    plt.yticks([])
    fname = os.path.join(plot_dir, f'average_m_{m}_n_{n}.png')
    plt.savefig(fname)

    # Create heatmap of cumulative classifier
    plt.figure()
    average_classifier[average_classifier > 0] = 1
    average_classifier[average_classifier < 0] = -1
    plt.imshow(average_classifier, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    formatted_n = f"{n:.0e}".split('e')[1]
    plt.title(f'Average Classifier with m={m}, n=$10^{formatted_n}$')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.xticks([])
    plt.yticks([])
    fname = os.path.join(plot_dir, f'classifier_m_{m}_n_{n}.png')
    plt.savefig(fname)

    return average_risk, all_risks

def plot_results():
    """
    Create 2 plots from data:
        line plot of average risks across all m and n combinations
        scatter plot of all risks across all m and n combinations
            draw line connecting fifth highest and fifth lowest of each m value
    :return:
    """
    with open(results_pkl, 'rb') as file:
        results = pickle.load(file)

    # extract data from saved pickle
    average_risks = np.array([result[0] for result in results]).reshape(len(m_values), len(n_values))
    all_risks = np.array([result[1] for result in results]).reshape(len(m_values), len(n_values), num_trials)

    # Plot 1: Line plot of average risk
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(m_values):
        plt.plot(n_values, average_risks[i], label=f'm={m}')

    # Set the y-axis tick formatter to not use scientific notation and set the format string to plain numbers
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    plt.gca().ticklabel_format(style='plain', axis='y')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n values (log scale)')
    plt.ylabel('Average Empirical Risk (log scale)')
    plt.title('Average Empirical Risk vs n Values')
    plt.legend()
    plt.grid(True)
    fname = os.path.join(plot_dir, f"plot1_dynamic_{DYNAMIC_COIN_FLIP}.png")
    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")

    # Plot 2: Scatter plot of empirical risk with trendlines
    plt.figure()
    for i, m in enumerate(m_values):
        first = True
        for j, n in enumerate(n_values):
            if first:
                plt.scatter(n * np.ones(num_trials), all_risks[i, j], label=f"m={m}", alpha=0.2, color=f'C{i}')
                first = False
            else:
                plt.scatter(n * np.ones(num_trials), all_risks[i, j], alpha=0.2, color=f'C{i}')


    # Trendlines for the 5th best/worst empirical risk across different n values for each m value
    for i, m in enumerate(m_values):
        fifth_highest_risks = []
        fifth_lowest_risks = []
        for j, n in enumerate(n_values):
            sorted_risks = np.sort(all_risks[i, j])
            fifth_highest_risks.append(sorted_risks[-5])
            fifth_lowest_risks.append(sorted_risks[5])

        # Connect the 5th best/worst empirical risks across different n values
        plt.plot(n_values, fifth_highest_risks, label=f'm={m} 90%', linestyle='-', color=f'C{i}')
        plt.plot(n_values, fifth_lowest_risks, linestyle='-', color=f'C{i}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n values (log scale)')
    plt.ylabel('Empirical Risk (log scale)')
    plt.title('Empirical Risk vs n Values With Trendlines')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=1.5)  # Place legend centered on the right side
    plt.grid(True)
    # Set the y-axis tick formatter to display numbers without scientific notation (does not work)
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))

    fname = os.path.join(plot_dir, f"plot2_dynamic_{DYNAMIC_COIN_FLIP}.png")
    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")


def main():
    if RUN_TRIALS:
        test_buckets = build_test_buckets(X_test, Y_test)
        params = [(m, n, test_buckets) for m in m_values for n in n_values]

        # each config is independent, so can run in a separate thread
        # can plot once all are completed
        with Pool(processes=cpu_count() - 2) as pool:
            results = pool.map(run_config, params)

        # Save the results to a pickle file
        with open(results_pkl, 'wb') as file:
            pickle.dump(results, file)

    plot_results()


if __name__ == "__main__":
    main()
