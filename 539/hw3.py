import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pickle
from scipy.io import loadmat

RUN_TRIALS = False
DEBUG = False
# there are two ways to implement the classifier when the positive/negative label counts
# match in a given cell (or both equal 0):
#   DYNAMIC_COIN_FLIP = False: flip a coin once and use this label (pos or negative) against all test examples
#   DYNAMIC_COIN_FLIP = True: flip a coin for each training example
DYNAMIC_COIN_FLIP = True

mat_contents = loadmat('hw2_data.mat')
results_pkl = f'results_dynamic_{DYNAMIC_COIN_FLIP}.pickle'

X_train = mat_contents['X_train']
Y_train = mat_contents['y_train'][0]
X_test = mat_contents['X_test']
Y_test = mat_contents['y_test'][0]

Rmin = mat_contents['Rmin']

n_values = np.array([10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
m_values = np.array([2, 4, 8, 16])
num_trials = 100


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
	If the counts are equal (or zero), check the flip a coin to decide the class
		TODO: add ability to flip coin later
	:param points:
	:param labels:
	:param m:
	:return:
	"""
	# first count the majority class in each cell
	grid_counts = np.zeros((m, m), dtype=int)
	for point, label in zip(points.T, labels):
		x, y = point
		i, j = get_i_j(x, y, m)
		grid_counts[i, j] += label		# positive label increments, negative label decrements by same amount

	# use majority class to build the classifier
	classifier = np.zeros((m, m), dtype=int)
	for i in range(m):
		for j in range(m):
			count = grid_counts[i, j]
			# Assign the label based on counts
			if count > 0:
				classifier[i, j] = 1
			elif count < 0:
				classifier[i, j] = -1
			else:
				if DYNAMIC_COIN_FLIP:
					classifier[i, j] = 0        # test data will get a random class each time
				else:
					classifier[i, j] = np.random.choice([1, -1])    # classifier is solidified for all test data

	if DEBUG:
		print(f"{grid_counts=}")
		print(f"{classifier=}")

	return classifier


def compute_empirical_risk(classifier, test_points, test_labels, m):
	"""
	Calculate empirical risk of the classifier by running all test points through it
	The risk is calculated as the number of incorrectly classified test points
	divided by the total number of test points.
	:param classifier:
	:param test_points:
	:param test_labels:
	:param m:
	:return:
	"""
	counter = 0

	for point, label in zip(test_points.T, test_labels):
		x, y = point
		i, j = get_i_j(x, y, m)

		# Get the predicted label from the classifier
		predicted_label = classifier[i, j]
		if predicted_label == 0:                        # this will only be hit when DYNAMIC_COIN_FLIP is True
			predicted_label = np.random.choice([1, -1])

		# Check if the predicted label matches the true label
		if predicted_label != label:
			counter += 1

	# Compute the empirical risk
	empirical_risk = counter / len(test_points[0])
	return empirical_risk

def run_config(params):
	# individual run: run num_trials Monte Carlo runs of:
	#   build classifier using n training points
	#   calculate empirical risk using test data (of each trial and avg across all runs)
    m, n, X_train, Y_train, X_test, Y_test, num_trials = params
    all_risks = np.zeros(num_trials)
    for trial in range(num_trials):

        sample_indices = np.random.choice(len(X_train[0]), n, replace=False)
        sampled_indices = np.random.choice(X_train.shape[1], size=n, replace=False)
        sampled_points = X_train[:, sampled_indices]
        sampled_labels = Y_train[sample_indices]
        classifier = build_classifier(sampled_points, sampled_labels, m)
        emp_risk = compute_empirical_risk(classifier, X_test, Y_test, m)
        all_risks[trial] = emp_risk
        if DEBUG:
            print(f"{m=}, {n=}, Trial {trial+1}, Empirical Risk: {emp_risk}")

    average_risk = np.mean(all_risks)
    print(f"{m=}, {n=}, Average Risk: {average_risk}")
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
	plt.xlabel('n values')
	plt.ylabel('Average Empirical Risk')
	plt.title('Average Empirical Risk vs n Values')
	plt.legend()
	plt.grid(True)
	plt.savefig("plot1.png", bbox_inches='tight')

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

	plt.savefig("plot2.png", bbox_inches='tight')


def main():
	if RUN_TRIALS:
		params = [(m, n, X_train, Y_train, X_test, Y_test, num_trials) for m in m_values for n in n_values]

		# each config is independent, so can run in a separate thread
		# can plot once all are completed
		with Pool(processes=cpu_count() - 1) as pool:
			results = pool.map(run_config, params)

		# Save the results to a pickle file
		with open(results_pkl, 'wb') as file:
			pickle.dump(results, file)

	plot_results()


if __name__ == "__main__":
	main()
