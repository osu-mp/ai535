from FeedForward import ReLU, LinearLayer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class TestFeedForward(unittest.TestCase):
	def setUp(self):
		pass

	def skip_test_ReLU(self):
		# Define a function to perform a numerical gradient check
		# Define a function to perform a numerical gradient check
		def numerical_gradient_check(input_data, custom_layer):
			# Forward pass
			output_custom = custom_layer(input_data)

			# Compute gradients numerically using finite differences
			epsilon = 1e-4
			numerical_gradients = torch.zeros_like(input_data)
			for i in range(input_data.shape[0]):
				for j in range(input_data.shape[1]):
					for k in range(input_data.shape[2]):
						for l in range(input_data.shape[3]):
							input_data_plus = input_data.clone().detach()
							input_data_plus[i, j, k, l] += epsilon
							output_plus = custom_layer(input_data_plus)

							input_data_minus = input_data.clone().detach()
							input_data_minus[i, j, k, l] -= epsilon
							output_minus = custom_layer(input_data_minus)

							numerical_gradients[i, j, k, l] = (output_plus - output_minus) / (2 * epsilon)

			# Backward pass
			custom_layer.zero_grad()
			output_custom.sum().backward()
			analytical_gradients = input_data.grad

			# Check the difference between numerical and analytical gradients
			diff = torch.abs(numerical_gradients - analytical_gradients)
			max_diff = torch.max(diff)
			avg_diff = torch.mean(diff)

			return max_diff, avg_diff

		# Generate dummy data
		input_data = torch.randn(5, 3, 10, 10)  # Batch size 5, 3 channels, 10x10 images

		# Create instances of ReLU layers
		torch_relu = torch.nn.ReLU()
		custom_relu = ReLU()

		# Forward pass through PyTorch ReLU layer
		output_torch = torch_relu(input_data)

		# Forward pass through custom ReLU layer
		output_custom = custom_relu(input_data)

		# Compare outputs
		print("Maximum absolute difference:", torch.max(torch.abs(output_torch - output_custom)))
		print("Average absolute difference:", torch.mean(torch.abs(output_torch - output_custom)))

		# Perform gradient check
		max_diff, avg_diff = numerical_gradient_check(input_data, custom_relu)
		print("Maximum absolute difference in gradients:", max_diff)
		print("Average absolute difference in gradients:", avg_diff)

	def test_linear(self):
		np.random.seed(0)
		torch.manual_seed(0)

		input_size = 5
		output_size = 3
		batch_size = 10
		input_data = np.random.randn(batch_size, input_size).astype(np.float32)

		# Instantiate custom LinearLayer
		custom_linear = LinearLayer(input_size, output_size)

		# Forward pass
		output_custom = custom_linear.forward(input_data)

		# Backward pass
		labels = np.random.randn(batch_size, output_size).astype(np.float32)
		grad_input_custom = custom_linear.backward(labels)

		# Step function
		step_size = 0.1
		custom_linear.step(step_size)

		# Convert numpy arrays to torch tensors
		input_torch = torch.tensor(input_data, requires_grad=True)
		torch_linear = nn.Linear(input_size, output_size)
		torch_loss = torch.tensor(labels)

		# Forward pass with PyTorch Linear layer
		output_torch = torch_linear(input_torch)

		# Backward pass in PyTorch
		loss = torch.nn.MSELoss()(output_torch, torch_loss)
		loss.backward()

		# Get gradients from PyTorch tensor
		grad_input_torch = input_torch.grad.detach().numpy()

		print("Backward pass comparison:")
		print("Maximum absolute difference in gradients:", np.max(np.abs(grad_input_custom - grad_input_torch)))
		print("Average absolute difference in gradients:", np.mean(np.abs(grad_input_custom - grad_input_torch)))


if __name__ == '__main__':
	unittest.main()
