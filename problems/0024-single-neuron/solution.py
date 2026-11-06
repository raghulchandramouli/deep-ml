import math
import numpy

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	probabilities = []
	for row in features:
		total = 0
		for f, w in zip(row, weights):
			total += f * w
		total += bias

		sigmoid = 1/(1+math.exp(-total))
		probabilities.append(sigmoid)
	
	mse = 0
	for p, l in zip(probabilities, labels):
		mse += (p - l)**2

	return probabilities, mse/len(labels)