
import math

def single_neuron_model(features: list[list[float]], labels: list[int],
                        weights: list[float], bias: float) -> tuple[list[float], float]:

    prediction = []

    for samples in features:
        z = bias

        for i in range(len(samples)):
            z += samples[i] * weights[i]

        probability = 1 / (1 + math.exp(-z))
        prediction.append(probability)

    # Calculate MSE
    total_error = 0

    for i in range(len(labels)):
        error = prediction[i] - labels[i]
        total_error += error ** 2

    mse = total_error / len(labels)

    predictions = [round(p, 4) for p in prediction]

    return predictions, round(mse, 4)
