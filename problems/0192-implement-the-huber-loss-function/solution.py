import numpy

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""

	y1 = numpy.asarray(y_true)
	y2 = numpy.asarray(y_pred)

	losses = numpy.where(
		numpy.abs(y1 - y2) < delta,
		0.5 * (y1 - y2) ** 2,
		delta * (numpy.abs(y1 - y2) - 0.5 * delta)
	)

	return numpy.mean(losses)