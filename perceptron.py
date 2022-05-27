
x_input = [0.1, 0.5, 0.2]
w_weights = [0.6, 0.7, 0.7]
threshold =0.5


def perceptron():
	weighted_sum =0 
	for x,w in zip(x_input, w_weights):
		weighted_sum += x*w
		print(weighted_sum)
	if weighted_sum > threshold: 
		return 1
	else:
		return 0

output = perceptron()

print("Output:" + str(output))

