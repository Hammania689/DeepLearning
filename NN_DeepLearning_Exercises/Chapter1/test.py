# ----------------------
# - read the input data:
'''
'''
import Chapter1.mnist_loader
training_data, validation_data, test_data = Chapter1.mnist_loader.load_data_wrapper()
training_data = list(training_data)
# ---------------------
# - network.py example:
import Chapter1.NeuralNet

net = Chapter1.NeuralNet.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
