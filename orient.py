import sys
import random
import math
import json

from heapq import nlargest
from itertools import izip, imap

def sigmoid(u):
    if u >= 0:
        return 1.0/(1.0+math.exp(-u))
    temp = math.exp(u)
    return temp / (1.0 + temp)

sigmoid_ = lambda u: sigmoid(u) * (1 - sigmoid(u))
tanh = lambda u: math.tanh(u)
tanh_ = lambda u: 1 - math.tanh(u)**2
new = lambda u: 1.7159*math.tanh(2.0 * u / 3.0) 
new_ = lambda u: 1.14393 * tanh_(2.0 * u / 3.0)

try:
    from numpy import dot
except:
    def dot(x,y):
        if len(x) != len(y):
            raise "Unmatched dimensions."
        return sum(map(lambda (a,b): a*b, zip(x,y)))


class TestData(object):
    def __init__(self, id_str, orientation, data):
        self.id_str = id_str
        self.orientation = orientation
        self.data = data

    def __repr__(self):
        return "<%s> => %d\n%s"%(self.id_str, self.orientation, " ".join(map(str, self.data)))

def knn(train_data, test_data, k):
    k = int(k)
    euclidean_dist = lambda t1, t2: sum(map(lambda (x,y): (x-y)**2, zip(t1.data, t2.data)))
    for data in test_data:
        largest = nlargest(k, map(lambda x: (-euclidean_dist(x, data),x), train_data))
        orientations = map(lambda x: x[1].orientation, largest)
        yield data, max(set(orientations), key=orientations.count)

def load_data_file(fileName):
    return [TestData(line[0], int(line[1]), map(int,line[2:])) for line in map(lambda x: x.strip().split(), open(fileName))]


class NeuralNetwork(object):
    def __init__(self, train_data, hiddenCount, alpha, fn, fn_):
        rand_weight = lambda : random.random()*2 - 1
        self.num_nodes = [len(train_data[0].data)+1, hiddenCount, 4]
        self.activations = [[0] * x for x in self.num_nodes]
        self.weights = [
            [[rand_weight() for __ in range(self.num_nodes[1])] for _ in range(self.num_nodes[0])], 
            [[rand_weight() for __ in range(self.num_nodes[2])] for _ in range(self.num_nodes[1])]
        ]
        self.fn = fn
        self.fn_ = fn_
        self.alpha = alpha

    def train(self, train_data):
        o = lambda x: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]][x/90]
        alpha = self.alpha

        for iteration in range(50):
            print "Iteration:",iteration
            sum_errors = []
            for input_set, output_set in imap(lambda x: (x.data, o(x.orientation)), train_data):
                result = self.solve(input_set)
                
                errors = [[0] * self.num_nodes[1], [0] * self.num_nodes[2]]
                #Layer 2-3 errors
                errors[1] = [self.fn_(self.activations[2][x]) * (output_set[x] - self.activations[2][x]) \
                                for x in range(self.num_nodes[2])]
                
                #Layer 1-2 errors
                for neuron_index in range(self.num_nodes[1]):
                    error = 0.0
                    for x in range(self.num_nodes[2]):
                        error += errors[1][x] * self.weights[1][neuron_index][x]
                    #sum(errors[2][x]*self.weights[2][neuron_index][x] for x in range(self.num_nodes[2]))
                    errors[0][neuron_index] = self.fn_(self.activations[1][neuron_index]) * error

                #Update weights 2-3
                for neuron_index1 in range(self.num_nodes[1]):
                    for neuron_index2 in range(self.num_nodes[2]):
                        self.weights[1][neuron_index1][neuron_index2] += alpha * \
                                errors[1][neuron_index2] * self.activations[1][neuron_index1]

                #Update weight 1-2
                for neuron_index1 in range(self.num_nodes[0]):
                    for neuron_index2 in range(self.num_nodes[1]):
                        self.weights[0][neuron_index1][neuron_index2] += alpha * \
                                errors[0][neuron_index2] * self.activations[0][neuron_index1]
                
                sum_errors.append(sum((x-y)**2 for x,y in zip(output_set, result)))
            self.avg_error = sum(sum_errors) / float(len(sum_errors))
            print "Avg. Error:", self.avg_error

    def solve(self, input_data):
        #Layer #1
        self.activations[0] = map(lambda x: x/255.0, input_data + [1])

        #Layer #2
        for node_index in range(self.num_nodes[1]):
            total = 0.0
            for x in range(self.num_nodes[0]):
                total += self.activations[0][x] * self.weights[0][x][node_index]
            self.activations[1][node_index] = self.fn(total)
        #Layer #3
        for node_index in range(self.num_nodes[2]):
            total = sum(self.activations[1][x] * self.weights[1][x][node_index] \
                    for x in range(self.num_nodes[1]))
            self.activations[2][node_index] = self.fn(total)

        return list(self.activations[2])

    def save(self, fileName):
        obj = {
            "alpha": self.alpha,
            "fn": {sigmoid: "sigmoid", tanh: "tanh", new: "new"}[self.fn],
            "fn_": {sigmoid_: "sigmoid_", tanh_: "tanh_", new_: "new_"}[self.fn_],
            "num_nodes": self.num_nodes,
            "weights": self.weights,
            "avg_error": self.avg_error,
        }
        with open(fileName, "w") as f:
            json.dump(obj, f)

    def load(self, fileName):
        with open(fileName) as f:
            obj = json.load(f)
        self.alpha = obj["alpha"]
        self.fn = {"sigmoid":sigmoid, "tanh":tanh, "new":new}[obj["fn"]]
        self.fn_ = {"sigmoid_":sigmoid_, "tanh_":tanh_, "new_":new_}[obj["fn_"]]
        self.num_nodes = obj["num_nodes"]
        self.weights = obj["weights"]
        self.activations = [[0] * x for x in self.num_nodes]

def solve_neural_network(train_data, test_data, hiddenCount, fn=sigmoid, fn_=sigmoid_, alpha=0.8):
    network = NeuralNetwork(train_data, int(hiddenCount), alpha=alpha, fn=sigmoid, fn_=sigmoid_)
    print "Training Neural Network..."
    network.train(train_data)
    for input_data in test_data:
        res = network.solve(input_data.data)
        yield input_data, [0,90,180,270][max(enumerate(res), key=lambda x: x[1])[0]]
    network.save("model-%.5f.json"%network.avg_error)

def solve_best(train_data, test_data, model_file):
    network = NeuralNetwork(train_data, 5, alpha=0.8, fn=sigmoid, fn_=sigmoid_)
    network.load(model_file)
    for input_data in test_data:
        res = network.solve(input_data.data)
        yield input_data, [0,90,180,270][max(enumerate(res), key=lambda x: x[1])[0]]

def main():
    _, train_file, test_file, algorithm, param = sys.argv

    train_data = load_data_file(train_file)
    test_data = load_data_file(test_file)

    confusion_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    algo_func = {"knn": knn, "nnet": solve_neural_network, "best": solve_best}[algorithm]
    for inp, predicted_orientation in algo_func(train_data, test_data, param):
        correct_orientation = inp.orientation
        print "Correct:", correct_orientation
        print "Predicted:", predicted_orientation
        confusion_matrix[correct_orientation/90][predicted_orientation/90] += 1
        print "\n".join("%3d %3d %3d %3d"%tuple(row) for row in confusion_matrix)
        print
    print confusion_matrix


if __name__ == '__main__':
    random.seed()
    main()

