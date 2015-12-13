import sys
import random
import math

from heapq import nlargest
from itertools import izip, imap

sigmoid = lambda u: 1.0 / (1 + math.exp(-u))
sigmoid_ = lambda u: sigmoid(u) * (1 - sigmoid(u))
tanh = lambda u: math.tanh(u)
tanh_ = lambda u: 1 - math.tanh(u)**2
new = lambda u: 1.7159*math.tanh(2.0 * u / 3.0) 
new_ = lambda u: 1.14393 * tanh_(2.0 * u / 3.0)

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
                        self.weights[1][neuron_index1][neuron_index2] += self.alpha * \
                                errors[1][neuron_index2] * self.activations[1][neuron_index1]

                #Update weight 1-2
                for neuron_index1 in range(self.num_nodes[0]):
                    for neuron_index2 in range(self.num_nodes[1]):
                        self.weights[0][neuron_index1][neuron_index2] += self.alpha * \
                                errors[0][neuron_index2] * self.activations[0][neuron_index1]
                
                sum_errors.append(sum((x-y)**2 for x,y in zip(output_set, result)))
            print "Avg. Error:", sum(sum_errors) / float(len(sum_errors))

    def solve(self, input_data):
        #Layer #1
        self.activations[0] = input_data + [1]

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

def solve_neural_network(train_data, test_data, hiddenCount, fn=sigmoid, fn_=sigmoid_, alpha=0.5):
    network = NeuralNetwork(train_data, hiddenCount, alpha=alpha, fn=new, fn_=new_)
    print "Training Neural Network..."
    network.train(train_data)
    for input_data in test_data:
        res = network.solve(input_data.data)
        yield input_data, [0,90,180,270][max(enumerate(res), key=lambda x: x[1])[0]]

def main():
    _, train_file, test_file, algorithm, param = sys.argv

    train_data = load_data_file(train_file)
    test_data = load_data_file(test_file)

    confusion_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    algo_func = {"knn": knn, "nnet": solve_neural_network}[algorithm]
    for inp, predicted_orientation in algo_func(train_data, test_data, int(param)):
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

