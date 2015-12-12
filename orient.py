import sys
import random
import math

from heapq import nlargest
from itertools import izip, imap

sigmoid = lambda u: 1.0 / (1 + math.exp(-u))
sigmoid_ = lambda u: sigmoid(u) * (1 - sigmoid(u))
tanh = lambda u: 2 * Neuron.sigmoid(2 * u) - 1

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


def train_neural_network(train_data, hiddenCount, fn, fn_, alpha):
    featureLength = len(train_data[0].data)
    classLength = 4
    weights = [None, 
        [[random.random() for __ in range(featureLength)] for _ in range(hiddenCount)], 
        [[random.random() for __ in range(hiddenCount)] for _ in range(classLength)]
    ]
    errors = [
        [0] * featureLength,
        [0] * hiddenCount,
        [0] * classLength,
    ]
    o = lambda x: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]][x/90]
    for _ in range(50):
        for input_set, output_set in imap(lambda x: (x.data, o(x.orientation)), train_data):
            a = [input_set, [0]*hiddenCount, [0]*classLength]
            inp = [None, [0]*hiddenCount, [0]*classLength]

            for l in [1, 2]:
                for index, neuron_weights in enumerate(weights[l]):
                    inp[l][index] = dot(neuron_weights, a[l-1])
                    a[l][index] = fn(inp[l][index])

            #Propagate deltas backward.
            for j in range(classLength):
                errors[2][j] = fn_(inp[-1][j]) * (output_set[j] - a[-1][j])

            for l in [1, 0]:
                for index, neuron_weights  in enumerate(weights[l+1]):
                    #import pdb; pdb.set_trace()
                    temp = 0
                    for index1, neuron_weights1 in enumerate(weights[l+1]):
                        temp += weights[l+1][index1][index] * errors[l+1][index1]

                    errors[l][index] = fn_(inp[l+1][index]) * temp

            for l in [1, 2]:
                for neuron_index, neuron_weights in enumerate(weights[l]):
                    weights[l][neuron_index] += alpha * a[l][neuron_index] * errors[l][index]

    return weights

def solve_neural_network(train_data, test_data, hiddenCount, fn=sigmoid, fn_=sigmoid_, alpha=0.2):
    weights = train_neural_network(train_data, hiddenCount, fn=sigmoid, fn_=sigmoid_, alpha=alpha)

    for test in test_data:
        input_arr = test
        for l in [1,2]:
            input_arr = [fn(dot(neuron_weights, input_arr)) for index, neuron_weights in enumerate(weights[l])]
        yield [0,90,180,270][max(enumerate(input_arr), key=lambda x: x[1])[0]]

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

