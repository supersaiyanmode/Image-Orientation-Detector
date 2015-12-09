import sys
import math
from heapq import nlargest

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
        yield data, nlargest(k, map(lambda x: (-euclidean_dist(x, data),x), train_data))

def load_data_file(fileName):
    return [TestData(line[0], int(line[1]), map(int,line[2:])) for line in map(lambda x: x.strip().split(), open(fileName))]


def main():
    _, train_file, test_file, algorithm, param = sys.argv

    train_data = load_data_file(train_file)
    test_data = load_data_file(test_file)

    confusion_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for inp, closest in knn(train_data, test_data, int(param)):
        orientations = map(lambda x: x[1].orientation, closest)
        correct_orientation = inp.orientation
        predicted_orientation = max(set(orientations), key=orientations.count)
        print "Correct:", correct_orientation
        print "Predicted:", predicted_orientation
        confusion_matrix[correct_orientation/90][predicted_orientation/90] += 1
        print "\n".join("%3d %3d %3d %3d"%tuple(row) for row in confusion_matrix)
        print
    print confusion_matrix


if __name__ == '__main__':
    main()

