import bz2

def read_data(filename):
    speeches = []
    with bz2.open(filename, "rt") as f:
        for line in f:
            tokens = line.split()
            speeches.append((tokens[0], tokens[1:]))
    return speeches

import math

class Classifier(object):

    def __init__(self):
        self.classes = set()
        self.vocabulary = set()
        self.pc = {}
        self.pw = {}

    @staticmethod
    def train(data):
        cls = Classifier()
        # Compute the raw frequencies.
        for c, document in data:
            cls.classes.add(c)
            if c not in cls.pc:
                cls.pc[c] = 0
                cls.pw[c] = {}
            cls.pc[c] += 1
            for w in document:
                cls.vocabulary.add(w)
                if w not in cls.pw[c]:
                    cls.pw[c][w] = 0
                cls.pw[c][w] += 1
        # Add-k smoothing.
        k = 1
        for c in cls.classes:
            for w in cls.vocabulary:
                if w not in cls.pw[c]:
                    cls.pw[c][w] = 0
                cls.pw[c][w] += k
        # Compute the class probabilities.
        c_total = sum(cls.pc.values())
        for c in cls.classes:
            cls.pc[c] = math.log(cls.pc[c] / c_total)
        # Compute the word probabilities.
        for c in cls.classes:
            w_total = sum(cls.pw[c].values())
            for w in cls.vocabulary:
                cls.pw[c][w] = math.log(cls.pw[c][w] / w_total)
        # Return the cls
        return cls

    def predict(self, document):
        probs = {c: self.pc[c] for c in self.classes}
        for w in document:
            if w in self.vocabulary:
                for c in self.classes:
                    probs[c] += self.pw[c][w]
        return max(sorted(self.classes), key=lambda c: probs[c])

def accuracy(classifier, data):
    n_documents = 0
    n_correct = 0
    for c, document in data:
        n_documents += 1
        n_correct += classifier.predict(document) == c
    if n_documents == 0:
        return 0
    else:
        return n_correct / n_documents

def baseline_accuracy(train_data, test_data):
    counts = {}
    for c, document in train_data:
        if c not in counts:
            counts[c] = 0
        counts[c] += 1
    mfc = max(sorted(counts), key=lambda c: counts[c])
    n_documents = 0
    n_correct = 0
    for c, document in test_data:
        n_documents += 1
        n_correct += mfc == c
    if n_documents == 0:
        return 0
    else:
        return n_correct / n_documents
