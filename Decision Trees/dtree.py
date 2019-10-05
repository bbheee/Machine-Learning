import math
import random
import numpy as np

def entropy(dataset):
    "Calculate the entropy of a dataset"
    n = len(dataset)
    nPos = len([x for x in dataset if x.positive])
    nNeg = n - nPos
    if nPos == 0 or nNeg == 0:
        return 0.0
    return -float(nPos)/n * log2(float(nPos)/n) + \
        -float(nNeg)/n * log2(float(nNeg)/n)


def averageGain(dataset, attribute):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset) * len(subset)
    return entropy(dataset) - weighted/len(dataset)


def log2(x):
    "Logarithm, base 2"
    return math.log(x, 2)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    return [x for x in dataset if x.attribute[attribute] == value]


def bestAttribute(dataset, attributes):
    "Attribute with highest expected information gain"
    gains = [(averageGain(dataset, a), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]


def allPositive(dataset):
    "Check if all samples are positive"
    return all([x.positive for x in dataset])


def allNegative(dataset):
    "Check if all samples are negative"
    return not any([x.positive for x in dataset])


def mostCommon(dataset):
    "Majority class of the dataset"
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount


class TreeNode:
    "Decision tree representation"

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        "Produce readable (string) representation of the tree"
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class TreeLeaf:
    "Decision tree representation for leaf nodes"

    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        "Produce readable (string) representation of this leaf"
        if self.cvalue:
            return '+'
        return '-'


def buildTree(dataset, attributes, maxdepth=1000000):
    "Recursively build a decision tree"

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)
    a = bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return TreeNode(a, dict(branches), default)


def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def allPruned(tree):
    "Return a list of trees, each with one node replaced by the corresponding default class"
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives

def reducedErrorPrune(traindata, testdata, attributes, fractions, N):
    "Returns mean error and variance of the pruned trees using validation data"
    meanErr = np.empty([6,1], dtype = float)
    variance = np.empty([6,1], dtype = float)
    n = 0
    for fraction in fractions:
        currentErr = list()
        for i in range(0, N):
            train, validation = partition(traindata, fraction)
            t = buildTree(train, attributes)
            pt = allPruned(t)
            tPerform = 0
            maxPerform = 0
            bestPerform = 0
            for j in range(0, len(pt)):
                tPerform = check(pt[j], validation)
                if(tPerform > maxPerform):
                    maxPerform = tPerform
                    bestPerform = j
            currentErr.append(1 - check(pt[bestPerform], testdata))
        meanErr[n] = np.mean(currentErr)
        variance[n] = np.var(currentErr)
        n = n + 1
    return meanErr, variance
