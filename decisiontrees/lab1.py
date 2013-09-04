import sys
import random
sys.path.append('dectrees-py/')
import monkdata as monk
import dtree as dt
# Needed import for drawing the decision tree.
#import drawtree as drawtree

print("Entropy for monk1 dataset is {}".format(dt.entropy(monk.monk1)))
print("Entropy for monk2 dataset is {}".format(dt.entropy(monk.monk2)))
print("Entropy for monk3 dataset is {}".format(dt.entropy(monk.monk3)))
print()

datasets = [monk.monk1, monk.monk2, monk.monk3]
for i in range(1, 4):
    dataset = datasets[i - 1]
    print("Average gain for monk{} for each attribute".format(i))
    print("a1 = {0}, a2 = {1}, a3 = {2}, a4 = {3}, a5 = {4}, a6 = {5}".format(dt.averageGain(dataset, monk.attributes[0]), dt.averageGain(dataset, monk.attributes[1]), dt.averageGain(dataset, monk.attributes[2]), dt.averageGain(dataset, monk.attributes[3]), dt.averageGain(dataset, monk.attributes[4]), dt.averageGain(dataset, monk.attributes[5])))

# It is pretty clear that a5 is the best for splitting. We gain over 0.2 (does
# this mean 20% certain?) for this one alone on monk1 and monk3.

monk1a5 = [dt.select(monk.monk1, monk.attributes[4], 1), dt.select(monk.monk1, monk.attributes[4], 2), dt.select(monk.monk1, monk.attributes[4], 3), dt.select(monk.monk1, monk.attributes[4], 4)]
for i in range(1, 5):
    monk1 = monk1a5[i - 1]
    print()
    print("Average gain for monk1 where a5 = {} for each attribute".format(i))
    print("a1 = {0}, a2 = {1}, a3 = {2}, a4 = {3}, a6 = {4}".format(dt.averageGain(monk1, monk.attributes[0]), dt.averageGain(monk1, monk.attributes[1]), dt.averageGain(monk1, monk.attributes[2]), dt.averageGain(monk1, monk.attributes[3]), dt.averageGain(monk1, monk.attributes[5])))
    print("Majority class = {}".format(dt.mostCommon(monk1)))

# Best attribute for monk1. <- a5
#print()
#print()
#print(dt.bestAttribute(monk.monk1, monk.attributes))

#print()
#print()
#print(dt.mostCommon(dt.select(dt.select(monk.monk1, monk.attributes[0], 1),
    #monk.attributes[1], 1)))

# Building the decision tree.
tree1 = dt.buildTree(monk.monk1, monk.attributes)
tree2 = dt.buildTree(monk.monk2, monk.attributes)
tree3 = dt.buildTree(monk.monk3, monk.attributes)
# Drawing the decision tree.
#drawtree.drawTree(tree)

print()
datasets = [monk.monk1, monk.monk1test, monk.monk2, monk.monk2test, monk.monk3, monk.monk3test]
trees = [tree1, tree2, tree3]
for i in range(1, 4):
    tree = trees[i - 1]
    dataset1 = datasets[(i - 1) * 2]
    dataset2 = datasets[(i - 1) * 2 + 1]
    print("Error for Monk{} on train = {} and on test {}.".format(i, dt.check(tree, dataset1), dt.check(tree, dataset2)))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakpoint = int(len(ldata) * fraction)
    return ldata[:breakpoint], ldata[breakpoint:]


print()

for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    #Creating the test and evaluation sets for monk1 and monk3
    monk1train, monk1val = partition(monk.monk1, i)
    monk3train, monk3val = partition(monk.monk3, i)

    # Building the original decisiontrees.
    ptree1 = dt.buildTree(monk1train, monk.attributes)
    ptree3 = dt.buildTree(monk3train, monk.attributes)

    # Creating every possible pruned version of the tree.
    pruned1 = dt.allPruned(ptree1)
    pruned3 = dt.allPruned(ptree3)

    # Getting the unpruned value.
    check1 = dt.check(ptree1, monk1val)
    check3 = dt.check(ptree3, monk1val)

    maxVal = check1
    maxTree = ptree1

    for pTree in pruned1:
        temp = dt.check(pTree, monk1val)
        if temp >= maxVal:
            maxVal = temp
            maxTree = pTree

    print()
    print("Pruning on monk{} with fraction {} gives best performance = {} and without pruning = {}.".format(1, i, dt.check(maxTree, monk.monk1test), dt.check(tree1, monk.monk1test))) 
    maxVal = check3
    maxTree = ptree3

    for pTree in pruned3:
        temp = dt.check(pTree, monk1val)
        if temp >= maxVal:
            maxVal = temp
            maxTree = pTree

    print("Pruning on monk{} with fraction {} gives best performance = {} and without pruning = {}.".format(3, i, dt.check(maxTree, monk.monk3test), dt.check(tree3, monk.monk3test))) 

