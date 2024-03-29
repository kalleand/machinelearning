import sys
import random
import matplotlib.pyplot as plt

# Importing lab specific packages.
sys.path.append('dectrees-py/')
import monkdata as monk
import dtree as dt

# Needed import for drawing the decision tree.
#import drawtree as drawtree

# Datasets
train = [monk.monk1, monk.monk2, monk.monk3]
test = [monk.monk1test, monk.monk2test, monk.monk3test]

print("Entropy for monk1 dataset is {}".format(dt.entropy(monk.monk1)))
print("Entropy for monk2 dataset is {}".format(dt.entropy(monk.monk2)))
print("Entropy for monk3 dataset is {}".format(dt.entropy(monk.monk3)))

for i, dataset in enumerate(train):
    print("")
    print("Average gain for monk{} for each attribute".format(i + 1))
    for j, attribute in enumerate(monk.attributes):
        print("a{} = {}".format(j + 1, dt.averageGain(dataset, attribute)))

monk1a5 = [dt.select(monk.monk1, monk.attributes[4], 1), dt.select(monk.monk1, monk.attributes[4], 2), dt.select(monk.monk1, monk.attributes[4], 3), dt.select(monk.monk1, monk.attributes[4], 4)]

for i, monk1 in enumerate(monk1a5):
    print("")
    print("Average gain for monk1 where a5 = {} for each attribute".format(i + 1))
    for j, attribute in enumerate(monk.attributes):
        if j != 4:
            print("a{} = {}".format(j + 1, dt.averageGain(monk1, attribute)))
    print("Majority class = {}".format(dt.mostCommon(monk1)))


# Building the decision tree.
tree1 = dt.buildTree(monk.monk1, monk.attributes)
tree2 = dt.buildTree(monk.monk2, monk.attributes)
tree3 = dt.buildTree(monk.monk3, monk.attributes)
trees = [tree1, tree2, tree3]

# Drawing the decision tree.
#drawtree.drawTree(tree)

print("")
for i, (dataset1, dataset2, tree) in enumerate(zip(train, test, trees)):
    print("Error for Monk{} on train = {} and on test {}.".format(i, dt.check(tree, dataset1), dt.check(tree, dataset2)))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakpoint = int(len(ldata) * fraction)
    return ldata[:breakpoint], ldata[breakpoint:]

def prune(pruned_tree, test_tree):
    currentBase = pruned_tree
    oldVal = 0
    maxVal = 1
    while maxVal > oldVal:
        maxVal = dt.check(currentBase, test_tree)
        oldVal = maxVal
        maxTree = currentBase
        for pTree in dt.allPruned(currentBase):
            temp = dt.check(pTree, test_tree)
            if temp > maxVal:
                maxVal = temp
                maxTree = pTree
        currentBase = maxTree
    return maxTree

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ax = plt.subplot(1, 1, 1)
monk1Pruned = []
monk3Pruned = []

for frac in fractions:

    # Initializes values for the average scores.
    val1 = 0
    val3 = 0

    # Variable for easy handling of iterations.
    number_of_iterations = 100

    for it in range(number_of_iterations):
        #Creating the test and evaluation sets for monk1 and monk3
        monk1train, monk1val = partition(monk.monk1, frac)
        monk3train, monk3val = partition(monk.monk3, frac)

        # Building the original decisiontrees.
        ptree1 = dt.buildTree(monk1train, monk.attributes)
        ptree3 = dt.buildTree(monk3train, monk.attributes)

        # Pruning
        val1 += dt.check(prune(ptree1, monk1val), monk.monk1test)
        val3 += dt.check(prune(ptree3, monk3val), monk.monk3test)

    # Get the average score.
    val1 /= number_of_iterations
    val3 /= number_of_iterations

    # Saves this value.
    monk1Pruned.append(val1)
    monk3Pruned.append(val3)

pruned = [monk1Pruned, [], monk3Pruned]

for monk in [1, 3]:
    print("")
    print("Wihout pruning on monk{} the score is {}".format(monk,
        dt.check(trees[monk - 1], test[monk - 1])))
    print("With pruning the score is:")
    for i, frac in enumerate(fractions):
        print("{} = {}".format(frac, pruned[monk - 1][i]))

ax.plot(fractions, monk1Pruned, label=("MONK1"))
ax.plot(fractions, monk3Pruned, label=("MONK3"))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=4)
plt.xlabel('Fraction')
plt.ylabel('Score')
plt.show()
