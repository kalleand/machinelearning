import sys
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
t = dt.buildTree(monk.monk1, monk.attributes)
# Drawing the decision tree.
#drawtree.drawTree(t)

print()
print(dt.check(t, monk.monk1test))
