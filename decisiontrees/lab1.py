import sys
sys.path.append('dectrees-py/')
import monkdata as monk
import dtree as dt

print("Entropy for monk1 dataset is {}".format(dt.entropy(monk.monk1)))
print("Entropy for monk2 dataset is {}".format(dt.entropy(monk.monk2)))
print("Entropy for monk3 dataset is {}".format(dt.entropy(monk.monk3)))

datasets = [monk.monk1, monk.monk2, monk.monk3]
for i in range(1, 4):
    dataset = datasets[i - 1]
    print("Average gain for monk{} for each attribute".format(i))
    print("a1 = {0} ,a2 = {1}, a3 = {2} ,a4 = {3} ,a5 = {4}, a6 = {5}".format(dt.averageGain(dataset, monk.attributes[0]), dt.averageGain(dataset, monk.attributes[1]), dt.averageGain(dataset, monk.attributes[2]), dt.averageGain(dataset, monk.attributes[3]), dt.averageGain(dataset, monk.attributes[4]), dt.averageGain(dataset, monk.attributes[5])))

# It is pretty clear that a5 is the best for splitting. We gain over 0.2 (does
# this mean 20% certain?) for this one alone on monk1 and monk3.

t = dt.buildTree(monk.monk1, monk.attributes)
print(dt.check(t, monk.monk1test))
