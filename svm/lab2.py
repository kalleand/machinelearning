from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math

EPSILON = 1.0e-5
POWER = 3
OMEGA = 2
DELTA = 0.1
K = 0.05

#############################
# Generates the datapoints.
#
# NOTICE: We need to temper with these values for different kernel tricks to
# make sure it can find an optimal solution. 
def gen_datapoints_helper():
    classA = [(random.normalvariate(-1.5, 0.3), random.normalvariate(0.5, 0.3), 1.0)
                for i in range(5)] + [(random.normalvariate(1.5, 0.3), 
                random.normalvariate(0.5, 0.3), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.1), random.normalvariate(-0.5, 0.1), -1.0)
                for i in range(10)]
    return classA, classB

def gen_datapoints():
    classA, classB = gen_datapoints_helper()
    data = classA + classB
    random.shuffle(data)
    return data

def gen_datapoints_from_classes(classA, classB):
    data = classA + classB
    random.shuffle(data)
    return data

# End of methods generating data.
#
#############################

# Linear kernel trick. Can only produce a linear boundary.
def linear_kernel(x, y):
    return numpy.dot(x,y) + 1

# Polynomial kernel trick that produces boundaries that are not a linear plane.
def polynomial_kernel(x, y):
    return linear_kernel(x, y)**POWER

# The Radial Basis function Kernel trick. OMEGA determines the smoothness.
def radial_basis_function_kernel(x, y):
    diff = numpy.subtract(x, y)
    return math.exp(-(numpy.dot(diff, diff))/(2*OMEGA**2))

# Sigmoid kernel trick. K should be around 1/N and delta should not be a large
# negative.
def sigmoid_kernel(x, y):
    return numpy.tanh(numpy.dot(numpy.multiply(K, x) ,y) + DELTA)

# Wrapper for the kernel function to be used. Uncomment the kernel trick.
def kernel(x, y):
    #return linear_kernel(x, y)
    #return polynomial_kernel(x, y)
    #return radial_basis_function_kernel(x, y)
    return sigmoid_kernel(x, y)

# Calculates the value for a specified point in the P matrix.
def calculate_point(i, j, x, y):
    return i * j * kernel(x, y)

# Generates P, this is where the kernel trick is used for every element in the
# return vector of size NxN.
def gen_p(datapoints):
    N = len(datapoints)
    mat = numpy.empty(shape = (N, N))
    for i in range(N):
        for j in range(N):
            mat[i][j] = calculate_point(datapoints[i][2], datapoints[j][2],
                    datapoints[i][0:2], datapoints[j][0:2])
    return mat

# Generate q. this is a vector of length N with -1 as its elements.
def gen_q(length):
    return [-1. for i in range(length)]

# Generate h. this is a vector of length N with 0 as its elements.
def gen_h(length):
    return numpy.zeros(shape = (length))

# Generate G, this is -1 multiplied with the identity matrix of size N.
def gen_g(length):
    mat = numpy.zeros(shape = (length, length))
    for i in range(length):
        mat[i][i] = -1
    return mat

# Wrapper for qp where the matrices are converted in the correct way.
def call_qp(P, q, G, h):
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    return alpha

# Training the SVM by taking a dataset and generating the needed matrices for
# the call to qp.
#
# Returns a tuple of (alpha, datapoint).
def train(datapoints):
    length = len(datapoints)
    P = gen_p(datapoints)
    q = gen_q(length)
    G = gen_g(length)
    h = gen_h(length)
    l = call_qp(P, q, G, h)
    ret = []
    for i,d in zip(l, datapoints):
        if abs(i) > EPSILON:
            ret.append((i,d))
    return ret

def indicator_helper(alpha, t, x_star, x_i):
    return alpha * t * kernel(x_star, x_i)

# Classifies a point (x, y) with a training set.
def indicator(train, x, y):
    sum = 0
    for alpha, datapoint in train:
        sum += indicator_helper(alpha, datapoint[2], (x, y), datapoint[0:2])
    return sum

def main():
    print("Running main method.")
    print("")
    
    # Generate the datapoints and plot them.
    classA, classB = gen_datapoints_helper()
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
                [p[1] for p in classA],
                'bo')
    pylab.plot([p[0] for p in classB],
                [p[1] for p in classB],
                'ro')

    # Add the sets together and shuffle them around.
    datapoints = gen_datapoints_from_classes(classA, classB)

    t= train(datapoints)

    # Plot the decision boundaries.
    xr=numpy.arange(-4, 4, 0.05)
    yr=numpy.arange(-4, 4, 0.05)
    grid=matrix([[indicator(t, x, y) for y in yr] for x in xr])
    pylab.contour(xr, yr, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Now that we are done we show the ploted datapoints and the decision
    # boundary.
    pylab.show()


if __name__ == '__main__':
    main()
