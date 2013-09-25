from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

LINEAR = 0
POLYNOMIAL = 1
RADIAL_BASIS = 2
SIGMOID = 3
EPSILON = 1.0e-5
POWER = 3
OMEGA = 2
DELTA = 0.1
K = 0.05
C = 20

# Wrapper for the kernel function to be used.
def kernel(x, y, kernel_trick):
    if kernel_trick == LINEAR:
        return linear_kernel(x, y)
    elif kernel_trick == POLYNOMIAL:
        return polynomial_kernel(x, y)
    elif kernel_trick == RADIAL_BASIS:
        return radial_basis_function_kernel(x, y)
    elif kernel_trick == SIGMOID:
        return sigmoid_kernel(x, y)
    else:
        return 0

#############################
# Generates the datapoints.
#
# NOTICE: We need to temper with these values for different kernel tricks to
# make sure it can find an optimal solution. Not needed when using a
# sufficiently large slack variable.
def gen_datapoints_helper():
    classA = [(random.normalvariate(-2.5, 1.0), random.normalvariate(1.6, 0.8), 1.0)
                for i in range(5)] + [(random.normalvariate(2.5, 0.8), 
                random.normalvariate(2.0, 1.5), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 1.0), random.normalvariate(-1.2, 1.4), -1.0)
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


# Calculates the value for a specified point in the P matrix.
def calculate_point(i, j, x, y, kernel_trick):
    return i * j * kernel(x, y, kernel_trick)

# Generates P, this is where the kernel trick is used for every element in the
# return vector of size NxN.
def gen_p(datapoints, kernel_trick):
    N = len(datapoints)
    mat = numpy.empty(shape = (N, N))
    for i in range(N):
        for j in range(N):
            mat[i][j] = calculate_point(datapoints[i][2], datapoints[j][2],
                    datapoints[i][0:2], datapoints[j][0:2], kernel_trick)
    return mat

# Generate q. this is a vector of length N with -1 as its elements.
def gen_q(length):
    return [-1. for i in range(length)]

# Generate h. this is a vector of length N with 0 as its elements.
def gen_h(length):
    h = numpy.zeros(shape = (2*length))
    for i in range(length):
        h[i+length] = C
    return h

# Generate G, this is -1 multiplied with the identity matrix of size N.
def gen_g(length):
    mat = numpy.zeros(shape = (2*length, length))
    for i in range(length):
        mat[i][i] = -1
        mat[i+length][i] = 1
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
def train(datapoints, kernel_trick):
    length = len(datapoints)
    P = gen_p(datapoints, kernel_trick)
    q = gen_q(length)
    G = gen_g(length)
    h = gen_h(length)
    l = call_qp(P, q, G, h)
    ret = []
    for i,d in zip(l, datapoints):
        if abs(i) > EPSILON:
            ret.append((i,d))
    return ret

def indicator_helper(alpha, t, x_star, x_i, kernel_trick):
    return alpha * t * kernel(x_star, x_i, kernel_trick)

# Classifies a point (x, y) with a training set.
def indicator(train, x, y, kernel_trick):
    sum = 0
    for alpha, datapoint in train:
        sum += indicator_helper(alpha, datapoint[2], (x, y), datapoint[0:2],
                kernel_trick)
    return sum

def kernel_name(kernel_trick):
    if kernel_trick == LINEAR:
        return 'Linear kernel'
    elif kernel_trick == POLYNOMIAL:
        return 'Polynomial kernel'
    elif kernel_trick == RADIAL_BASIS:
        return 'Radial Basis Function kernel'
    elif kernel_trick == SIGMOID:
        return 'Sigmoid kernel'
    else:
        return 'Fail'


def main():
    print("Running main method.")
    print("")
    
    # Generate the datapoints and plot them.
    classA, classB = gen_datapoints_helper()

    kernels = [LINEAR, POLYNOMIAL, RADIAL_BASIS, SIGMOID]
    for k in kernels:
        pylab.figure()
        pylab.title(kernel_name(k))
        pylab.plot([p[0] for p in classA],
                    [p[1] for p in classA],
                    'bo')
        pylab.plot([p[0] for p in classB],
                    [p[1] for p in classB],
                    'ro')

        # Add the sets together and shuffle them around.
        datapoints = gen_datapoints_from_classes(classA, classB)

        t= train(datapoints, k)

        # Plot the decision boundaries.
        xr=numpy.arange(-4, 4, 0.05)
        yr=numpy.arange(-4, 4, 0.05)
        grid=matrix([[indicator(t, x, y, k) for y in yr] for x in xr])
        pylab.contour(xr, yr, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Now that we are done we show the ploted datapoints and the decision
    # boundary.
    pylab.show()


if __name__ == '__main__':
    main()
