from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math

EPSILON = 1.0e-5
POWER = 3
OMEGA = 2
DELTA = 0.1
K = 2

def linear_kernel(x, y):
    return numpy.dot(x,y) + 1

def polynomial_kernel(x, y):
    return linear_kernel(x, y)**POWER

def radial_basis_function_kernel(x, y):
    diff = numpy.subtract(x, y)
    return math.exp(-(numpy.dot(diff, diff))/(2*OMEGA**2))

def sigmoid_kernel(x, y):
    tmp = numpy.tanh(numpy.dot(x,y))
    print(tmp)
    return tmp

def kernel(x, y):
    #return linear_kernel(x, y)
    #return polynomial_kernel(x, y)
    #return radial_basis_function_kernel(x, y)
    return sigmoid_kernel(x, y)

def calculate_point(i, j, x, y):
    return i * j * kernel(x, y)

def gen_p(datapoints):
    N = len(datapoints)
    mat = numpy.empty(shape = (N, N))
    for i in range(N):
        for j in range(N):
            mat[i][j] = calculate_point(datapoints[i][2], datapoints[j][2],
                    datapoints[i][0:2], datapoints[j][0:2])
    return mat

def gen_datapoints_helper():
    classA = [(random.normalvariate(-1.5, 0.5), random.normalvariate(0.5, 0.5), 1.0)
                for i in range(5)]+ \
                [(random.normalvariate(1.5, 0.5), random.normalvariate(0.5, 0.5), 1.0)
                for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.2), random.normalvariate(0.5, 1.2), -1.0)
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

def gen_q(length):
    return [-1. for i in range(length)]

def gen_h(length):
    return numpy.zeros(shape = (length))

def gen_g(length):
    mat = numpy.zeros(shape = (length, length))
    for i in range(length):
        mat[i][i] = -1
    return mat

def call_qp(P, q, G, h):
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    return alpha

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

def indicator(train, x, y):
    sum = 0
    for alpha, datapoint in train:
        sum += indicator_helper(alpha, datapoint[2], (x, y), datapoint[0:2])
    return sum

def main():
    print("Running main method.")
    print("")
    ############
    # Visualizing the datapoints.
    #
    classA, classB = gen_datapoints_helper()
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
                [p[1] for p in classA],
                'bo')
    pylab.plot([p[0] for p in classB],
                [p[1] for p in classB],
                'ro')
    datapoints = gen_datapoints_from_classes(classA, classB)
    t= train(datapoints)
    xr=numpy.arange(-4, 6, 0.05)
    yr=numpy.arange(-4, 6, 0.05)
    grid=matrix([[indicator(t, x, y) for y in yr] for x in xr])
    pylab.contour(xr, yr, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    pylab.show()


if __name__ == '__main__':
    main()
