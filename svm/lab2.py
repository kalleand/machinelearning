from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math

def linear_kernel(x, y):
    return numpy.dot(x,y) + 1

def calculate_point(i, j, x, y):
    return i * j * linear_kernel(x, y)

def gen_p(datapoints):
    N = len(datapoints)
    mat = numpy.empty(shape = (N, N))
    for i in range(N):
        for j in range(N):
            mat[i][j] = calculate_point(datapoints[i][2], datapoints[j][2],
                    datapoints[i][0:2], datapoints[j][0:2])
    return mat

def gen_datapoints_helper():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0)
                for i in range(5)]+ \
                [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0)
                for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0)
                for i in range(10)]
    return classA, classB

def gen_datapoints():
    classA, classB = gen_datapoints_helper()
    data = classA+classB
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

def main():
    print("Running main method.")
    print()
    datapoints = gen_datapoints()
    length = len(datapoints)
    P = gen_p(datapoints)
    print(P)
    q = gen_q(length)
    G = gen_g(length)
    h = gen_h(length)
    print(call_qp(P, q, G, h))
    

    ###########
    # Printing the generated q-vector.
    #
    #print(gen_q(len(datapoints)))

    ###########
    # Printing the generated h-vector.
    #
    #print(gen_h(len(datapoints)))

    ###########
    # Printing the generated h-vector.
    #
    #print(gen_G(len(datapoints)))

    ############
    # Printing the generated p-matrix.
    #
    #print(gen_p(datapoints))

    ############
    # Visualizing the datapoints.
    #
    #classA, classB = gen_datapoints_helper()
    #pylab.hold(True)
    #pylab.plot([p[0] for p in classA],
                #[p[1] for p in classA],
                #'bo')
    #pylab.plot([p[0] for p in classB],
                #[p[1] for p in classB],
                #'ro')
    #pylab.show()

if __name__ == '__main__':
    main()
