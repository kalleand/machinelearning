from animate import draw
from random import random, randint
from numpy import log

class Environment:
    def __init__ (self, state):
        self.state = state
        self.trans = self.get_trans()
        self.rew = self.get_rew()

    def go(self, move):
        r = self.rew[self.state][move]
        self.state = self.trans[self.state][move]
        return self.state, r

    def get_loop(self, Q, s):
        it = s
        visited = [0 for i in range(16)]
        while 1:
            if visited[it]:
                start_loop = it
                break
            else:
                visited[it] = 1
                it = self.trans[it][argmax(lambda a: Q[it][a], range(4))]

        res = [start_loop]
        it = self.trans[it][argmax(lambda a: Q[it][a], range(4))]
        while it != start_loop:
            res.append(it)
            it = self.trans[it][argmax(lambda a: Q[it][a], range(4))]
        return res


    def get_trans(self):
        return ((1,3,4,12),# 0
                (0,2,5,13),# 1
                (3,1,6,14),# 2
                (2,0,7,15),# 3
                (5,7,0,8),# 4
                (4,6,1,9),# 5
                (7,5,2,10),# 6
                (6,4,3,11),# 7
                (9,11,12,4),# 8
                (8,10,13,5),# 9
                (11,9,14,6),# 10
                (10,8,15,7),# 11
                (13,15,8,0),# 12
                (12,14,9,1),# 13
                (15,13,10,2),# 14
                (14,12,11,3))# 15

    def get_rew(self):
        length = len(self.trans)
        base = 0
        rew = [[base for la in range(4)] for lb in range(16)]
        point = 10

        # Good moves.
        rew[13][3] += point
        rew[7][1] += point
        rew[14][3] += point
        rew[11][1] += point

        # Bad moves
        rew[1][3] += -1 * point
        rew[4][1] += -1 * point
        rew[2][3] += -1 * point
        rew[8][1] += -1 * point

        # Punishments for both legs in air
        # To state 5
        rew[1][2] += -1 * point
        rew[4][0] += -1 * point
        rew[6][1] += -1 * point
        rew[9][3] += -1 * point

        # To state 6
        rew[2][2] += -1 * point
        rew[5][1] += -1 * point
        rew[7][0] += -1 * point
        rew[10][3] += -1 * point

        # To state 9
        rew[5][3] += -1 * point
        rew[8][0] += -1 * point
        rew[10][1] += -1 * point
        rew[13][2] += -1 * point

        # To state 10
        rew[6][3] += -1 * point
        rew[9][1] += -1 * point
        rew[11][0] += -1 * point
        rew[14][2] += -1 * point

        # Because we do not condone sliding.
        rew[0][1] += -1 * point
        rew[0][3] += -1 * point
        rew[3][1] += -1 * point
        rew[3][3] += -1 * point
        rew[12][3] += -1 * point
        rew[12][1] += -1 * point
        rew[15][3] += -1 * point
        rew[15][1] += -1 * point

        return rew


def argmax(f, args):
    mi = None
    m = -1e10
    for i in args:
        v = f(i)
        if v >= m:
            m = v
            mi = i
    return mi

def max(f, args):
    m = -1e10
    for i in args:
        v = f(i)
        if v >= m:
            m = v
    return m

def calculate_route(T, n, gamma, env, s, epsilon):
    number_of_states = 16 # number of states
    number_of_transitions = 4 # number of transitions
    q_mat = [[100 for ia in range(number_of_transitions)] for ib in
            range(number_of_states)]

    state = s
    for step in range(T):
        if(random() < epsilon):
            move = randint(0,3)
        else:
            move = argmax(lambda tmp: q_mat[state][tmp], range(4))
        new_state, reward = env.go(move)
        q_prim = max(lambda ne: q_mat[new_state][ne], range(4))
        q_mat[state][move] = q_mat[state][move] + n * (reward + (gamma * q_prim) -  q_mat[state][move])
        state = new_state
    return q_mat

def main():
    # Get the variables used to do reinforcement learning
    gamma = 0.7
    T = 1000000
    epsilon = 0.2
    s = 12
    n = 0.1
    e = Environment(s)

    Q = calculate_route(T, n, gamma, e, s, epsilon)
    print("calculation done")

    draw(e.get_loop(Q, s))

if __name__ == '__main__':
    main()
