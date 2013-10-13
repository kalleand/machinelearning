from animate import draw

def argmax(f, args):
    mi = None
    m = -1e10
    for i in args:
        v = f(i)
        if v > m:
            m = v
            mi = i
    return mi

def calculate_route(T, trans, rew, gamma, length):
    policy = [None for s in trans]
    value = [0 for s in trans]
    for p in range(T):
        for s in range(length):
            policy[s] = argmax( lambda a: rew[s][trans[s][a]] + gamma * value[trans[s][a]], range(4))

        for s in range(length):
            a = policy[s]
            value[s] = rew[s][trans[s][a]] + gamma * value[trans[s][a]]

    print(value)
    return policy

def get_trans():
    return ((1,3,4,12),
            (0,2,5,13),
            (3,1,6,14),
            (2,0,7,15),
            (5,7,0,8),
            (4,6,1,9),
            (7,5,2,10),
            (6,4,3,11),
            (9,11,12,4),
            (8,10,13,5),
            (11,9,14,6),
            (10,8,15,7),
            (13,15,8,0),
            (12,14,9,1),
            (15,13,10,2),
            (14,12,11,3))

def get_rew(length):
    rew = [[0 for la in range(length)] for lb in range(length)]
    point = 1
    # Good moves.
    rew[13][1] = 1 * point
    rew[7][4] = 1 * point
    rew[14][2] = 1 * point
    rew[11][8] = 1 * point
    # Bad moves
    rew[1][13] = -1 * point
    rew[4][7] = -1 * point
    rew[2][14] = -1 * point
    rew[8][11] = -1 * point
    # Punishments...
    for i in range(length):
        rew[i][5] = -1 * point
        rew[i][6] = -1 * point
        rew[i][9] = -1 * point
        rew[i][10] = -1 * point

    # Because we do not condone sliding.
    # Like you are some kind of animal
    rew[0][3] = -1 * point / 2
    rew[0][12] = -1 * point / 2
    rew[3][0] = -1 * point / 2
    rew[3][15] = -1 * point / 2
    rew[12][0] = -1 * point / 2
    rew[12][15] = -1 * point / 2
    rew[15][12] = -1 * point / 2
    rew[15][3] = -1 * point / 2
    return rew

def get_loop(trans, route, start):
    length = len(route)
    visited = [0 for i in route]
    i = start
    while 1:
        if(visited[i] != 0):
            loop_start = i
            break
        visited[i] = 1
        i = trans[i][route[i]]
    i = trans[i][route[i]]
    res = [loop_start]
    while (i != loop_start):
        res.append(i)
        i = trans[i][route[i]]
    return res

def main():
    trans = get_trans()
    length = len(trans)
    rew = get_rew(length)
    gamma = 0.5
    T = 100
    policy = [None for s in trans]
    value = [0 for s in trans]
    route = calculate_route(100, trans, rew, gamma, length)
    for i in range(length):
        print("%d: %d" % (i, trans[i][route[i]]))
    draw(get_loop(trans, route, 0))




if __name__ == '__main__':
    main()
# We want this behaviour.
#draw((12, 13, 1, 2, 3, 7, 4, 8))
