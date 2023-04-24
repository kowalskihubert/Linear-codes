import numpy as np
import matplotlib.pyplot as plt

# a)
print("A")
matrix = np.random.randint(0, 5, (4, 10))

print(matrix)

# b)
print()
print("B")
matrix_normed = matrix / 4

print(matrix_normed)
plt.imshow(matrix_normed, cmap='hot', interpolation='nearest')
plt.show()

# c)
print()
print("C")
G = np.fromstring("1 0 0 0 0 4 4 2 0 1 1 0 1 0 0 0 3 0 2 2 1 0 0 0 1 0 0 2 0 1 1 1 1 0 0 0 1 1 0 0 0 4 3 0", dtype=int,
                  sep=" ")
G = G.reshape(4, 11)
print(G)

# d)
print()
print("D")


# TODO: dopytac sie o to
def encode(v, G):
    # wektor v podawany poziomo
    return np.dot(v, G).T % 5


encoded_arr = []
for i in range(len(matrix[0])):
    encoded_arr.append(encode(matrix[:, i], G))

print(encoded_arr)

# e)
print()
print("E")


def send_simulation(v):
    for i in range(len(v)):
        if np.random.randint(0, 100) < 5:
            v[i] = (v[i] + 3) % 5
        else:
            pass
    return v


encoded_arr_simulated = [send_simulation(v) for v in encoded_arr]
print(encoded_arr_simulated)

# f)
print()
print("F")


def hamming_distance(v1, v2):
    return np.count_nonzero(v1 - v2)


def minimize_hamming_distance(G, v):
    B = []
    for i in G:
        B.append(i)
    field = [0, 1, 2, 3, 4]
    w = []
    for a in field:
        for b in field:
            for c in field:
                for d in field:
                    u = (a * B[0] + b * B[1] + c * B[2] + d * B[3]) % 5
                    w.append((hamming_distance(u, v), u))
    m = min(w, key=lambda x: x[0])[0]
    target_w = [x[1] for x in w if x[0] == m]
    target_w_multiplied = []
    for i in target_w:
        target_w_multiplied.append(np.dot(G, i) % 5)
    return target_w_multiplied


encoded_arr_simulated_minimized = [minimize_hamming_distance(G,v) for v in encoded_arr_simulated]

# g)
print()
print("G")

new_matrix = np.array(encoded_arr_simulated_minimized).ravel().reshape(4, 10)
print(new_matrix)

# i)
print()
print("I")

new_matrix_normed = new_matrix / 4

print(new_matrix_normed)
plt.imshow(new_matrix_normed, cmap='hot', interpolation='nearest')
plt.show()