import numpy as np
import matplotlib.pyplot as plt
import numpy.random

numpy.random.seed(2023)

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

# TODO: dopytac sie o to

# d)
print()
print("D")

def print_for_d(matrix):
    for i in range(len(matrix)):
        print(f"v{i+1} ", end="")
        print(f"[", end="")
        for j in range(len(matrix[0])):
            if(j == len(matrix[0])-1):
                print(f"{matrix[i][j]}]")
                break
            print(f"{matrix[i][j]}, ", end="")

def encode(v, G):
    # wektor v podawany poziomo
    return np.dot(v, G).T % 5

encoded_arr = []

for i in range(len(matrix[0])):
    encoded_arr.append(encode(matrix[:, i], G))

# print(encoded_arr)
print_for_d(encoded_arr)

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
print_for_d(encoded_arr_simulated)

# f)
print()
print("F")


def hamming_distance(v1, v2):
    return np.count_nonzero(v1 - v2)

# def minimize_hamming_distance(G, v):
#     B = []
#     for i in G:
#         B.append(i)
#     field = [0, 1, 2, 3, 4]
#     w = []
#     for a in field:
#         for b in field:
#             for c in field:
#                 for d in field:
#                     u = (a * B[0] + b * B[1] + c * B[2] + d * B[3]) % 5
#                     w.append((hamming_distance(u, v), u))
#     m = min(w, key=lambda x: x[0])[0]
#     target_w = [x[1] for x in w if x[0] == m]
#     target_w_multiplied = []
#     for i in target_w:
#         target_w_multiplied.append(np.dot(G, i) % 5)
#     return target_w_multiplied

def find_coordinates_in_basis(v, B): # brute force
    # B - baza 4-wymiarowa
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    u = (a * B[0] + b * B[1] + c * B[2] + d * B[3]) % 5
                    if np.array_equal(u, v):
                        return [a, b, c, d]
    return None

def minimize_hamming_distance_v2(G, v):
    B = []
    for i in G:
        B.append(i)
    field = [0, 1, 2, 3, 4]
    C = []
    for a in field:
        for b in field:
            for c in field:
                for d in field:
                    u = (a * B[0] + b * B[1] + c * B[2] + d * B[3]) % 5
                    C.append(u)
                    # C zawiera wszystkie wektory z przestrzeni L(B) jako wiersze

    min_dist = hamming_distance(min(C, key = lambda x: hamming_distance(x, v)), v)
#    min_dist_vec_first = min(C, key = lambda x: hamming_distance(x, v)) # pierwszy z brzegu taki wektor
    L = []
    for u in C:
        if hamming_distance(u, v) == min_dist:
            L.append(u)
    ind = np.random.randint(0, len(L))
    w = L[ind]
    return find_coordinates_in_basis(w, B)


decoded_arr = [minimize_hamming_distance_v2(G,v) for v in encoded_arr_simulated]
print_for_d(decoded_arr)

# g)
print()
print("G")

new_matrix = np.array(decoded_arr).T
print(new_matrix)

# h)
print()
print("H")

properly_decoded_counter = 0
for i in range(len(matrix[0])):
    if np.array_equal(matrix[:, i], new_matrix[:, i]):
        properly_decoded_counter += 1
print(f"Properly decoded: {properly_decoded_counter} out of {len(matrix[0])}")

# i)
print()
print("I")

new_matrix_normed = new_matrix / 4

print(new_matrix_normed)
plt.imshow(new_matrix_normed, cmap='hot', interpolation='nearest')
plt.show()


