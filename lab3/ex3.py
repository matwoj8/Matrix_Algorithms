import numpy as np
from PIL import Image
from typing import Tuple

class Node:
    def __init__(self, t_min, t_max, s_min, s_max, rank = 0):

        self.t_min = t_min
        self.t_max = t_max
        self.s_min = s_min
        self.s_max = s_max
        self.size = (t_max - t_min + 1, s_max - s_min + 1)
        self.rank = rank
        self.singularvalues = None
        self.sons = []
        self.U = None
        self.V = None


def RGB():
    img = Image.open("photo.jpg").convert('RGB')
    data = np.array(img, dtype = np.uint8)
    R = data[:, :, 0].astype(np.float64)
    G = data[:, :, 1].astype(np.float64)
    B = data[:, :, 2].astype(np.float64)

    return [R,G,B]

def truncatedSVD(M, t_min, t_max, s_min, s_max, r):
    block = M[t_min: t_max + 1, s_min: s_max + 1]
    U, sigma, V = np.linalg.svd(block, full_matrices=False)
    return [U, sigma, V, block]


def CompressMatrix(M, t_min, t_max, s_min, s_max, r):
    [U, sigma, Vh, block] = truncatedSVD(M, t_min, t_max, s_min, s_max, r + 1)

    if np.allclose(block, 0): #1
        v = Node(t_min, t_max, s_min, s_max) #2
        return v #3

    actual_rank = min(r, len(sigma))

    v = Node (t_min, t_max, s_min, s_max, rank = actual_rank)  #6
    v.singularvalues = sigma[:actual_rank] #7
    v.U = U[:,:actual_rank] #8
    D = np.diag(sigma[:actual_rank])
    v.V = D @ Vh[:actual_rank,:] #9

    return v # 11

def CreateTree(M, t_min, t_max, s_min, s_max, r, e):
    [U, sigma, Vh, block] = truncatedSVD(M, t_min, t_max, s_min, s_max, r + 1) #1

    current_size = min(block.shape)

    if current_size <= r:
        is_admissible = True
    else:
        if sigma[r] < e:
            is_admissible = True
        else:
            is_admissible = False


    if is_admissible:
        v = CompressMatrix(M, t_min, t_max, s_min, s_max, r) #3
    else: #4
        v = Node(t_min, t_max, s_min, s_max) #5

        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2

        v.sons.append(CreateTree(M, t_min, t_new_max, s_min, s_new_max, r, e)) #6
        v.sons.append(CreateTree(M, t_min, t_new_max, s_new_max + 1, s_max, r, e)) #7
        v.sons.append(CreateTree(M, t_new_max + 1, t_max, s_min, s_new_max, r, e)) #8
        v.sons.append(CreateTree(M, t_new_max + 1, t_max, s_new_max + 1, s_max, r, e)) #9

    return v # 11


def reconstruct_matrix(node):
    if node.sons:
        TL = reconstruct_matrix(node.sons[0])  # Top-Left
        TR = reconstruct_matrix(node.sons[1])  # Top-Right
        BL = reconstruct_matrix(node.sons[2])  # Bottom-Left
        BR = reconstruct_matrix(node.sons[3])  # Bottom-Right

        Top_row = np.hstack((TL, TR))
        Bottom_row = np.hstack((BL, BR))
        return np.vstack((Top_row, Bottom_row))
    else:
        if node.rank == 0:
            rows, cols = node.size
            return np.zeros((rows, cols), dtype=np.float64)
        else:
            return node.U @ node.V


def reconstruct_bitmap(R_root, G_root, B_root, original_size, filename):
    R_comp = reconstruct_matrix(R_root)
    G_comp = reconstruct_matrix(G_root)
    B_comp = reconstruct_matrix(B_root)

    R_final = np.clip(R_comp, 0, 255).astype(np.uint8)
    G_final = np.clip(G_comp, 0, 255).astype(np.uint8)
    B_final = np.clip(B_comp, 0, 255).astype(np.uint8)

    compressed_data = np.stack([R_final, G_final, B_final], axis=2)

    img_comp = Image.fromarray(compressed_data, 'RGB')

    img_comp.save(filename)
    print("Skompresowany obraz zapisano jako: '{filename}'")


def main():
    r = 10
    e = 50.0
    RGB_matrices = RGB()

    R_orig, G_orig, B_orig = RGB_matrices
    N, M = R_orig.shape

    print(f"Wymiary macierzy: {N}x{M}. Parametry: r={r}, e={e}")

    print("\nBudowanie drzewa R...")
    R_root = CreateTree(R_orig, 0, N - 1, 0, M - 1, r, e)
    print("Budowanie drzewa G...")
    G_root = CreateTree(G_orig, 0, N - 1, 0, M - 1, r, e)
    print("Budowanie drzewa B...")
    B_root = CreateTree(B_orig, 0, N - 1, 0, M - 1, r, e)

    reconstruct_bitmap(R_root, G_root, B_root, (N, M),f"skompresowany_r{r}_e{e}.png")

if __name__ == '__main__':
    main()