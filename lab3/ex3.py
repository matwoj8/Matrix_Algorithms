import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple
import os

class Node:
    def __init__(self, t_min, t_max, s_min, s_max, rank=0):
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
    data = np.array(img, dtype=np.uint8)
    R = data[:, :, 0].astype(np.float64)
    G = data[:, :, 1].astype(np.float64)
    B = data[:, :, 2].astype(np.float64)
    return [R, G, B]

def truncatedSVD(M, t_min, t_max, s_min, s_max, r):
    block = M[t_min: t_max + 1, s_min: s_max + 1]
    U, sigma, V = np.linalg.svd(block, full_matrices=False)
    return [U, sigma, V, block]

def CompressMatrix(M, t_min, t_max, s_min, s_max, r):
    [U, sigma, Vh, block] = truncatedSVD(M, t_min, t_max, s_min, s_max, r + 1)

    if np.allclose(block, 0):
        v = Node(t_min, t_max, s_min, s_max)
        return v

    actual_rank = min(r, len(sigma))

    v = Node(t_min, t_max, s_min, s_max, rank=actual_rank)
    v.singularvalues = sigma[:actual_rank]
    v.U = U[:, :actual_rank]
    D = np.diag(sigma[:actual_rank])
    v.V = D @ Vh[:actual_rank, :]

    return v

def CreateTree(M, t_min, t_max, s_min, s_max, r, e):
    [U, sigma, Vh, block] = truncatedSVD(M, t_min, t_max, s_min, s_max, r + 1)

    current_size = min(block.shape)

    if current_size <= r:
        is_admissible = True
    else:
        if sigma[r] < e:
            is_admissible = True
        else:
            is_admissible = False

    if is_admissible:
        v = CompressMatrix(M, t_min, t_max, s_min, s_max, r)
    else:
        v = Node(t_min, t_max, s_min, s_max)

        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2

        v.sons.append(CreateTree(M, t_min, t_new_max, s_min, s_new_max, r, e))
        v.sons.append(CreateTree(M, t_min, t_new_max, s_new_max + 1, s_max, r, e))
        v.sons.append(CreateTree(M, t_new_max + 1, t_max, s_min, s_new_max, r, e))
        v.sons.append(CreateTree(M, t_new_max + 1, t_max, s_new_max + 1, s_max, r, e))

    return v

def reconstruct_matrix(node):
    if node.sons:
        TL = reconstruct_matrix(node.sons[0])
        TR = reconstruct_matrix(node.sons[1])
        BL = reconstruct_matrix(node.sons[2])
        BR = reconstruct_matrix(node.sons[3])

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
    print(f"Skompresowany obraz zapisano jako: '{filename}'")

def save_original_channels(R_orig, G_orig, B_orig):
    """Zapisuje oryginalne kanały jako obrazy w skali szarości"""
    R_orig_uint8 = np.clip(R_orig, 0, 255).astype(np.uint8)
    G_orig_uint8 = np.clip(G_orig, 0, 255).astype(np.uint8)
    B_orig_uint8 = np.clip(B_orig, 0, 255).astype(np.uint8)

    Image.fromarray(R_orig_uint8, 'L').save("original_R.png")
    Image.fromarray(G_orig_uint8, 'L').save("original_G.png")
    Image.fromarray(B_orig_uint8, 'L').save("original_B.png")
    
    # Zapisz oryginalny obraz RGB
    original_rgb = np.stack([R_orig_uint8, G_orig_uint8, B_orig_uint8], axis=2)
    Image.fromarray(original_rgb, 'RGB').save("original_RGB.png")
    
    print("Zapisano oryginalne kanały R, G, B oraz obraz RGB")

def save_compressed_channels(R_root, G_root, B_root, method_name):
    """Zapisuje skompresowane kanały jako obrazy w skali szarości"""
    R_comp = reconstruct_matrix(R_root)
    G_comp = reconstruct_matrix(G_root)
    B_comp = reconstruct_matrix(B_root)

    R_img = Image.fromarray(np.clip(R_comp, 0, 255).astype(np.uint8), 'L')
    G_img = Image.fromarray(np.clip(G_comp, 0, 255).astype(np.uint8), 'L')
    B_img = Image.fromarray(np.clip(B_comp, 0, 255).astype(np.uint8), 'L')

    R_img.save(f"compressed_{method_name}_R.png")
    G_img.save(f"compressed_{method_name}_G.png")
    B_img.save(f"compressed_{method_name}_B.png")
    print(f"Zapisano skompresowane kanały dla metody: {method_name}")

def plot_singular_values(sigma_R, sigma_G, sigma_B):
    """Rysuje wykres wartości osobliwych dla kanałów R, G, B"""
    plt.figure(figsize=(15, 10))
    
    # Wykres liniowy
    plt.subplot(2, 2, 1)
    plt.plot(sigma_R, 'r-', label='Kanał R', alpha=0.7, linewidth=2)
    plt.plot(sigma_G, 'g-', label='Kanał G', alpha=0.7, linewidth=2)
    plt.plot(sigma_B, 'b-', label='Kanał B', alpha=0.7, linewidth=2)
    plt.xlabel('Indeks wartości osobliwej')
    plt.ylabel('Wartość osobliwa')
    plt.title('Wartości osobliwe kanałów R, G, B\n(skala liniowa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wykres logarytmiczny
    plt.subplot(2, 2, 2)
    plt.semilogy(sigma_R, 'r-', label='Kanał R', alpha=0.7, linewidth=2)
    plt.semilogy(sigma_G, 'g-', label='Kanał G', alpha=0.7, linewidth=2)
    plt.semilogy(sigma_B, 'b-', label='Kanał B', alpha=0.7, linewidth=2)
    plt.xlabel('Indeks wartości osobliwej')
    plt.ylabel('Wartość osobliwa (skala log)')
    plt.title('Wartości osobliwe kanałów R, G, B\n(skala logarytmiczna)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wykres pierwszych 50 wartości
    plt.subplot(2, 2, 3)
    n_show = min(50, len(sigma_R))
    plt.plot(range(n_show), sigma_R[:n_show], 'r-', label='Kanał R', alpha=0.7, linewidth=2)
    plt.plot(range(n_show), sigma_G[:n_show], 'g-', label='Kanał G', alpha=0.7, linewidth=2)
    plt.plot(range(n_show), sigma_B[:n_show], 'b-', label='Kanał B', alpha=0.7, linewidth=2)
    plt.xlabel('Indeks wartości osobliwej')
    plt.ylabel('Wartość osobliwa')
    plt.title('Pierwsze 50 wartości osobliwych\n(skala liniowa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statystyki
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    STATYSTYKI WARTOŚCI OSOBLIWYCH:
    
    Kanał R:
    - σ₁ = {sigma_R[0]:.2f}
    - σ₁₀ = {sigma_R[9]:.2f}
    - σ₅₀ = {sigma_R[49]:.2f}
    - Liczba wartości: {len(sigma_R)}
    
    Kanał G:
    - σ₁ = {sigma_G[0]:.2f}
    - σ₁₀ = {sigma_G[9]:.2f}
    - σ₅₀ = {sigma_G[49]:.2f}
    - Liczba wartości: {len(sigma_G)}
    
    Kanał B:
    - σ₁ = {sigma_B[0]:.2f}
    - σ₁₀ = {sigma_B[9]:.2f}
    - σ₅₀ = {sigma_B[49]:.2f}
    - Liczba wartości: {len(sigma_B)}
    """
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('singular_values.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_compression_visualization(R_root, G_root, B_root, method_name, original_size):
    """Tworzy wizualizację kompresji dla danej metody"""
    R_comp = reconstruct_matrix(R_root)
    G_comp = reconstruct_matrix(G_root)
    B_comp = reconstruct_matrix(B_root)
    
    R_final = np.clip(R_comp, 0, 255).astype(np.uint8)
    G_final = np.clip(G_comp, 0, 255).astype(np.uint8)
    B_final = np.clip(B_comp, 0, 255).astype(np.uint8)
    
    # Stwórz obraz RGB
    rgb_compressed = np.stack([R_final, G_final, B_final], axis=2)
    
    # Zapisz obrazy
    Image.fromarray(R_final, 'L').save(f"compressed_{method_name}_R.png")
    Image.fromarray(G_final, 'L').save(f"compressed_{method_name}_G.png")
    Image.fromarray(B_final, 'L').save(f"compressed_{method_name}_B.png")
    Image.fromarray(rgb_compressed, 'RGB').save(f"compressed_{method_name}_RGB.png")
    
    print(f"Zapisano wizualizację dla metody: {method_name}")

def main():
    # Wczytaj obraz
    print("Wczytywanie obrazu...")
    RGB_matrices = RGB()
    R_orig, G_orig, B_orig = RGB_matrices
    N, M = R_orig.shape

    print(f"Wymiary obrazu: {N}x{M}")

    # Zapisz oryginalne kanały i obraz RGB
    print("Zapisywanie oryginalnych kanałów...")
    save_original_channels(R_orig, G_orig, B_orig)

    # Oblicz pełne SVD dla każdego kanału
    print("Obliczanie SVD dla kanału R...")
    U_R, sigma_R, V_R = np.linalg.svd(R_orig, full_matrices=False)
    print("Obliczanie SVD dla kanału G...")
    U_G, sigma_G, V_G = np.linalg.svd(G_orig, full_matrices=False)
    print("Obliczanie SVD dla kanału B...")
    U_B, sigma_B, V_B = np.linalg.svd(B_orig, full_matrices=False)

    # Narysuj wykres wartości osobliwych
    print("Tworzenie wykresu wartości osobliwych...")
    plot_singular_values(sigma_R, sigma_G, sigma_B)

    # Określ k
    n_singular = min(N, M)
    k = int(np.log2(n_singular))
    print(f"Liczba wartości osobliwych: {n_singular}, k = {k}")

    # Metody kompresji - 6 wariantów
    methods = [
        
        # kombinacja
        ('r1_delta_sigma1', 1, sigma_R[0]),  # r = 1, delta = sigma1
        ('r1_delta_sigma2k', 1, sigma_R[2**k - 1]),  # r = 1, delta = sigma(2^k)
        ('r1_delta_sigma2k2', 1, sigma_R[2**(k-1) - 1]),  # r = 1, delta = sigma(2^k/2)
        ('r4_delta_sigma1', 4, sigma_R[0]),  # r = 4, delta = sigma1
        ('r4_delta_sigma2k', 4, sigma_R[2**k - 1]),  # r = 4, delta = sigma(2^k)
        ('r4_delta_sigma2k2', 4, sigma_R[2**(k-1) - 1]),  # r = 4, delta = sigma(2^k/2)
    ]

    print(f"\nWartości progowe dla metod delta:")
    print(f"  sigma1: {sigma_R[0]:.2f}")
    print(f"  sigma(2^k): {sigma_R[2**k - 1]:.2f} (indeks {2**k - 1})")
    print(f"  sigma(2^k/2): {sigma_R[2**(k-1) - 1]:.2f} (indeks {2**(k-1) - 1})")

    # Kompresja dla każdej metody
    for method_name, r, e in methods:
        print(f"\n{'='*50}")
        print(f"Kompresja metodą: {method_name}")
        print(f"Parametry: r={r}, e={e:.2f}")

        # Kompresja każdego kanału
        print("Kompresja kanału R...")
        R_root = CreateTree(R_orig, 0, N-1, 0, M-1, r, e)
        print("Kompresja kanału G...")
        G_root = CreateTree(G_orig, 0, N-1, 0, M-1, r, e)
        print("Kompresja kanału B...")
        B_root = CreateTree(B_orig, 0, N-1, 0, M-1, r, e)

        # Odtworzenie i zapisanie wyników
        create_compression_visualization(R_root, G_root, B_root, method_name, (N, M))

    print(f"\n{'='*50}")
    print("WSZYSTKIE METODY KOMPRESJI ZOSTAŁY WYKONANE!")
    print("Utworzono następujące pliki:")
    print("- singular_values.png - wykres wartości osobliwych")
    print("- original_R.png, original_G.png, original_B.png - oryginalne kanały")
    print("- original_RGB.png - oryginalny obraz")
    print("- Dla każdej metody: compressed_[nazwa]_R/G/B/RGB.png - skompresowane wersje")

if __name__ == '__main__':
    main()