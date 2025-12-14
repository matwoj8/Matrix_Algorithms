import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def draw_compression_tree(node, ax, max_depth=None, current_depth=0):
    """
    Rysuje wizualizację drzewa kompresji H-macierzy.
    
    Args:
        node: korzeń drzewa
        ax: osie matplotlib do rysowania
        max_depth: maksymalna głębokość do narysowania (None = bez ograniczeń)
        current_depth: aktualna głębokość (używana rekurencyjnie)
    """
    if max_depth is not None and current_depth > max_depth:
        return
    
    # Oblicz wymiary bloku
    x = node.s_min
    y = node.t_min
    width = node.s_max - node.s_min + 1
    height = node.t_max - node.t_min + 1
    
    # Rysuj prostokąt reprezentujący blok
    if node.sons:
        # Jeśli ma dzieci - to blok podzielony (nie skompresowany)
        rect = Rectangle((x, y), width, height, linewidth=1, 
                        edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
        # Rekurencyjnie rysuj dzieci
        for son in node.sons:
            draw_compression_tree(son, ax, max_depth, current_depth + 1)
    else:
        # Jeśli nie ma dzieci - to blok liścia (skompresowany lub zerowy)
        if node.rank > 0:
            # Blok skompresowany z rank > 0
            color = 'green' if node.rank <= 4 else 'orange'
            rect = Rectangle((x, y), width, height, linewidth=1,
                            edgecolor='darkgreen', facecolor=color, alpha=0.5)
            ax.add_patch(rect)
            
            # Dodaj tekst z rankiem
            # ax.text(x + width/2, y + height/2, f'rank={node.rank}', 
            #        ha='center', va='center', fontsize=8, fontweight='bold')
        else:
            # Blok zerowy
            rect = Rectangle((x, y), width, height, linewidth=1,
                            edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(rect)
            
            # Dodaj tekst "zero"
            # ax.text(x + width/2, y + height/2, 'zero', 
            #        ha='center', va='center', fontsize=8, fontweight='bold')

def visualize_compression_structure(R_root, G_root, B_root, method_name, original_size):
    """
    Tworzy wizualizację struktury kompresji dla wszystkich kanałów.
    
    Args:
        R_root, G_root, B_root: korzenie drzew dla każdego kanału
        method_name: nazwa metody kompresji
        original_size: krotka (wysokość, szerokość)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    channels = ['R (Czerwony)', 'G (Zielony)', 'B (Niebieski)']
    roots = [R_root, G_root, B_root]
    
    for idx, (ax, channel, root) in enumerate(zip(axes, channels, roots)):
        # Rysuj strukturę drzewa
        draw_compression_tree(root, ax, max_depth=5)
        
        # Konfiguruj wykres
        ax.set_xlim(0, original_size[1])
        ax.set_ylim(0, original_size[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Oś Y rosnąca w dół (jak w obrazach)
        ax.set_title(f'Struktura kompresji - Kanał {channel}\nMetoda: {method_name}')
        ax.set_xlabel('Szerokość (piksele)')
        ax.set_ylabel('Wysokość (piksele)')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dodaj legendę
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', alpha=0.7, label='Blok podzielony'),
        Patch(facecolor='green', edgecolor='darkgreen', alpha=0.5, label='Blok rank≤4'),
        Patch(facecolor='orange', edgecolor='darkgreen', alpha=0.5, label='Blok rank>4'),
        Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Blok zerowy')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig(f'compression_structure_{method_name}.png', dpi=300, 
                bbox_inches='tight')
    plt.close(fig)
    print(f"Zapisano wizualizację struktury kompresji: 'compression_structure_{method_name}.png'")

def visualize_single_channel_compression(root, channel_name, method_name, original_size, max_depth=None):
    """
    Tworzy szczegółową wizualizację struktury kompresji dla pojedynczego kanału.
    
    Args:
        root: korzeń drzewa dla kanału
        channel_name: nazwa kanału
        method_name: nazwa metody kompresji
        original_size: krotka (wysokość, szerokość)
        max_depth: maksymalna głębokość do wyświetlenia
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pierwszy wykres: pełna struktura
    draw_compression_tree(root, ax1)
    ax1.set_xlim(0, original_size[1])
    ax1.set_ylim(0, original_size[0])
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title(f'Pełna struktura kompresji\nKanał {channel_name}, Metoda: {method_name}')
    ax1.set_xlabel('Szerokość (piksele)')
    ax1.set_ylabel('Wysokość (piksele)')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Drugi wykres: ograniczona głębokość
    if max_depth is not None:
        draw_compression_tree(root, ax2, max_depth=max_depth)
        ax2.set_xlim(0, original_size[1])
        ax2.set_ylim(0, original_size[0])
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.set_title(f'Struktura kompresji (głębokość ≤ {max_depth})\nKanał {channel_name}, Metoda: {method_name}')
        ax2.set_xlabel('Szerokość (piksele)')
        ax2.set_ylabel('Wysokość (piksele)')
        ax2.grid(True, alpha=0.2, linestyle='--')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Ograniczenie głębokości\nnie zastosowane', 
                ha='center', va='center', fontsize=12, 
                transform=ax2.transAxes)
    
    # Dodaj statystyki
    stats_text = collect_tree_statistics(root)
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'compression_structure_{method_name}_{channel_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Zapisano szczegółową wizualizację: 'compression_structure_{method_name}_{channel_name}.png'")

def collect_tree_statistics(node, stats=None):
    """
    Zbiera statystyki drzewa kompresji.
    
    Returns:
        string ze statystykami
    """
    if stats is None:
        stats = {
            'total_blocks': 0,
            'leaf_blocks': 0,
            'divided_blocks': 0,
            'zero_blocks': 0,
            'compressed_blocks': 0,
            'max_rank': 0,
            'min_block_size': float('inf'),
            'max_block_size': 0
        }
    
    stats['total_blocks'] += 1
    
    # Oblicz rozmiar bloku
    block_size = (node.t_max - node.t_min + 1) * (node.s_max - node.s_min + 1)
    stats['min_block_size'] = min(stats['min_block_size'], block_size)
    stats['max_block_size'] = max(stats['max_block_size'], block_size)
    
    if node.sons:
        stats['divided_blocks'] += 1
        for son in node.sons:
            collect_tree_statistics(son, stats)
    else:
        stats['leaf_blocks'] += 1
        if node.rank == 0:
            stats['zero_blocks'] += 1
        else:
            stats['compressed_blocks'] += 1
            stats['max_rank'] = max(stats['max_rank'], node.rank)
    
    if stats == stats:  # Sprawdzenie czy to korzeń
        stats_text = f"""
        STATYSTYKI STRUKTURY KOMPRESJI:
        - Całkowita liczba bloków: {stats['total_blocks']}
        - Bloki liście: {stats['leaf_blocks']}
        - Bloki podzielone: {stats['divided_blocks']}
        - Bloki skompresowane: {stats['compressed_blocks']}
        - Bloki zerowe: {stats['zero_blocks']}
        - Maksymalny rank: {stats['max_rank']}
        - Minimalny rozmiar bloku: {stats['min_block_size']} pikseli
        - Maksymalny rozmiar bloku: {stats['max_block_size']} pikseli
        """
        return stats_text

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
        ('r8_delta_sigma1', 8, sigma_R[0]),  # r = 1, delta = sigma1
        ('r8_delta_sigma2k', 8, sigma_R[2**k - 1]),  # r = 1, delta = sigma(2^k)
        ('r8_delta_sigma2k2', 8, sigma_R[2**(k-1) - 1]),  # r = 1, delta = sigma(2^k/2)
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
        
        # Stwórz wizualizację struktury kompresji
        print("Tworzenie wizualizacji struktury kompresji...")
        visualize_compression_structure(R_root, G_root, B_root, method_name, (N, M))
        
        # Stwórz szczegółową wizualizację dla kanału R
        visualize_single_channel_compression(R_root, 'R', method_name, (N, M), max_depth=4)

    print(f"\n{'='*50}")
    print("WSZYSTKIE METODY KOMPRESJI ZOSTAŁY WYKONANE!")
    print("Utworzono następujące pliki:")
    print("- singular_values.png - wykres wartości osobliwych")
    print("- original_R.png, original_G.png, original_B.png - oryginalne kanały")
    print("- original_RGB.png - oryginalny obraz")
    print("- Dla każdej metody: compressed_[nazwa]_R/G/B/RGB.png - skompresowane wersje")
    print("- Dla każdej metody: compression_structure_[nazwa].png - wizualizacja struktury")
    print("- Dla każdej metody: compression_structure_[nazwa]_R.png - szczegółowa wizualizacja kanału R")

if __name__ == '__main__':
    main()