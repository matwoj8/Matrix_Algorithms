from lab1.ex1 import *
import numpy as np


def inv(A, method, cnt = 0):
    A = np.array(A)
    m, n = A.shape

    if m == 1 and n == 1:
        return np.array([[1 / A[0][0]]]), cnt + 1

    if m <= 2 and n <= 2:
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        A = 1 / det * np.array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]])
        return A, cnt + 7

    A_11, A_12 = A[:m // 2, :n // 2], A[:m // 2, n // 2:]
    A_21, A_22 = A[m // 2:, :n // 2], A[m // 2:, n // 2:]

    A_11_inv, cnt = inv(A_11, method, cnt)

    temp1, cnt = rec(A_21, A_11_inv, method, cnt)
    temp2, cnt = rec(temp1, A_12, method, cnt)
    S = A_22 - temp2 # S_22
    S_inv, cnt = inv(S, method, cnt) # S_22_inv

    temp3, cnt = rec(A_11_inv, A_12, method, cnt)
    temp4, cnt = rec(temp3, S_inv, method, cnt)
    temp5, cnt = rec(temp4, A_21, method, cnt)
    temp6, cnt = rec(temp5, A_11_inv, method, cnt)
    top_left = A_11_inv + temp6 # B_11

    temp7, cnt = rec(A_11_inv, A_12, method, cnt)
    temp8, cnt = rec(temp7, S_inv, method, cnt)
    top_right = - temp8 # B_12

    temp9, cnt = rec(S_inv, A_21, method, cnt)
    temp10, cnt = rec(temp9, A_11_inv, method, cnt)
    bottom_left = - temp10 # B_21

    bottom_right = S_inv # B_22

    cnt += 4 * (m // 2) ** 2

    new_A = np.block([[top_left, top_right], [bottom_left, bottom_right]])
    return new_A, cnt

def LU(A, method, cnt = 0):
    A = np.array(A)
    n = A.shape[0]

    if n == 1:
        L = np.array([[1.0]])
        U = np.array([[A[0, 0]]])
        return L, U, cnt
    if n == 2:
        if n == 2:
            L = np.eye(2)
            U = np.zeros((2, 2))

            U[0, 0] = A[0, 0]
            U[0, 1] = A[0, 1]
            L[1, 0] = A[1, 0] / U[0, 0]
            U[1, 1] = A[1, 1] - L[1, 0] * U[0, 1]

            cnt += 3
            return L, U, cnt

    if n % 2 == 0:
        A_11, A_12 = A[:n // 2, :n // 2], A[:n // 2, n // 2:]
        A_21, A_22 = A[n // 2:, :n // 2], A[n // 2:, n // 2:]
    else:
        A_11, A_12 = A[:n - 1, :n - 1], A[:n - 1, n - 1:]
        A_21, A_22 = A[n - 1:, :n - 1], A[n - 1:, n - 1:]
    
    L_11, U_11, cnt = LU(A_11, method, cnt)

    U_11_rev, cnt = inv(U_11, method, cnt)
    L_21, cnt = rec(A_21, U_11_rev, method, cnt)

    L_11_rev, cnt = inv(L_11, method, cnt)
    U_12, cnt = rec(L_11_rev, A_12, method, cnt)

    temp1, cnt = rec(A_21, U_11_rev, method, cnt)
    temp2, cnt = rec(temp1, L_11_rev, method, cnt)
    temp3, cnt = rec(temp2, A_12, method, cnt)
    S = A_22 - temp3
    L_22, U_22, cnt = LU(S, method, cnt)

    if n % 2 == 0:
        L_12 = np.zeros((n // 2, n // 2))
        U_21 = np.zeros((n // 2, n // 2))
    else:
        L_12 = np.zeros((n - 1, 1))
        U_21 = np.zeros((1, n - 1))

    L = np.block([[L_11, L_12], [L_21, L_22]])
    U = np.block([[U_11, U_12], [U_21, U_22]])

    return L, U, cnt


def det_(A, method, cnt = 0):
    n = A.shape[0]
    if n == 1:
        cnt += 1
        return np.linalg.det(A), cnt
    if n == 2:
        cnt += 3
        return np.linalg.det(A), cnt

    L, U, cnt = LU(A, method, cnt)

    det_A = 1
    for i in range(n):
        # det_A *= L[i][i]
        det_A *= U[i][i]
        cnt += 1

    return det_A, cnt

def gauss_rec(A, b, method, cnt = 0):
    A = np.array(A)
    b = np.array(b)
    n = A.shape[0]

    if n == 1:
        x = b[0] / A[0][0]
        cnt += 1
        return np.array([x]), cnt
    if n == 2:
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        x1 = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        x2 = (b[1] * A[0, 0] - b[0] * A[1, 0]) / det
        return np.array([x1, x2]), cnt + 11

    half = n // 2
    A_11 = A[:half, :half]
    A_12 = A[:half, half:]
    A_21 = A[half:, :half]
    A_22 = A[half:, half:]
    b_1 = b[:half]
    b_2 = b[half:]

    L_11, U_11, cnt = LU(A_11, method, cnt)
    L_11_inv, cnt = inv(L_11, method, cnt)
    U_11_inv, cnt = inv(U_11, method, cnt)

    tmp1, cnt = rec(A_21, U_11_inv, method, cnt)
    tmp2, cnt = rec(tmp1, L_11_inv, method, cnt)
    tmp3, cnt = rec(tmp2, A_12, method, cnt)
    S = A_22 - tmp3
    cnt += half * half

    L_S, U_S, cnt = LU(S, method, cnt)
    L_S_inv, cnt = inv(L_S, method, cnt)
    C_11 = U_11
    C_12, cnt = rec(L_11_inv, A_12, method, cnt)
    C_22 = U_S

    RHS1, cnt = rec(L_11_inv, b_1.reshape(-1,1), method, cnt)
    RHS1 = RHS1.flatten()

    tmp4, cnt = rec(L_S_inv, b_2.reshape(-1,1), method, cnt)
    tmp4 = tmp4.flatten()

    tmp5, cnt = rec(L_S_inv, A_21, method, cnt)
    tmp6, cnt = rec(tmp5, U_11_inv, method, cnt)
    tmp7, cnt = rec(tmp6, L_11_inv, method, cnt)
    tmp8, cnt = rec(tmp7, b_1.reshape(-1,1), method, cnt)
    tmp8 = tmp8.flatten()

    RHS2 = tmp4 - tmp8
    cnt += half * half

    x2, cnt = gauss_rec(C_22, RHS2, method, cnt)
    tmp, cnt = rec(C_12, x2.reshape(-1,1), method, cnt)
    tmp2 = RHS1 - tmp.flatten()
    x1, cnt = gauss_rec(C_11, tmp2, method, cnt)
    cnt += half * half

    x = np.concatenate([x1, x2])
    return x, cnt


def plot_all_inv(max_n=183, step=2):
    ns = range(3, max_n, step)
    times_binet, times_strass = [], []
    counts_binet, counts_strass = [], []
    mem_binet, mem_strass = [], []

    for n in ns:
        A = np.random.rand(n, n)

        start = time.time()
        _, c1 = inv(A, "Binet")
        times_binet.append(time.time() - start)
        counts_binet.append(c1)
        mem_binet.append((A.nbytes) / 1024**2)

        start = time.time()
        _, c2 = inv(A, "Strassen")
        times_strass.append(time.time() - start)
        counts_strass.append(c2)
        mem_strass.append((A.nbytes) / 1024**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ns, times_binet, label="Binet")
    axes[0].plot(ns, times_strass, label="Strassen")
    axes[0].set_xlabel("Rozmiar n")
    axes[0].set_ylabel("Czas [s]")
    axes[0].set_title("Porównanie czasu działania")
    axes[0].legend()

    axes[1].plot(ns, counts_binet, label="Binet")
    axes[1].plot(ns, counts_strass, label="Strassen")
    axes[1].set_xlabel("Rozmiar n")
    axes[1].set_ylabel("Liczba operacji (count)")
    axes[1].set_title("Porównanie liczby operacji")
    axes[1].legend()

    axes[2].plot(ns, mem_binet, label="Binet")
    axes[2].plot(ns, mem_strass, label="Strassen")
    axes[2].set_xlabel("Rozmiar n")
    axes[2].set_ylabel("Zużycie pamięci [MB]")
    axes[2].set_title("Porównanie zużycia pamięci")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def plot_all_gauss_rec(max_n=183, step=2):
    ns = range(3, max_n, step)
    times_binet, times_strass = [], []
    counts_binet, counts_strass = [], []
    mem_binet, mem_strass = [], []

    for n in ns:
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        start = time.time()
        _, c1 = gauss_rec(A, b, "Binet")
        times_binet.append(time.time() - start)
        counts_binet.append(c1)
        mem_binet.append((A.nbytes + b.nbytes) / 1024**2)

        start = time.time()
        _, c2 = gauss_rec(A, b, "Strassen")
        times_strass.append(time.time() - start)
        counts_strass.append(c2)
        mem_strass.append((A.nbytes + b.nbytes) / 1024**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ns, times_binet, label="Binet")
    axes[0].plot(ns, times_strass, label="Strassen")
    axes[0].set_xlabel("Rozmiar n")
    axes[0].set_ylabel("Czas [s]")
    axes[0].set_title("Porównanie czasu działania")
    axes[0].legend()

    axes[1].plot(ns, counts_binet, label="Binet")
    axes[1].plot(ns, counts_strass, label="Strassen")
    axes[1].set_xlabel("Rozmiar n")
    axes[1].set_ylabel("Liczba operacji (count)")
    axes[1].set_title("Porównanie liczby operacji")
    axes[1].legend()

    axes[2].plot(ns, mem_binet, label="Binet")
    axes[2].plot(ns, mem_strass, label="Strassen")
    axes[2].set_xlabel("Rozmiar n")
    axes[2].set_ylabel("Zużycie pamięci [MB]")
    axes[2].set_title("Porównanie zużycia pamięci")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def plot_all_LU(max_n=183, step=2):
    ns = range(3, max_n, step)
    times_binet, times_strass = [], []
    counts_binet, counts_strass = [], []
    mem_binet, mem_strass = [], []

    for n in ns:
        A = np.random.rand(n, n)

        start = time.time()
        _, _, c1 = LU(A, "Binet")
        times_binet.append(time.time() - start)
        counts_binet.append(c1)
        mem_binet.append((A.nbytes) / 1024**2)

        start = time.time()
        _, _, c2 = LU(A, "Strassen")
        times_strass.append(time.time() - start)
        counts_strass.append(c2)
        mem_strass.append((A.nbytes) / 1024**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ns, times_binet, label="Binet")
    axes[0].plot(ns, times_strass, label="Strassen")
    axes[0].set_xlabel("Rozmiar n")
    axes[0].set_ylabel("Czas [s]")
    axes[0].set_title("Porównanie czasu działania")
    axes[0].legend()

    axes[1].plot(ns, counts_binet, label="Binet")
    axes[1].plot(ns, counts_strass, label="Strassen")
    axes[1].set_xlabel("Rozmiar n")
    axes[1].set_ylabel("Liczba operacji (count)")
    axes[1].set_title("Porównanie liczby operacji")
    axes[1].legend()

    axes[2].plot(ns, mem_binet, label="Binet")
    axes[2].plot(ns, mem_strass, label="Strassen")
    axes[2].set_xlabel("Rozmiar n")
    axes[2].set_ylabel("Zużycie pamięci [MB]")
    axes[2].set_title("Porównanie zużycia pamięci")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
plot_all_inv()
plot_all_gauss_rec()
plot_all_LU()
