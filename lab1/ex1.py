import numpy as np
import matplotlib.pyplot as plt
import time

AlphaTensor = False
original_size2 = None


def padding(A, B, base):
    m, n = A.shape
    l = B.shape[1]

    max_dim = max(m, n, l)
    size = base
    while size < max_dim:
        size *= base

    A_padded = np.zeros((size, size), dtype=A.dtype)
    B_padded = np.zeros((size, size), dtype=B.dtype)
    A_padded[:m, :n] = A
    B_padded[:n, :l] = B

    return A_padded, B_padded


def alpha_tensor_3x3(A, B, cnt):
    a = A.flatten()
    b = B.flatten()
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = a
    b11, b12, b13, b21, b22, b23, b31, b32, b33 = b

    m1 = (a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22
    m2 = (a11 - a21) * (-b12 + b22)
    m3 = a22 * (-b11 + b12 + b21 - b22 - b23 - b31 + b33)
    m4 = (-a11 + a21 + a22) * (b11 - b12 + b22)
    m5 = (a21 + a22) * (-b11 + b12)
    m6 = a11 * b11
    m7 = (-a11 + a31 + a32) * (b11 - b13 + b23)
    m8 = (-a11 + a31) * (b13 - b23)
    m9 = (a31 + a32) * (-b11 + b13)
    m10 = (a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23
    m11 = a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32)
    m12 = (-a13 + a32 + a33) * (b22 + b31 - b32)
    m13 = (a13 - a33) * (b22 - b32)
    m14 = a13 * b31
    m15 = (a32 + a33) * (-b31 + b32)
    m16 = (-a13 + a22 + a23) * (b23 + b31 - b33)
    m17 = (a13 - a23) * (b23 - b33)
    m18 = (a22 + a23) * (-b31 + b33)
    m19 = a12 * b21
    m20 = a23 * b32
    m21 = a21 * b13
    m22 = a31 * b12
    m23 = a33 * b33

    c11 = m6 + m14 + m19
    c12 = m1 + m4 + m5 + m6 + m12 + m14 + m15
    c13 = m6 + m7 + m9 + m10 + m14 + m16 + m18
    c21 = m2 + m3 + m4 + m6 + m14 + m16 + m17
    c22 = m2 + m4 + m5 + m6 + m20
    c23 = m14 + m16 + m17 + m18 + m21
    c31 = m6 + m7 + m8 + m11 + m12 + m13 + m14
    c32 = m12 + m13 + m14 + m15 + m22
    c33 = m6 + m7 + m8 + m9 + m23

    cnt += 136

    C = np.array([[c11, c12, c13],
                  [c21, c22, c23],
                  [c31, c32, c33]])
    return C, cnt


def rec(A, B, method, cnt=0):
    global AlphaTensor, original_size2
    A = np.array(A)
    B = np.array(B)

    if AlphaTensor == False and method == "AlphaTensor":
        original_size2 = A.shape[0]
        AlphaTensor = True
        A, B = padding(A, B, 3)

    m, n = A.shape
    k, l = B.shape

    if m <= 2 or n <= 2 or k <= 2 or l <= 2:
        cnt += 2 * m * n * k - m * n
        return A @ B, cnt

    if method == "AlphaTensor" and m == 3 and n == 3 and k == 3 and l == 3:
        return alpha_tensor_3x3(A, B, cnt)

    if method == "Binet":
        A_11, A_12 = A[:m // 2, :n // 2], A[:m // 2, n // 2:]
        A_21, A_22 = A[m // 2:, :n // 2], A[m // 2:, n // 2:]
        B_11, B_12 = B[:k // 2, :l // 2], B[:k // 2, l // 2:]
        B_21, B_22 = B[k // 2:, :l // 2], B[k // 2:, l // 2:]
    elif method == "Strassen":
        if m % 2 == 0 and n % 2 == 0 and l % 2 == 0:
            A_11, A_12 = A[:m // 2, :n // 2], A[:m // 2, n // 2:]
            A_21, A_22 = A[m // 2:, :n // 2], A[m // 2:, n // 2:]
            B_11, B_12 = B[:k // 2, :l // 2], B[:k // 2, l // 2:]
            B_21, B_22 = B[k // 2:, :l // 2], B[k // 2:, l // 2:]
        else:
            m_even = m - (m % 2)
            n_even = n - (n % 2)
            l_even = l - (l % 2)
            cnt += 3

            A_11, A_12 = A[:m_even, :n_even], A[:m_even, n_even:]
            A_21, A_22 = A[m_even:, :n_even], A[m_even:, n_even:]
            B_11, B_12 = B[:n_even, :l_even], B[:n_even, l_even:]
            B_21, B_22 = B[n_even:, :l_even], B[n_even:, l_even:]


    elif method == "AlphaTensor":
        block_m, block_n = m // 3, n // 3
        block_k, block_l = k // 3, l // 3
        a11 = A[0 * block_m:1 * block_m, 0 * block_n:1 * block_n]
        a12 = A[0 * block_m:1 * block_m, 1 * block_n:2 * block_n]
        a13 = A[0 * block_m:1 * block_m, 2 * block_n:3 * block_n]
        a21 = A[1 * block_m:2 * block_m, 0 * block_n:1 * block_n]
        a22 = A[1 * block_m:2 * block_m, 1 * block_n:2 * block_n]
        a23 = A[1 * block_m:2 * block_m, 2 * block_n:3 * block_n]
        a31 = A[2 * block_m:3 * block_m, 0 * block_n:1 * block_n]
        a32 = A[2 * block_m:3 * block_m, 1 * block_n:2 * block_n]
        a33 = A[2 * block_m:3 * block_m, 2 * block_n:3 * block_n]

        b11 = B[0 * block_k:1 * block_k, 0 * block_l:1 * block_l]
        b12 = B[0 * block_k:1 * block_k, 1 * block_l:2 * block_l]
        b13 = B[0 * block_k:1 * block_k, 2 * block_l:3 * block_l]
        b21 = B[1 * block_k:2 * block_k, 0 * block_l:1 * block_l]
        b22 = B[1 * block_k:2 * block_k, 1 * block_l:2 * block_l]
        b23 = B[1 * block_k:2 * block_k, 2 * block_l:3 * block_l]
        b31 = B[2 * block_k:3 * block_k, 0 * block_l:1 * block_l]
        b32 = B[2 * block_k:3 * block_k, 1 * block_l:2 * block_l]
        b33 = B[2 * block_k:3 * block_k, 2 * block_l:3 * block_l]

    if method == "Binet":
        C_11, cnt = rec(A_11, B_11, "Binet", cnt)
        C_12, cnt = rec(A_11, B_12, "Binet", cnt)
        C_21, cnt = rec(A_21, B_11, "Binet", cnt)
        C_22, cnt = rec(A_21, B_12, "Binet", cnt)
        tmp1, cnt = rec(A_12, B_21, "Binet", cnt)
        tmp2, cnt = rec(A_12, B_22, "Binet", cnt)
        tmp3, cnt = rec(A_22, B_21, "Binet", cnt)
        tmp4, cnt = rec(A_22, B_22, "Binet", cnt)

        C_11 += tmp1
        C_12 += tmp2
        C_21 += tmp3
        C_22 += tmp4

        cnt += 4 * (m // 2) ** 2

        C_top = np.hstack((C_11, C_12))
        C_bot = np.hstack((C_21, C_22))
        C = np.vstack((C_top, C_bot))
        return C, cnt

    if method == "Strassen":
        if m % 2 == 0 and n % 2 == 0 and l % 2 == 0:
            P_1, cnt = rec(A_11 + A_22, B_11 + B_22, "Strassen", cnt)
            P_2, cnt = rec(A_21 + A_22, B_11, "Strassen", cnt)
            P_3, cnt = rec(A_11, B_12 - B_22, "Strassen", cnt)
            P_4, cnt = rec(A_22, B_21 - B_11, "Strassen", cnt)
            P_5, cnt = rec(A_11 + A_12, B_22, "Strassen", cnt)
            P_6, cnt = rec(A_21 - A_11, B_11 + B_12, "Strassen", cnt)
            P_7, cnt = rec(A_12 - A_22, B_21 + B_22, "Strassen", cnt)

            C_11 = P_1 + P_4 - P_5 + P_7
            C_12 = P_3 + P_5
            C_21 = P_2 + P_4
            C_22 = P_1 - P_2 + P_3 + P_6

            cnt += 18 * (m // 2) ** 2

            C_top = np.hstack((C_11, C_12))
            C_bot = np.hstack((C_21, C_22))
            C = np.vstack((C_top, C_bot))
            return C, cnt
        else:

            C_11, cnt = rec(A_11, B_11, "Strassen", cnt)
            C_11 += A_12 @ B_21
            C_12, cnt = rec(A_11, B_12, "Strassen", cnt)
            C_12 += A_12 @ B_22
            C_21, cnt = rec(A_21, B_11, "Strassen", cnt)
            C_21 += A_22 @ B_21
            C_22, cnt = rec(A_21, B_12, "Strassen", cnt)
            C_22 += A_22 @ B_22

            cnt += m_even * l_even  # A_12 @ B_21
            cnt += m_even  # A_12 @ B_22
            cnt += l_even  # A_22 @ B_21
            cnt += 1  # A_22 @ B_22

            cnt += m_even ** 2  # C11 +=
            cnt += m_even  # C12 +=
            cnt += l_even  # C21 +=
            cnt += 1  # C22 +=
            # cnt += 2 * (m - 1) ** 2 + 3 *(m - 1) + 2

            C_top = np.hstack((C_11, C_12))
            C_bot = np.hstack((C_21, C_22))
            C = np.vstack((C_top, C_bot))

            return C, cnt

    if method == "AlphaTensor":
        m1, cnt = rec(a11 + a12 + a13 - a21 - a22 - a32 - a33, b22, "AlphaTensor", cnt)
        m2, cnt = rec(a11 - a21, -b12 + b22, "AlphaTensor", cnt)
        m3, cnt = rec(a22, -b11 + b12 + b21 - b22 - b23 - b31 + b33, "AlphaTensor", cnt)
        m4, cnt = rec(-a11 + a21 + a22, b11 - b12 + b22, "AlphaTensor", cnt)
        m5, cnt = rec(a21 + a22, -b11 + b12, "AlphaTensor", cnt)
        m6, cnt = rec(a11, b11, "AlphaTensor", cnt)
        m7, cnt = rec(-a11 + a31 + a32, b11 - b13 + b23, "AlphaTensor", cnt)
        m8, cnt = rec(-a11 + a31, b13 - b23, "AlphaTensor", cnt)
        m9, cnt = rec(a31 + a32, -b11 + b13, "AlphaTensor", cnt)
        m10, cnt = rec(a11 + a12 + a13 - a22 - a23 - a31 - a32, b23, "AlphaTensor", cnt)
        m11, cnt = rec(a32, -b11 + b13 + b21 - b22 - b23 - b31 + b32, "AlphaTensor", cnt)
        m12, cnt = rec(-a13 + a32 + a33, b22 + b31 - b32, "AlphaTensor", cnt)
        m13, cnt = rec(a13 - a33, b22 - b32, "AlphaTensor", cnt)
        m14, cnt = rec(a13, b31, "AlphaTensor", cnt)
        m15, cnt = rec(a32 + a33, -b31 + b32, "AlphaTensor", cnt)
        m16, cnt = rec(-a13 + a22 + a23, b23 + b31 - b33, "AlphaTensor", cnt)
        m17, cnt = rec(a13 - a23, b23 - b33, "AlphaTensor", cnt)
        m18, cnt = rec(a22 + a23, -b31 + b33, "AlphaTensor", cnt)
        m19, cnt = rec(a12, b21, "AlphaTensor", cnt)
        m20, cnt = rec(a23, b32, "AlphaTensor", cnt)
        m21, cnt = rec(a21, b13, "AlphaTensor", cnt)
        m22, cnt = rec(a31, b12, "AlphaTensor", cnt)
        m23, cnt = rec(a33, b33, "AlphaTensor", cnt)

        c11 = m6 + m14 + m19
        c12 = m1 + m4 + m5 + m6 + m12 + m14 + m15
        c13 = m6 + m7 + m9 + m10 + m14 + m16 + m18
        c21 = m2 + m3 + m4 + m6 + m14 + m16 + m17
        c22 = m2 + m4 + m5 + m6 + m20
        c23 = m14 + m16 + m17 + m18 + m21
        c31 = m6 + m7 + m8 + m11 + m12 + m13 + m14
        c32 = m12 + m13 + m14 + m15 + m22
        c33 = m6 + m7 + m8 + m9 + m23

        cnt += 136 * (m // 3) ** 2

        top = np.hstack([c11, c12, c13])
        mid = np.hstack([c21, c22, c23])
        bot = np.hstack([c31, c32, c33])
        C = np.vstack([top, mid, bot])

        return C, cnt


def next_power_of(base, n):
    size = base
    while size < n:
        size *= base
    return size


def plot_time():
    ns = range(3, 65)
    times_binet, times_strass, times_alpha = [], [], []

    for n in ns:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        start = time.time()
        rec(A, B, "Binet")
        times_binet.append(time.time() - start)

        start = time.time()
        rec(A, B, "Strassen")
        times_strass.append(time.time() - start)

        s = next_power_of(3, n)
        A3 = np.zeros((s, s))
        B3 = np.zeros((s, s))
        A3[:n, :n] = A
        B3[:n, :n] = B
        start = time.time()
        rec(A3, B3, "AlphaTensor")
        times_alpha.append(time.time() - start)

    plt.plot(ns, times_binet, label="Binet")
    plt.plot(ns, times_strass, label="Strassen")
    plt.plot(ns, times_alpha, label="AlphaTensor")
    plt.xlabel("Rozmiar n")
    plt.ylabel("Czas [s]")
    plt.title("Porównanie czasu działania")
    plt.legend()
    plt.show()


def plot_count():
    ns = range(3, 65)
    counts_binet, counts_strass, counts_alpha = [], [], []

    for n in ns:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        _, c1 = rec(A, B, "Binet")

        s = next_power_of(2, n)
        A2 = np.zeros((s, s))
        B2 = np.zeros((s, s))
        A2[:n, :n] = A
        B2[:n, :n] = B
        _, c2 = rec(A, B, "Strassen")

        s = next_power_of(3, n)
        A3 = np.zeros((s, s))
        B3 = np.zeros((s, s))
        A3[:n, :n] = A
        B3[:n, :n] = B
        _, c3 = rec(A3, B3, "AlphaTensor")

        counts_binet.append(c1)
        counts_strass.append(c2)
        counts_alpha.append(c3)

    plt.plot(ns, counts_binet, label="Binet")
    plt.plot(ns, counts_strass, label="Strassen")
    plt.plot(ns, counts_alpha, label="AlphaTensor")
    plt.xlabel("Rozmiar n")
    plt.ylabel("Liczba operacji (count)")
    plt.title("Porównanie liczby operacji")
    plt.legend()
    plt.show()


def plot_memory():
    ns = range(3, 65)
    mem_binet, mem_strass, mem_alpha = [], [], []

    for n in ns:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        mem_binet.append((A.nbytes + B.nbytes) / 1024 ** 2)

        s = next_power_of(2, n)
        A2 = np.zeros((s, s))
        B2 = np.zeros((s, s))
        mem_strass.append((A.nbytes + B.nbytes) / 1024 ** 2)

        s = next_power_of(3, n)
        A3 = np.zeros((s, s))
        B3 = np.zeros((s, s))
        mem_alpha.append((A3.nbytes + B3.nbytes) / 1024 ** 2)

    plt.plot(ns, mem_binet, label="Binet")
    plt.plot(ns, mem_strass, label="Strassen")
    plt.plot(ns, mem_alpha, label="AlphaTensor")
    plt.xlabel("Rozmiar n")
    plt.ylabel("Zużycie pamięci [MB]")
    plt.title("Porównanie zużycia pamięci")
    plt.legend()
    plt.show()


def plot_all(max_n=163, step=1):
    ns = range(3, max_n, step)
    times_binet, times_strass, times_alpha = [], [], []
    counts_binet, counts_strass, counts_alpha = [], [], []
    mem_binet, mem_strass, mem_alpha = [], [], []

    for n in ns:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        start = time.time()
        _, c1 = rec(A, B, "Binet")
        times_binet.append(time.time() - start)
        counts_binet.append(c1)
        mem_binet.append((A.nbytes + B.nbytes) / 1024 ** 2)

        start = time.time()
        _, c2 = rec(A, B, "Strassen")
        times_strass.append(time.time() - start)
        counts_strass.append(c2)
        mem_strass.append((A.nbytes + B.nbytes) / 1024 ** 2)

        s = next_power_of(3, n)
        A3 = np.zeros((s, s))
        B3 = np.zeros((s, s))
        A3[:n, :n] = A
        B3[:n, :n] = B
        start = time.time()
        _, c3 = rec(A3, B3, "AlphaTensor")
        times_alpha.append(time.time() - start)
        counts_alpha.append(c3)
        mem_alpha.append((A3.nbytes + B3.nbytes) / 1024 ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ns, times_binet, label="Binet")
    axes[0].plot(ns, times_strass, label="Strassen")
    axes[0].plot(ns, times_alpha, label="AlphaTensor")
    axes[0].set_xlabel("Rozmiar n")
    axes[0].set_ylabel("Czas [s]")
    axes[0].set_title("Porównanie czasu działania")
    axes[0].legend()

    axes[1].plot(ns, counts_binet, label="Binet")
    axes[1].plot(ns, counts_strass, label="Strassen")
    axes[1].plot(ns, counts_alpha, label="AlphaTensor")
    axes[1].set_xlabel("Rozmiar n")
    axes[1].set_ylabel("Liczba operacji (count)")
    axes[1].set_title("Porównanie liczby operacji")
    axes[1].legend()

    axes[2].plot(ns, mem_binet, label="Binet")
    axes[2].plot(ns, mem_strass, label="Strassen")
    axes[2].plot(ns, mem_alpha, label="AlphaTensor")
    axes[2].set_xlabel("Rozmiar n")
    axes[2].set_ylabel("Zużycie pamięci [MB]")
    axes[2].set_title("Porównanie zużycia pamięci")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# plot_all(max_n=170,step=2)
#
# plot_time()
# plot_memory()
# plot_count()