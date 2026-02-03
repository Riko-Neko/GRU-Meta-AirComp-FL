import numpy as np

def optimize_beam_ris(H_BR, h_RUs):
    """
    Jointly optimize BS beamforming vector f and RIS phase vector theta for AirComp.
    H_BR: numpy array (N, M) channel from BS (M antennas) to RIS (N elements).
    h_RUs: numpy array (K, N) each row is RIS-to-user channel for a user.
    Returns:
      f_opt: optimized beamforming vector (M,) as complex numpy array.
      theta_opt: optimized RIS phase vector (N,) as complex numpy array (unit-modulus).
    """
    K, N = h_RUs.shape
    M = H_BR.shape[1]
    # Initialize f and theta
    f = np.ones(M, dtype=np.complex64)
    theta = np.ones(N, dtype=np.complex64)
    # Iterative alternating optimization
    iterations = 3
    for it in range(iterations):
        # Compute effective combined channel for each RIS element with current f
        # g = H_BR * f (N x 1 result)
        g = H_BR.dot(f)  # shape (N,)
        # Optimize theta: align phases for sum of all users
        # A_n = conj(g_n) * sum_{k}(h_RUs[k, n])
        conj_g = np.conj(g)
        h_sum = np.sum(h_RUs, axis=0)  # shape (N,)
        A = conj_g * h_sum
        # Set theta_n = exp(-j * arg(A_n)) to align phases
        theta = np.exp(-1j * np.angle(A))
        # Optimize f: maximize combined signal power
        # Use principal eigenvector of sum_k (H_BR * diag(theta) * h_k)^*(H_BR * diag(theta) * h_k)
        # Compute combined channel vectors a_k = H_BR^H * diag(theta) * h_k (dimension M)
        S = np.zeros((M, M), dtype=np.complex128)
        Theta = np.diag(theta)
        for k in range(K):
            # a_k = H_BR^H * (theta * h_k) where theta * h_k is elementwise
            a_k = H_BR.conj().T.dot(theta * h_RUs[k])
            # accumulate outer product
            S += np.outer(a_k, a_k.conj())
        # Get principal eigenvector of S
        try:
            eigvals, eigvecs = np.linalg.eigh(S)
            f = eigvecs[:, np.argmax(eigvals)]
        except np.linalg.LinAlgError:
            # Fallback: just use the first column of H_BR as beam if eigen decomposition fails
            f = H_BR[0, :].conj()
        # Normalize f (unit norm)
        if np.linalg.norm(f) > 0:
            f = f / np.linalg.norm(f)
    return f.astype(np.complex64), theta.astype(np.complex64)