import numpy as np
from PIL import Image
import timeit

# Define finite-difference gradient operator
def gradient(u):
    H, W = u.shape
    grad = np.zeros((H, W, 2))
    grad[:, :, 0] = np.roll(u, -1, 1) - u   # difference in x-direction
    grad[:, :, 1] = np.roll(u, -1, 0) - u   # difference in y-direction
    return grad

# Divergence operator (the adjoint of the gradient)
def divergence(p):
    # p has shape (H, W, 2)
    div = -((p[:, :, 0] - np.roll(p[:, :, 0], 1, 1)) + 
            (p[:, :, 1] - np.roll(p[:, :, 1], 1, 0)))
    return div

# Symmetric gradient operator for a vector field w (w has two components)
def sym_gradient(w):
    H, W, _ = w.shape
    sym_grad = np.zeros((H, W, 3))
    # First diagonal: ∂x w_0
    sym_grad[:, :, 0] = np.roll(w[:, :, 0], -1, 1) - w[:, :, 0]
    # Second diagonal: ∂y w_1
    sym_grad[:, :, 1] = np.roll(w[:, :, 1], -1, 0) - w[:, :, 1]
    # Off-diagonals: 0.5*(∂y w_0 + ∂x w_1)
    tmp1 = np.roll(w[:, :, 0], -1, 0) - w[:, :, 0]
    tmp2 = np.roll(w[:, :, 1], -1, 1) - w[:, :, 1]
    sym_grad[:, :, 2] = 0.5 * (tmp1 + tmp2)
    return sym_grad

# Adjoint of the symmetric gradient
def sym_divergence(q):
    H, W, _ = q.shape
    sym_div = np.zeros((H, W, 2))
    # For the first component:
    sym_div[:, :, 0] = -(q[:, :, 0] - np.roll(q[:, :, 0], 1, 1)) \
        - 0.5*(q[:, :, 2] - np.roll(q[:, :, 2], 1, 0))
    # For the second component:
    sym_div[:, :, 1] = -(q[:, :, 1] - np.roll(q[:, :, 1], 1, 0)) \
        - 0.5*(q[:, :, 2] - np.roll(q[:, :, 2], 1, 1))
    return sym_div

# Projection onto the l2-ball for dual variable p (pointwise)
def proj_p(p, alpha1):
    norm = np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2)
    factor = np.maximum(1, norm / alpha1)
    p[:, :, 0] /= factor
    p[:, :, 1] /= factor
    return p

# Projection for the dual variable q
# Note: Here the "norm" is defined as sqrt(q_0^2 + q_1^2 + 2*q_2^2) according to the structure of the symmetric gradient.
def proj_q(q, alpha0):
    norm = np.sqrt(q[:, :, 0]**2 + q[:, :, 1]**2 + 2 * q[:, :, 2]**2)
    factor = np.maximum(1, norm / alpha0)
    q[:, :, 0] /= factor
    q[:, :, 1] /= factor
    q[:, :, 2] /= factor
    return q

# Main TGV denoising function
def tgv_denoise(u0, lam=1.0, alpha0=2.0, alpha1=1.0, tau=0.125, sigma=0.125, n_iter=300):
    """
    Total Generalized Variation (TGV) denoising.

    Args:
        u0 (np.ndarray): Input image (grayscale).
        lam (float, optional): Scaling parameter. Defaults to 1.0.
        alpha0 (float, optional): Regularization parameter for the second-order term. Defaults to 2.0.
        alpha1 (float, optional): Regularization parameter for the first-order term. Defaults to 1.0.
        tau (float, optional): Time step for the primal variable. Defaults to 0.125.
        sigma (float, optional): Time step for the dual variable. Defaults to 0.125.
        n_iter (int, optional): Number of iterations. Defaults to 300.

    Returns:
        np.ndarray: Denoised image.
    """
    H, W = u0.shape
    # Initialize variables
    u = u0.copy()
    w = np.zeros((H, W, 2))
    p = np.zeros((H, W, 2))  # dual variable for (grad u - w)
    q = np.zeros((H, W, 3))  # dual variable for sym_gradient(w)

    u_bar = u.copy()
    w_bar = w.copy()

    for i in range(n_iter):
        # ===== Dual updates =====
        # Update p: p = proj_{|p|<=alpha1}( p + sigma*(grad(u_bar) - w_bar) )
        grad_u_bar = gradient(u_bar)
        p += sigma * (grad_u_bar - w_bar)
        p = proj_p(p, alpha1 * lam)

        # Update q: q = proj_{|q|<=alpha0}( q + sigma*(sym_gradient(w_bar)) )
        q += sigma * sym_gradient(w_bar)
        q = proj_q(q, alpha0 * lam)

        # ===== Primal updates =====
        # Save old u and w for extrapolation
        u_old = u.copy()
        w_old = w.copy()

        # Update u:
        u -= tau * divergence(p)
        # Data fidelity proximal update (L2)
        u = (u + tau * u0) / (1 + tau)

        # Update w:
        # The term - p comes from derivative of (grad u - w),
        # and symdiv(q) is the adjoint of sym_gradient(w).
        w -= tau * (-p + sym_divergence(q))

        # ===== Extrapolation =====
        u_bar = 2 * u - u_old
        w_bar = 2 * w - w_old

        # # Optional: print the energy every 50 iterations
        # if (i+1) % 50 == 0:
        #     # Compute the residuals (this is just indicative)
        #     primal_res = np.linalg.norm(u - u_old)
        #     # print(f"Iteration {i+1:03d}, primal change = {primal_res:.4e}")
        #     print("Iteration %03d, primal change = %0.4e" % (i+1, primal_res))

    return u


from joblib import Parallel, delayed
import os
def parallel_tgv_denoise(u0, lam=1.0, alpha0=2.0, alpha1=1.0, tau=0.125, sigma=0.125, n_iter=300):
    patch_size = 32
    
    H, W = u0.shape
    num_patches_x = H // patch_size
    num_patches_y = W // patch_size

    # Create a vector to store the denoised patches
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patches.append(u0[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])

    # Create a vector to store the denoised patches
    denoised_patches = Parallel(n_jobs=os.cpu_count())(delayed(tgv_denoise)(patch, lam, alpha0, alpha1, tau, sigma, n_iter) for patch in patches)

    # Create a new image to store the denoised patches
    denoised_img = np.zeros((H, W))
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            denoised_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = denoised_patches[i*num_patches_y + j]

    return denoised_img
    

if __name__ == "__main__":
    img = np.array(Image.open("astronaut.png"))
    print(img.shape)
    print(img.min(), img.max())
    img = img.mean(axis=2).astype(np.float32)
    img += np.random.normal(0, 10., img.shape)
    noisy_img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(noisy_img, mode='L').save("noisy_img_py.png")
    print(noisy_img.min(), noisy_img.max())
    time = timeit.timeit(lambda: parallel_tgv_denoise(noisy_img.astype(np.float32), 10., 2.0, 1.0, 0.125, 0.125, 300), number=5)
    avg_time = time / 5
    print(f"Time taken: {avg_time} seconds on 5 runs of 300 iterations on average")
    denoised_img = parallel_tgv_denoise(noisy_img.astype(np.float32), 10., 2.0, 1.0, 0.125, 0.125, 300)
    print(denoised_img.min(), denoised_img.max())
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
    Image.fromarray(denoised_img, mode='L').save("denoised_img_py.png")