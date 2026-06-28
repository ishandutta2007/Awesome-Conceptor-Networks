# 🧠 Matrix Conceptors (Traditional)

Traditional Matrix Conceptors are the foundational mathematical building block of Conceptor Networks, introduced by Herbert Jaeger in 2014. They act as soft projection operators that characterize the linear subspace spanned by a network's activation states.

---

## 📐 Mathematical Formulation

Given a matrix of neural activations $X \in \mathbb{R}^{M \times N}$ (where $M$ is the number of time steps or samples, and $N$ is the number of neurons), the state correlation matrix $R$ is defined as:

$$R = E[X X^T] \approx \frac{1}{M} X X^T$$

The matrix conceptor $C$ is computed by minimizing a regularized quadratic loss, yielding:

$$C = R (R + \alpha^{-2} I)^{-1}$$

where:
*   $\alpha \in (0, \infty)$ is the **aperture parameter**, governing the sensitivity scale of the filter.
*   $I$ is the $N \times N$ identity matrix.

---

## 📊 Computation & State Flow

```mermaid
graph TD
    A[Neural State Matrix X] --> B[Compute Correlation R = E[XXᵀ]]
    B --> C[Introduce Aperture Alpha]
    C --> D[Compute Regularized Inverse R + α⁻²I]
    D --> E[Matrix Conceptor C = R * R + α⁻²I ⁻¹]
    E --> F[Filter Neural States: X_filtered = C * X]
```

---

## ⚖️ Trade-Offs & Complexity
*   **Acoustic / Spatial Precision:** High. Captures full directional cross-correlations across all hidden dimensions.
*   **Space Complexity:** $\mathcal{O}(N^2)$ to store the conceptor matrix $C$.
*   **Time Complexity:** $\mathcal{O}(N^3)$ to compute the matrix inversion.
