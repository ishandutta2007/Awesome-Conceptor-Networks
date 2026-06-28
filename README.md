# Awesome-Conceptor-Networks
## 🧠 The Conceptor Networks Map

> **A comprehensive reference guide for Conceptor Networks—mapping their core mathematical types, adaptive variations, logical properties, and applications across RNNs and state-of-the-art Transformers.**

Conceptor Networks bridge the gap between subsymbolic neural activation trajectories (high-dimensional vectors) and symbolic conceptual logic. By defining conceptors as soft-projection regularizers on neural state matrices, this architecture allows a single recurrent or feedforward network to learn, store, de-noise, and morph a vast repertoire of distinct temporal or static patterns without catastrophic interference.

---

## 🧭 Mathematical Types & Variants

Conceptors vary in how they characterize and constrain high-dimensional neural activation spaces (X), balancing computational performance against parameter footprint.

| Variant / Type | Year | First Paper | Mathematical Definition / Trade-offs |
| :--- | :--- | :--- | :--- |
| **Matrix Conceptors (Traditional)**<br>The foundational variant. An N × N matrix C computed from the network state correlation matrix \(R = E[XX^T]\). | 2014 | [Controlling Recurrent Neural Networks by Conceptors](https://arxiv.org/abs/1403.3369) | **Mathematical Definition:** \(C = R(R + \alpha^{-2}I)^{-1}\), where α is the aperture (sensitivity scaling).<br>**Trade-off:** High performance and precise geometrical control, but storage costs scale quadratically (O(N²)) with the number of hidden layer neurons. |
| **Diagonal Conceptors**<br>Simplifies matrix conceptors by treating all off-diagonal values as zero, reducing the operator to a scaling vector of length N. | 2021 | [Controlling Recurrent Neural Networks by Diagonal Conceptors](https://arxiv.org/abs/2107.07968) | **Trade-off:** Drastically lowers storage costs to O(N), but compromises the ability to capture directional cross-correlations in spatial representations. |
| **Random Feature Conceptors (RFCs)**<br>An expansion of diagonal conceptors that projects the original network states into a higher-dimensional random feature space before applying a diagonal filter. | 2017 | [Using Conceptors to Manage Neural Long-Term Memories for Temporal Patterns](https://www.jmlr.org/papers/v18/16-130.html) | **Trade-off:** Recovers a major portion of Matrix Conceptor performance while keeping compute lightweight. |
| **PCA-Enhanced RFCs**<br>The latest architectural refinement utilizing Principal Component Analysis (PCA) to structure the projection before scaling. | 2025 | [Controlling Recurrent Neural Networks with Improved Feature Conceptors](https://fse.studenttheses.ub.rug.nl/id/eprint/34845) | **Trade-off:** Captures optimal variances across latent dimensions to maximize the expressive power of random projections. |

---

## ⚡ Framework Variations & Algorithms

Extensions built on the mathematical properties of conceptors to address specialized training constraints and network topologies.

| Framework / Algorithm | Year | First Paper | Key Concept & Impact |
| :--- | :--- | :--- | :--- |
| **Semi-Supervised Conceptors** | 2020 | [Semi-supervised Conceptors and Conceptor Logic](https://publishup.uni-potsdam.de/frontdoor/index/index/docId/48446) | Learns explicit concept bounds from partially labeled neural trajectories, isolating concepts even when background signals are highly chaotic. |
| **Conceptor-Aided Backpropagation (CAB)** | 2018 | [Overcoming Catastrophic Interference using Conceptor-Aided Backpropagation](https://openreview.net/forum?id=B13njo1R-) | A variation of standard deep learning optimization where gradients are shielded by conceptors during backpropagation.<br>**Impact:** Acts as a spatial gate that explicitly prevents new task updates from overwriting gradients belonging to previously mastered tasks, mitigating catastrophic forgetting. |
| **Conceptor Logic / Quasi-Boolean Logic** | 2014 | [Controlling Recurrent Neural Networks by Conceptors](https://arxiv.org/abs/1403.3369) | A structural property enabling standard Boolean operators (`NOT`, `AND`, `OR`) to be executed directly on conceptor matrices.<br>**Impact:** If C is a conceptor for "Walking" and D is "Running", a new conceptor can be generated mathematically via \(C \vee D\) without retraining the neural network. |

---

## 🚀 Real-World Applications by Domain

Conceptor networks have moved beyond Echo State Networks (ESNs) into contemporary, production-grade applications.

### 🤖 LLMs & Generative AI (Semantic Steering)
*   **Application:** Guiding the emotional tone, stylistic attributes, or safety alignment of large transformer models during text generation.
*   **Conceptor Mechanism:** Placed as forward-pass hooks inside a chosen Transformer layer. Activations are mapped to a conceptor trained on the union of a concept's poles (e.g., pooling both positive and negative sentiment examples into a unified correlation matrix) to steer semantic generation smoothly.
*   **Why:** Provides more nuanced, stable, and multi-directional control compared to static, unipolar steering vectors.

### 🔁 Reservoir Computing & Long-Term Memory (LTM)
*   **Application:** Managing and reproducing vast catalogs of complex temporal patterns (e.g., motor control traces, chaotic time series) within a single Echo State Network (ESN).
*   **Conceptor Mechanism:** The model acts similarly to a content-addressable Hopfield Network but for dynamical movements rather than static images. Conceptors filter noise and seamlessly morph between distinct target behaviors by shifting the aperture parameter (α).

### 🔄 Continual Learning (Lifelong Machine Learning)
*   **Application:** Training autonomous agents or sequential vision models on disjoint datasets over long operational lifespans.
*   **Conceptor Mechanism:** Employs CAB to dynamically track which structural directions of the weight space are critical to old behaviors, forcing new parameter optimization strictly into orthogonal subspaces.

---

## 🛠️ Implementation Architecture & Pseudocode Example

The fundamental implementation pattern for computing a conceptor matrix in PyTorch:

```python
import torch

def compute_matrix_conceptor(X: torch.Tensor, aperture: float) -> torch.Tensor:
    """
    Computes a standard Matrix Conceptor.
    Args:
        X: Tensor of neural activations, shape (num_samples, num_neurons)
        aperture: Alpha parameter governing sensitivity scaling
    Returns:
        C: The Conceptor matrix, shape (num_neurons, num_neurons)
    """
    num_samples = X.shape[0]
    num_neurons = X.shape[1]
    
    # 1. Compute the Correlation Matrix R
    R = torch.matmul(X.T, X) / num_samples
    
    # 2. Form the regularized identity matrix
    regularizer = (aperture ** -2) * torch.eye(num_neurons, device=X.device)
    
    # 3. Compute C = R * (R + alpha^-2 * I)^-1
    C = torch.matmul(R, torch.inverse(R + regularizer))
    return C
```
