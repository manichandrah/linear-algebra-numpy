# 🧮 Linear Algebra & NumPy — From Scratch

> Building deep learning foundations through pure NumPy implementations. Phase 1.1 of a systematic roadmap toward AI/ML engineering.

## 📋 Overview

This repository documents my journey through **Phase 1.1: Mathematics & NumPy Foundations** — a 78-day deep dive covering:

- **NumPy mastery** — arrays, broadcasting, axis operations, einsum
- **Linear algebra** — eigenvalues, SVD, PCA from scratch
- **Calculus & optimization** — gradient descent, Adam, backpropagation
- **Probability & statistics** — Bayes, CLT, entropy, KL divergence
- **Capstone** — Full neural network in pure NumPy achieving >95% on MNIST

## 📂 Repository Structure

```
├── docs/
│   └── Phase_1_1_Mathematics_and_NumPy.md   # Full 78-day execution plan
├── week01_numpy_foundations/
│   └── day01_array_creation/                 # Day 1: Arrays — Creation & Attributes
│       ├── ex01_linspace_2pi.py              # np.linspace for 2π range
│       ├── ex02_identity_matrix.py           # 5×5 identity, trace, diagonal
│       ├── ex03_float32_vs_float64.py        # Memory comparison & precision
│       ├── ex04_3d_zeros_array.py            # 3D array creation
│       ├── ex05_arange_reshape_diagonal.py   # Reshape + diagonal extraction
│       └── ex06_dtype_conversion.py          # Type casting & memory
└── README.md
```

## 🗓 Progress

| Week | Days | Topic | Status |
|------|------|-------|--------|
| 1 | 1–6 | NumPy Foundations | 🔄 In Progress |
| 2 | 7–12 | Linear Algebra Part 1 | ⏳ Upcoming |
| 3 | 13–18 | Linear Algebra Part 2 (Eigen, SVD, PCA) | ⏳ |
| 4 | 19–24 | Calculus & Optimization | ⏳ |
| 5 | 25–30 | Backpropagation & Neural Nets | ⏳ |
| 6 | 31–36 | Probability & Statistics | ⏳ |
| 7 | 37–42 | Classical ML from Scratch | ⏳ |
| 8–9 | 43–48 | NumForge Library | ⏳ |
| 10 | 49–54 | **Capstone: MNIST >95%** | ⏳ |
| 11–13 | 55–78 | Review, Einsum, Polish | ⏳ |

## 🔑 Key Takeaways (So Far)

### Day 1 — Array Creation & Attributes
- `np.linspace` vs `np.arange`: linspace controls **number of points**, arange controls **step size**
- `float32` uses **half the memory** of `float64` — critical for large-scale ML
- NumPy arrays are contiguous blocks of typed memory → orders of magnitude faster than Python lists
- `.nbytes` reveals actual memory usage: `(elements × bytes_per_element)`

## 🛠 Tech Stack

- **Python 3.x** + **NumPy** (no frameworks — everything from scratch)
- Matplotlib for visualizations
- Pure implementations — no scikit-learn, no PyTorch, no TensorFlow

## 📖 Full Roadmap

See [`docs/Phase_1_1_Mathematics_and_NumPy.md`](docs/Phase_1_1_Mathematics_and_NumPy.md) for the complete 78-day execution plan with daily exercises, starter code, and checklists.

---

*Part of a larger roadmap targeting AI/Fintech internships. Phase 1.1 runs Aug–Oct 2026.*
