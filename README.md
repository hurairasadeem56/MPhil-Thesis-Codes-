# Nonlinear Biological Systems' Analysis Using Physics Informed Neural Networks: A Study of Growth and Expression Dynamics

Source code for the M.Phil thesis by **Hurrira Sadeem** (Institute of Numerical Sciences, Kohat University of Science & Technology, Kohat, 2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hurairasadeem56/MPhil-Thesis-Codes-/blob/main/Huraira_Codes.ipynb)
![Python](https://img.shields.io/badge/python-3-blue)
![DeepXDE](https://img.shields.io/badge/DeepXDE-1.15.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains the implementation of a continuous, meshless **Physics-Informed Neural Network (PINN)** framework for forward simulation of two nonlinear biological models:

| Model | Equation(s) | Purpose |
|---|---|---|
| **Gompertz tumor growth** | dN/dt = αN ln(K/N), N(0) = N₀ | Tumor volume growth toward a carrying capacity |
| **Coupled mRNA–protein gene expression** | dm/dt = α_m − β_m m, dp/dt = α_p m − β_p p | Transcription and translation dynamics |

The governing equations are built into the loss function, and time derivatives are computed by automatic differentiation in [DeepXDE](https://github.com/lululxvi/deepxde) with a TensorFlow backend. For each model, four experiments study how PINN accuracy depends on:

1. **Optimizer**: Adam vs Adam + L-BFGS
2. **Hyperparameters**: number of hidden layers and neurons per layer
3. **Observation noise**: 0%, 5% and 10% Gaussian noise
4. **Collocation sampling**: Latin Hypercube Sampling (LHS) vs uniform sampling

## Thesis

- **Title:** Nonlinear Biological Systems' Analysis Using Physics Informed Neural Networks: A Study of Growth and Expression Dynamics
- **Degree:** Master of Philosophy (M.Phil), Mathematics
- **Institution:** Institute of Numerical Sciences, Kohat University of Science & Technology (KUST), Kohat
- **Supervisor I:** Dr. Wali Khan Mashwani, Institute of Numerical Sciences, KUST, Kohat
- **Supervisor II:** Dr. Tahir Mehmood, School of Natural Sciences, National University of Sciences and Technology (NUST), Islamabad

The mathematical formulation is in Chapter 2 and Appendix A of the thesis, and the experimental results are in Chapter 3.

## Models and Parameters

### Gompertz tumor growth

| Parameter | Description | Value |
|---|---|---|
| α | Growth rate | 0.35 |
| K | Carrying capacity | 1.0 |
| N₀ | Initial tumor volume | 0.05 |
| t ∈ [t₀, t_f] | Time domain | [0, 10] |

Analytical solution used for evaluation: N(t) = K · exp( ln(N₀/K) · e^(−αt) ).

The network output is passed through N = K · sigmoid(z), which keeps 0 < N < K so the logarithm ln(K/N) is always defined.

### Gene expression (mRNA m(t), protein p(t))

| Parameter | Description | Value |
|---|---|---|
| α_m | mRNA production rate | 1.0 |
| β_m | mRNA degradation rate | 0.5 |
| α_p | Protein production rate | 2.0 |
| β_p | Protein degradation rate | 0.4 |
| m₀, p₀ | Initial concentrations | 0.0, 0.0 |
| t ∈ [t₀, t_f] | Time domain | [0, 10] |

Closed-form solutions for m(t) and p(t) are used as references (thesis Appendix A.2.1). The ODEs are also integrated with SciPy `solve_ivp` (RK45, 500 time points) as a numerical reference.

## PINN Setup

| Setting | Value |
|---|---|
| Framework | DeepXDE 1.15.0, TensorFlow 2.20.0 backend |
| Network | Fully connected feedforward, `tanh` activation, Glorot uniform initialization |
| Default architecture | 1 → 64 → 64 → 64 → 64 → 1 (Gompertz); 1 → 64 → 64 → 64 → 64 → 2 (gene expression) |
| Collocation points | 2,000 interior, 50 boundary, 1,000 test |
| Default sampling | Latin Hypercube Sampling |
| Training schedule | Adam 1e-3 (10,000 it) → Adam 5e-4 (10,000 it) → Adam 1e-4 (20,000 it) → L-BFGS (optional, Adam + L-BFGS runs) |
| Stopping criteria | `EarlyStopping` (min_delta = 1e-7, patience = 5000); L-BFGS runs until its convergence tolerance |
| Checkpointing | `ModelCheckpoint`, best model saved every 1000 iterations |
| Loss weights | Gompertz: [100, 1] (residual, initial condition); gene expression: [100, 100, 1, 1] (two residuals, two initial conditions); Gompertz noise experiment: [100, 1, 10] (adds data term) |
| Random seed | 1234 |
| Evaluation | MSE, RMSE, MAE, relative L2 error, maximum error, R², training time |

**Noise experiments.** Observations are generated from the analytical solution and perturbed with Gaussian noise with standard deviation equal to *p*% of the standard deviation of the exact signal (p = 0, 5, 10). Gompertz uses 50 observation points and gene expression uses 200, equally spaced on [0, 10].

## Repository Contents

```
.
├── Huraira_Codes.ipynb   # Full implementation (both models, all experiments)
├── requirements.txt      # Pinned dependencies
├── README.md
├── LICENSE
└── results/              # Output CSVs and figures (generated when you run the notebook)
```

The notebook is organized into two parts. Each part contains the four experiments.

| Notebook section | Model | Experiment |
|---|---|---|
| Model 1 – Experiment 1 | Gompertz | Optimizer comparison (Adam vs Adam + L-BFGS) |
| Model 1 – Experiment 2 | Gompertz | Hyperparameter sensitivity (3 or 4 layers × 32, 64, 128 neurons) |
| Model 1 – Experiment 3 | Gompertz | Noise robustness (0%, 5%, 10%) |
| Model 1 – Experiment 4 | Gompertz | Sampling strategy (LHS vs uniform) |
| Model 2 – Experiment 1 | Gene expression | Optimizer comparison |
| Model 2 – Experiment 2 | Gene expression | Hyperparameter sensitivity |
| Model 2 – Experiment 3 | Gene expression | Noise robustness |
| Model 2 – Experiment 4 | Gene expression | Sampling strategy |

Each experiment writes its CSV summaries and figures (comparison, absolute error, relative error, correlation, residual) and saves the best model checkpoint.

## Installation

The code was run on Google Colab with a GPU. To run it locally:

```bash
git clone https://github.com/hurairasadeem56/MPhil-Thesis-Codes-.git
cd MPhil-Thesis-Codes-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Huraira_Codes.ipynb
```

`requirements.txt`:

```
deepxde==1.15.0
tensorflow==2.20.0
numpy
scipy
matplotlib
pandas
scikit-learn
```

A GPU is optional. CPU runs work but are slower.

## How to Reproduce the Experiments

1. Open `Huraira_Codes.ipynb` in Colab (badge above) or Jupyter.
2. Run the installation and import cells first. They set the random seed to 1234 and set the DeepXDE backend to TensorFlow.
3. Run the problem-definition cell of the experiment you want, then its training cell, then its evaluation and plotting cells. Training and evaluation cells share variables, so run them in order.
4. Some training cells are switched between settings by commenting and uncommenting lines. Before running, set the following values for the case you want:

| Experiment | Setting to select |
|---|---|
| Optimizer comparison | `optimizer_mode` = `"adam"` or `"adam_lbfgs"` (Gompertz); skip or run the L-BFGS stage (gene expression) |
| Hyperparameter sensitivity | `hidden_layers` ∈ {3, 4}, `neurons` ∈ {32, 64, 128}, and the matching `MODEL_DIR` |
| Noise robustness | `NOISE_LEVEL` ∈ {0, 5, 10} |
| Sampling strategy | `"LHS"` or `"uniform"`, and the matching `RESULT_DIR` |

5. Outputs are written to the results folders (CSV tables, PNG figures at 300 dpi, saved models).

Because training uses stochastic initialization and GPU arithmetic, exact numbers can vary slightly between runs and hardware.

## Main Results

All values are from the thesis (Chapter 3).

### Gompertz tumor growth

| Experiment | Setting | MSE | Relative L2 | R² | Time (s) |
|---|---|---|---|---|---|
| Optimizer | Adam | 1.37 × 10⁻⁷ | 6.05 × 10⁻⁴ | 0.999998 | 31.82 |
| Optimizer | **Adam + L-BFGS** | **6.80 × 10⁻¹¹** | **1.35 × 10⁻⁵** | ≈ 1 | 48.19 |
| Architecture | **4 layers × 64 neurons (best)** | **2.29 × 10⁻¹¹** | **7.83 × 10⁻⁶** | ≈ 1 | 41.04 |
| Noise | 0% | 8.23 × 10⁻¹² | 4.70 × 10⁻⁶ | 1.000000 | 54.28 |
| Noise | 5% | 2.36 × 10⁻⁶ | 2.52 × 10⁻³ | 0.999970 | 54.32 |
| Noise | 10% | 2.21 × 10⁻⁵ | 7.70 × 10⁻³ | 0.999717 | 69.20 |
| Sampling | LHS | 2.89 × 10⁻¹¹ | 8.81 × 10⁻⁶ | ≈ 1 | 37.85 |
| Sampling | Uniform | 2.69 × 10⁻¹¹ | 8.49 × 10⁻⁶ | ≈ 1 | 35.97 |

### Gene expression

| Experiment | Setting | MSE | Relative L2 | R² | Time (s) |
|---|---|---|---|---|---|
| Optimizer | Adam | 2.82 × 10⁻⁶ | 2.30 × 10⁻⁴ | 0.999999650 | 64.11 |
| Optimizer | **Adam + L-BFGS** | **1.27 × 10⁻⁸** | **2.93 × 10⁻⁵** | 0.999999991 | 70.78 |
| Architecture | **3 layers × 128 neurons (best MSE)** | **5.04 × 10⁻⁹** | **2.09 × 10⁻⁵** | 0.999999995 | 68.94 |
| Noise | 0% | 2.87 × 10⁻⁹ | 1.60 × 10⁻⁵ | 0.999999997 | 81.26 |
| Noise | 5% | 1.88 × 10⁻⁴ | 2.68 × 10⁻³ | 0.999947 | 78.77 |
| Noise | 10% | 7.52 × 10⁻⁴ | 5.36 × 10⁻³ | 0.999788 | 82.41 |
| Sampling | LHS | 1.05 × 10⁻⁸ | 3.12 × 10⁻⁵ | 0.999999989 | 68.95 |
| Sampling | Uniform | 8.49 × 10⁻⁹ | 2.86 × 10⁻⁵ | 0.999999990 | 79.73 |

### Key findings

- Adam + L-BFGS reduces the error substantially compared with Adam alone for both models, at a modest increase in training time.
- Gompertz: the 4 × 64 network performed best. Gene expression: the 3 × 128 network performed best.
- Accuracy degrades gradually with observation noise, and R² stays above 0.9997 even at 10% noise.
- LHS and uniform sampling both give highly accurate solutions, with uniform sampling slightly more accurate in these low-dimensional ODE problems.

## Limitations

- Both models are low-dimensional ODEs with known analytical solutions, so the results show the behavior of the framework in a controlled setting.
- Noise is synthetic Gaussian noise, not experimental biological data.
- Results are for the forward problem only (no parameter inference).

See Section 4.2 and Section 4.3 of the thesis for the full discussion of limitations and future work.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{sadeem2026pinn,
  author  = {Sadeem, Hurrira},
  title   = {Nonlinear Biological Systems' Analysis Using Physics Informed Neural Networks: A Study of Growth and Expression Dynamics},
  school  = {Kohat University of Science and Technology},
  address = {Kohat, Khyber Pakhtunkhwa, Pakistan},
  year    = {2026},
  type    = {M.Phil thesis}
}
```

## Acknowledgements

Supervised by Dr. Wali Khan Mashwani (KUST, Kohat) and Dr. Tahir Mehmood (NUST, Islamabad). Built with [DeepXDE](https://github.com/lululxvi/deepxde), TensorFlow, SciPy, NumPy, pandas, scikit-learn and Matplotlib.

## License

Released under the MIT License. See `LICENSE` for details.

## Contact

Hurrira Sadeem — GitHub: [@hurairasadeem56](https://github.com/hurairasadeem56)
