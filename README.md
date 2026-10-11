#  QForest — Quantum-Enhanced Deforestation Detection

**A hybrid quantum-classical pipeline for satellite change detection, executed end-to-end on real IBM quantum hardware.**

QForest investigates a central question: **Can quantum feature maps improve deforestation detection when labeled satellite data is scarce?**

Built for the **Qollab Hackathon (October 9–11, 2026)**, the project combines satellite imagery, deep learning, quantum feature extraction, and classical machine learning. Our controlled experiments suggest that quantum features are most valuable in the regime where classical pipelines are most fragile: **when training labels are limited.**

---

## 📌 TL;DR

| Experiment | Key Result |
|---|---|
| Fully supervised encoder | Quantum: **0.966** vs. classical: 0.961 balanced accuracy |
| Real quantum hardware | IBM Fez: **0.965** vs. matched simulation: 0.970 |
| Label-starved encoders | Quantum advantage: **+0.089 ± 0.025** over classical |
| Linear-correlation control | Quantum advantage: **+0.107 ± 0.023** |
| Label efficiency | Quantum at 50 labels: **0.796**, vs. classical at 300 labels: 0.719 |

*Metrics are balanced accuracy. Results depend on the experimental protocol and the single study area evaluated.*

## 🎯 The Problem

Deforestation monitoring is data-rich but label-poor. Satellite imagery is widely available, but creating reliable, expert-annotated deforestation masks—such as those produced by Brazil's PRODES monitoring program—is expensive and time-consuming.

Classical machine-learning pipelines can struggle when trained with only a small number of labeled examples.

**Our research question:** Can a fixed quantum feature map extract useful nonlinear structure from satellite embeddings and improve detection when labeled data is scarce?

## 🧠 Methodology

QForest implements a simplified, DQFE-inspired quantum feature extraction pipeline for environmental change detection.

### Pipeline Architecture

```text
Sentinel-2 Satellite Tiles (128 × 128)
                 │
                 ▼
       Frozen ResNet-50 Encoder
                 │
                 ▼
      Trainable 15-D Bottleneck
                 │
                 ▼
        Embedding z ∈ R^15
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  Classical Features   Quantum Feature Map
                            │
                   15-Qubit Circuit
                            │
                    Hadamard Layer
                            │
                 RZZ(γ · mᵢⱼ) on 24 Edges
                            │
                    RY(γ · π · zᵢ)
                            │
                    Measurement (Z)
                            │
                            ▼
                39 Quantum Observables
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Classical RF                Quantum RF
              └─────────────┬─────────────┘
                            ▼
                 Deforestation Detection
```

### Quantum Feature Extraction

The quantum feature map uses 15 qubits and a coupling graph derived from pairwise mutual information (MI).

1. **Graph construction:** Select the top 24 MI edges using training data only.
2. **Quantum initialization:** Prepare the 15-qubit system with Hadamard gates.
3. **Entangling layer:** Apply RZZ rotations parameterized by edge weights.
4. **Sample encoding:** Apply RY rotations parameterized by each tile's embedding.
5. **Measurement:** Estimate expectation values using 1,024 shots.

The resulting 39-dimensional quantum feature vector contains:

- 15 single-qubit observables: ⟨Zᵢ⟩
- 24 pairwise observables: ⟨ZᵢZⱼ⟩

The rotation parameter is γ = 0.8, selected from {0.3, 0.5, 0.8, 1.2}. Performance was relatively stable across γ = 0.5–0.8.

**Experimental controls:** The same classifier protocol is used across feature sets. The coupling graph is fitted using training patches only, and the held-out test patches are excluded from feature-graph construction.

## 🛰️ Dataset

**Source:** [Deforestation Detection Dataset — Kaggle](https://www.kaggle.com/)

The dataset, attributed to Chhibber et al., combines satellite imagery and PRODES deforestation masks.

| Property | Description |
|---|---|
| Satellite imagery | Sentinel-2, four bands |
| Tile dimensions | 128 × 128 pixels |
| Total tiles | 7,027 |
| Deforested tiles | 4,465 |
| Forest tiles | 2,562 |
| Deforested threshold | At least 10% mask coverage |
| Forest threshold | At most 1% mask coverage |
| Held-out patches | 1, 4, 11, and 14 |
| Held-out test tiles | 1,767 |

Tiles between the two mask thresholds are excluded from the binary classification task.

### Spatial Evaluation

Evaluation uses spatially disjoint patches rather than a random tile-level split. GroupKFold is used for patch-aware evaluation where applicable.

This reduces the risk of spatial leakage between training and testing data. However, the experiments cover only one study area, so generalization to other regions remains untested.

## 📊 Results

### Study 1 — Fully Trained Encoder

Balanced accuracy on held-out patches, averaged over 20 seeds.

| Feature Set | Dimensions | 20 Labels | 50 Labels | 100 Labels | 300 Labels | Full |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 15 | 0.958 | 0.959 | 0.962 | 0.963 | 0.961 |
| Quantum | 39 | 0.958 | 0.960 | 0.963 | **0.965** | **0.966** |
| Hybrid | 54 | 0.958 | 0.960 | 0.962 | 0.964 | 0.963 |

The quantum feature map matches or slightly exceeds the classical baseline across the evaluated label budgets.

Combining classical and quantum features does not improve performance further, suggesting that the quantum map may reorganize useful information already present in the classical representation rather than simply adding complementary information.

The reported separability analysis supports this interpretation: per-feature Fisher separability changes from 14.1 to 3.2, while LDA-axis AUC remains nearly unchanged at 0.992 versus 0.989.

### ⚛️ Real Quantum Hardware — IBM Fez

We executed 500 circuits on IBM Fez, a Heron r2 processor with 156 qubits, using 1,024 shots per circuit.

| Configuration | Balanced Accuracy |
|---|---:|
| Noiseless Aer simulation | 0.970 |
| Real IBM Fez hardware | 0.965 |
| Absolute difference | 0.005 |

**Key finding:** The observed hardware result is within 0.5 percentage points of the matched noiseless simulation.

The evaluated circuits had a reported depth of 186 and 89 two-qubit gates.

This result suggests that the learned class structure in the quantum observables can survive device noise reasonably well under the evaluated configuration. It does not establish that quantum hardware outperforms classical computation in speed or cost.

### Study 2 — Label-Starved Encoders

In this experiment, the 15-dimensional encoder itself is retrained using only a limited number of labeled examples.

Each replicate varies the sample selection and initialization seed. All feature arms share the same embedding within each replicate.

Three approaches are compared:

- **Classical:** The original 15-dimensional bottleneck.
- **Linear control:** A linear mapping to [-1, 1] plus pairwise products on the same 24 MI edges, giving 39 dimensions without quantum computation.
- **Quantum:** The 39-dimensional quantum feature map.

#### Balanced Accuracy

| Labels | Classical | Linear Control | Quantum | Δ (Q − C) | Δ (Q − Control) |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.571 ± 0.205 | 0.567 ± 0.181 | **0.743 ± 0.149** | +0.171 | +0.176 |
| 50 | 0.679 ± 0.095 | 0.649 ± 0.069 | **0.796 ± 0.048** | +0.117 | +0.147 |
| 100 | 0.704 ± 0.126 | 0.689 ± 0.119 | **0.760 ± 0.077** | +0.057 | +0.071 |
| 300 | 0.719 ± 0.060 | 0.694 ± 0.056 | **0.729 ± 0.081** | +0.010 | +0.035 |

*Values are mean ± standard deviation across eight replicates per label budget.*

#### Pooled Results

Across 32 paired replicates:

- Quantum versus classical: **+0.089 ± 0.025 standard error**
- Quantum versus linear control: **+0.107 ± 0.023 standard error**
- Quantum wins in 25 of 32 replicates.
- Reported sign-test p-value: approximately 0.002.

The linear-correlation control performs 0.018 ± 0.007 below the classical baseline, suggesting that naïvely expanding pairwise correlations may overfit when labels are scarce.

The quantum map's bounded nonlinear feature representation is consistent with a regularization effect, although further controls are needed to establish the precise mechanism.

The advantage decreases as the number of training labels increases, matching our hypothesis that quantum feature extraction may be particularly useful in low-data regimes.

### Label Efficiency

At 50 labels, the quantum pipeline achieves a balanced accuracy of **0.796**, compared with **0.719** for the classical pipeline trained using 300 labels.

This is an encouraging result for label-constrained remote sensing, where acquiring reliable annotations can be expensive.

## 🔬 Additional Analysis

### Rescue Cases

Several low-performing classical encoder replicates improve substantially with quantum features under the same labels, embeddings, and classifier protocol:

```text
0.197 → 0.787
0.455 → 0.763
0.509 → 0.764
0.548 → 0.843
```

These examples illustrate potential recovery in difficult low-label settings. They are selected examples rather than an estimate of average performance.

A failure case is also retained in the results: replicate 5 at m = 20, where quantum balanced accuracy is 0.454.

## ⚠️ Honest Limitations

We aim to report both promising results and the limitations of this study.

- **Single study area:** Cross-region generalization has not been evaluated.
- **Simplified quantum circuit:** The implementation uses Hadamard, RZZ, and RY gates rather than the full counterdiabatic DQFE method.
- **Limited circuit size:** The feature extractor uses 15 qubits, compared with up to 156 qubits in the referenced work.
- **Binary classification:** The task distinguishes forest from deforested tiles.
- **Statistical uncertainty:** Replicates share held-out patches, potentially making standard errors anticonservative. A patch-aware permutation test would provide a stronger inferential analysis.
- **Limited hardware evaluation:** One hardware run was performed, without error mitigation.
- **No demonstrated quantum speedup:** The experiments assess predictive performance, not computational advantage over optimized classical methods.
- **Scope of the claim:** The results support a promising low-label hypothesis, not universal quantum superiority.

## 🚀 Reproduce the Experiments

### Requirements

- Python 3.11 or another compatible Python environment
- Qiskit and Qiskit Aer
- scikit-learn
- NumPy
- PyTorch
- Streamlit
- Jupyter Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/prachithanki1-lgtm/Quantum-Satellite-Image-Change-Detection-Pipeline-v2
cd Quantum-Satellite-Image-Change-Detection-Pipeline-v2
```


### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, you can invoke the environment's Python executable directly instead.

### 3. Install Dependencies

```powershell
python -m pip install qiskit qiskit-aer scikit-learn numpy torch streamlit jupyter matplotlib pillow
```

Install any additional dependencies required by your notebook.

### 4. Run the Notebook

```powershell
python -m jupyter notebook
```

Open `Qollabhack.ipynb` and execute its cells in order. The experiments save intermediate artifacts as specified by the notebook.

### 5. Launch the Streamlit Demo

```powershell
python -m streamlit run app.py
```

The demo uses precomputed features and does not make live quantum hardware calls.

### 6. Real Quantum Hardware

Reproducing the IBM Fez experiment requires suitable IBM Quantum Runtime access, compatible account configuration, and the circuit execution code. Hardware availability and backend access may change.

The demo can be run independently without live hardware access.

## 📁 Key Artifacts

| Artifact | Description |
|---|---|
| `Qollabhack.ipynb` | Main experiment notebook |
| `app.py` | Streamlit demonstration application |
| `quantum_n15.npz` | Precomputed quantum features |
| `real_quantum.npz` | Real-hardware experiment outputs |
| `hardware_meta.json` | Hardware metadata and job IDs |
| `encoder_fewshot_ext.json` | Study 2 results across 32 replicates |
| `results/` | Generated figures and visualizations |


## 📚 References and Credits

**Base method**

Zhang, Simen, Flores-Garrigós et al., *Quantum-enhanced satellite image classification*. arXiv:2602.18350.

- [Research paper on arXiv](https://arxiv.org/abs/2602.18350)

**Dataset**

Chhibber et al., *Deforestation Detection Dataset*, Kaggle. Please consult the dataset page for the exact citation and applicable license terms before redistribution.

**Quantum hardware**

IBM Quantum Runtime — IBM Fez backend.

**Team**

- Abhishek Kumar
- Prachi Thanki

**Event:** Qollab Hackathon, October 9–11, 2026.

---

##  Why QForest?

QForest explores how quantum computing might contribute to a practical environmental challenge: detecting deforestation when high-quality labeled data is limited.

Rather than claiming that quantum computing replaces classical machine learning, we investigate a narrower and testable hypothesis: **quantum feature maps may offer their greatest predictive value when training data is scarce.**

Our next steps are to validate the result across multiple geographical regions, strengthen statistical testing, and compare against additional classical feature maps.

**Quantum computing for a greener planet.** 
 ye final readme hai
