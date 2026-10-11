# Quantum-Enhanced Deforestation Detection

A hybrid classical-quantum pipeline that classifies satellite tiles as **deforested** or **forest**, built for the Qollab Quantum Hackathon.

> Note: the dataset is a single date (Sentinel-2, July 2020), so this is **deforestation detection**, not true before/after change detection.

## Pipeline

```
Sentinel-2 tile (128x128, RGB)
  -> ResNet50 (frozen, ImageNet) -> 2048 features
  -> dense bottleneck layer      -> 15 features
  -> quantum circuit (15 qubits) -> 39 features (15 <Z_i> + 24 <Z_i Z_j>)
  -> Random Forest               -> deforested / forest
```

**Quantum step:** each of the 15 features is encoded as an `RY` rotation on its own qubit. Fixed `RZZ` couplings link the 24 most related feature pairs (chosen by mutual information, using training data only). Each circuit is run for 1024 shots and the Pauli-Z expectation values are used as features. The design is inspired by the Digitized Quantum Feature Extraction approach of Zhang et al., "Quantum-enhanced satellite image classification" (Kipu Quantum / IBM, 2026), but it is a simplified version, not a reproduction.

## Data

- Dataset: *Amazon Rainforest Deforestation Dataset* by Akhil Chhibber (CC BY), Sentinel-2 cloud-free patches with PRODES-based masks.
- 16 large patches were cut into 128x128 tiles (about 7,000 labeled tiles). A tile counts as deforested if at least 10% of its pixels are deforested, and as forest if at most 1% are. Tiles in between are dropped.
- Train/test split is **by whole patch** (test patches: 1, 4, 11, 14) to avoid leakage between neighboring tiles.

## Results (held-out patches, balanced accuracy)

| Experiment | Result |
|---|---|
| Classical baseline (ResNet50 + dense 15 + Random Forest, patch-wise CV) | about 0.94 |
| All training labels: classical / quantum / hybrid | 0.961 / 0.966 / 0.963 (essentially a tie) |
| Real hardware (`ibm_fez`, 500 circuits, 1024 shots) | 0.965 |
| Few-label study (encoder trained on 20-300 labels) | quantum features scored higher on average, but results vary a lot between repeats |

### Honest caveats

- With plentiful labels, quantum and classical features perform about the same.
- In the few-label study, the control used so far is a **linear** feature set. A fairer control is a classical sin/cos feature map on the same graph (the circuit effectively applies sine-type nonlinearities), so the gain should not yet be claimed as a quantum advantage.
- The test set comes from only 4 patches, so differences of 1-2 points are within noise.

## Repository structure

```
notebooks/   Qollabhack.ipynb, full pipeline and experiments
data/        Z.npy, y.npy, pid.npy, test.npy, graph_meta.npz, quantum_n15.npz
results/     figures, JSON/CSV results, hardware metadata, real_quantum.npz
```

## Run

```bash
pip install numpy scikit-learn qiskit qiskit-aer matplotlib
```

Put the files from `data/` next to the notebook (or edit the paths) and open `notebooks/Qollabhack.ipynb`.
Real-hardware runs need an IBM Quantum account. **Never commit your API key**; save it once with `QiskitRuntimeService.save_account(...)`.

## Credits

- Dataset: Akhil Chhibber et al. (CC BY).
- Method inspiration: Zhang et al., *Quantum-enhanced satellite image classification*, arXiv:2602.18350.
- Built with Qiskit, PyTorch, scikit-learn.
