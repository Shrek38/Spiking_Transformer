# Spiking Transformer (A²OS²A) with a Quantized, Learnable-Step K
 
**Deep Learning course project (Group 9)**
 
We take the addition-only spiking self-attention from **A²OS²A** (*Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer*, CVPR 2025) and change one thing. The paper's **full-precision ReLU K** is replaced with a **b-bit spiking neuron that has a learnable step size for each channel**. We compare b = 1 and b = 3 against the paper's full-precision K on CIFAR-10 and CIFAR-100.
 
---
 
## Idea in one paragraph
 
In A²OS²A, **Q** is binary {0, 1} and **V** is ternary {−1, 0, +1}, so attention `(Q·Kᵀ)·V` needs only additions and subtractions. **K**, however, is a full-precision ReLU output. That makes it unbounded, 32 bits per value, and the least "spiking" part of the block. The paper argues that more bits mean more information capacity, but it does not test anything between binary and full precision. We make K a **bounded 2^b-level spiking neuron**:
 
```
n_t = clip( round(v_t / θ_c), 0, 2^b − 1 )      # integer spike count per channel c
K_t = n_t · θ_c                                   # θ_c is a learnable step size for each channel
v_t ← v_t − K_t                                   # soft reset
```
 
Q stays binary, V stays ternary, and there is still no softmax and no scaling. Attention therefore remains **addition-only**, and the notebook checks this numerically. Each K value now needs only **b bits instead of 32**.
 
---
 
## Repository contents
 
| File | What it is |
|---|---|
| `dl-code.ipynb` | Complete pipeline in one notebook: data loading, model, sanity checks, training, results table, plots, and efficiency analysis. Saved with all outputs. |
| `README.md` | This file |
 
---
 
## Notebook structure
 
| § | Section | What it does |
|---|---|---|
| 0 | GPU check | Runs `nvidia-smi -L` |
| 1 | Settings | **Every setting you may want to change is in this one cell** (preset, datasets, bit-widths, epochs, seed) |
| 2 | Data | Finds the CIFAR folders automatically, decodes the images once into a cache, keeps them on the GPU as uint8, and augments on the GPU (random crop with 4-px padding and a horizontal flip) |
| 3 | Model | LIF neurons, the b-bit learnable-step K neuron, spiking self-attention, encoder blocks, and the patch-splitting module |
| 4 | Sanity checks | Checks that Q ∈ {0,1}, V ∈ {−1,0,+1}, and K uses at most 2^b levels, and that addition-only attention gives the same result as the matrix product (should print `OK`) |
| 5 | Training code | AdamW with cosine LR and warm-up, label smoothing 0.1, fp16 autocast, and gradient clipping. Saves a checkpoint every epoch and resumes after interruptions. |
| — | Speed benchmark | Optional (~20 s). Estimates the total run time on your GPU. |
| 6 | Run | CIFAR-10 and CIFAR-100 × {paper FP-K, ours b=1, ours b=3}. Finished runs are skipped. |
| 7 | Results | Results table, accuracy plots, and the smallest bit-width within 0.5 points of FP-K |
| 8 | Efficiency and K analysis | K memory, accumulate operations, Q/K/V firing rates, K level histograms, and learned step sizes θ |
| 9 | Notes | Points for the report and viva, including known limitations |
 
---
 
## Model and training setup (`fast` preset, used for the main results)
 
| Setting | Value |
|---|---|
| Model | Spiking Transformer **2-256** (2 encoder blocks, embedding dim 256, 8 heads, MLP ratio 4) |
| Parameters | ~2.56 M (CIFAR-10) / ~2.59 M (CIFAR-100) |
| Time steps T | 2 |
| Neurons | Multi-step LIF, τ = 2, hard reset; attention output threshold 0.5 |
| Patch splitting | conv-BN followed by 3 × [spike-conv-BN], max-pool after conv1 and conv2 (32 → 16 → 8, so 64 tokens), plus a relative position embedding |
| Optimiser | AdamW, lr 1.5e-3, weight decay 0.05 (none on BN, bias, or θ), 1 warm-up epoch, cosine decay |
| Batch / epochs | 256 / 20 |
| Other | Label smoothing 0.1, gradient clip 1.0, fp16 autocast, seed 42 |
| Hardware | Kaggle, Tesla T4 (single GPU) |
 
The three presets in §1 are `smoke` (1-minute test), `fast` (default), and `balanced` (T = 4, 25 epochs).
 
> This is a lighter setup than the paper's (shallower encoder, T = 2, 20 epochs), so absolute accuracies are lower than the published ones. The **comparison between FP-K and quantized K is still fair**, because every run uses the same model, seed, schedule, and augmentation.
 
---
 
## Results
 
Final-epoch test accuracy (%), taken from the saved notebook outputs. Δ is the difference from the paper's full-precision K in the same setup.
 
| Setup | Paper FP-K (32-bit) | Ours b = 1 | Ours b = 3 |
|---|---|---|---|
| CIFAR-10 · 2-256 · T=2 · 20 ep | **84.88** | 84.81 (−0.07) | 84.80 (−0.08) |
| CIFAR-10 · 4-256 · T=4 · 30 ep* | 87.30 | **87.80** (+0.50) | not run |
| CIFAR-100 · 2-256 · T=2 · 20 ep | 55.50 | 56.00 (+0.50) | **56.72** (+1.22) |
 
\* The 4-256 / T=4 / 30-epoch rows come from an earlier, larger run that was saved in `results.json` and loaded by the notebook. None of the current presets reproduces that configuration.
 
### Efficiency (2-256, T=2)
 
| | Paper FP-K | Ours b = 1 | Ours b = 3 |
|---|---|---|---|
| K memory per sample (CIFAR-10) | 256 KB | **8 KB** (32× smaller) | 24 KB (~10.7× smaller) |
| K non-zero rate (CIFAR-10) | 0.482 | **0.108** | 0.293 |
| K non-zero rate (CIFAR-100) | 0.464 | **0.180** | 0.304 |
| Attention accumulate ops per sample (CIFAR-10) | 0.70 M | 0.75 M | 0.71 M |
 
**Takeaways**
- **1-bit K matches full-precision K**: within 0.1 points on CIFAR-10 and 0.5 points ahead on CIFAR-100, with 32× less K memory.
- **b = 3 gives the best CIFAR-100 result** (+1.22 points over FP-K).
- Attention stays addition-only for every variant (verified in §4).
---
 
## How to run (Kaggle)
 
1. Create a new Kaggle notebook and upload `dl-code.ipynb`.
2. Under **Add data**, attach the **CIFAR-10** and **cifar-100-image-dataset-original** datasets. The data cell finds the `train/` and `test/` folders automatically. Each image's class is the name of its parent folder.
3. Set the accelerator to **GPU (T4 or P100)**.
4. *(Optional)* Change settings in **§1 Settings**. For example, use `PRESET = "smoke"` for a 1-minute test, or `EPOCHS_OVERRIDE = 10` to shorten every run.
5. Click **Run All**. The full `fast` plan (6 runs × 20 epochs) takes about **2 hours** on a T4, at roughly 18–20 minutes per run.
The notebook is **safe to re-run**. It skips finished runs, resumes interrupted runs from their last checkpoint, and writes everything to `/kaggle/working/results.json`.
 
**Running elsewhere:** set `INPUT_ROOT`, `WORK_DIR`, and (if auto-detection fails) `CIFAR10_ROOT` / `CIFAR100_ROOT` in §1. Each root should be the folder that contains `train/` and `test/`.
 
### Requirements
Python 3, PyTorch (CUDA), NumPy, Pillow, pandas, Matplotlib. The saved run used `torch 2.11.0+cu130`.
 
---
 
## Limitations
 
- **Single seed (42).** Differences below about 0.5 points are within run-to-run noise.
- **CIFAR only.** ImageNet-1K was not trained.
- **Efficiency figures are proxies** (K memory and accumulate-operation counts), not measured energy on neuromorphic hardware.
- **Lighter setup than the paper**, so absolute accuracies are below the published numbers. The comparison between variants is the main result.
---
 
## Reference
 
Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer (A²OS²A). *CVPR 2025.*
 
