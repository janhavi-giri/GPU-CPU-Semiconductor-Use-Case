# GPU vs CPU Acceleration in Semiconductor Wafer Map Classification

This repository demonstrates how GPU acceleration with **NVIDIA RAPIDS** can significantly improve performance in AI/ML workflows for semiconductor manufacturing analytics. Using the **WM-811K wafer map dataset**, we apply a **Variational Autoencoder (VAE)** for feature extraction, followed by dimensionality reduction (UMAP) and classification (Random Forest).

## 📌 Key Features
- End-to-end pipeline for wafer defect classification
- Variational Autoencoder (VAE) for feature extraction
- Benchmark comparison: **CPU (scikit-learn, umap-learn)** vs **GPU (RAPIDS cuML, cuDF, cuPy)**
- Visual runtime analysis of CPU vs GPU execution
- Case study highlighting where GPUs deliver massive speedups (UMAP) and where CPUs can remain competitive (Random Forest on smaller data)

## 📊 Dataset
- **WM-811K** wafer map dataset from Kaggle: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- Contains 811,457 wafer maps across 8 defect categories: `Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full`

## 🚀 Usage

1. Open the notebook in Google Colab:
   ```bash
   GPU_CPU_UMAP_VAE_Wafer_Map_Classifier_Learning_Demo.ipynb
   ```

2. Make sure you set the runtime to **GPU** in Colab:
   - `Runtime > Change runtime type > GPU`

3. Install dependencies (see `requirements.txt`) and run the cells step by step.

4. Compare CPU vs GPU performance results in the notebook outputs.

## 🖥️ Benchmark Insights
- **UMAP**: ~38x faster on GPU due to heavy parallel computations
- **Random Forest**: CPU competitive on small-scale, low-dimensional data due to overhead in GPU memory transfers
- **Takeaway**: GPUs shine for large, parallel workloads; CPUs remain efficient for smaller or tree-based tasks.

## 📈 Visuals
The notebook also generates comparative runtime charts showing execution time and speedup factors.

## 📦 Requirements
See [`requirements.txt`](requirements.txt) for full dependency list.

---
👩‍💻 Author: Janhavi Giri  
