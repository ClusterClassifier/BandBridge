# BandBridge: 10-Band Sentinel-2 LULC Classification (CNNs vs. Vision Transformers)

**BandBridge** is an end-to-end multispectral benchmarking and spectral-bridging framework comparing Convolutional Neural Networks (**ResNet-50**, **EfficientNet-B0**) and Vision Transformers (**ViT-B/16**, **Swin-Tiny**) on uncurated 10-band Copernicus Sentinel-2 Level-2A surface reflectance imagery over **Bengaluru, India** (Tile `T43PGQ`, `EPSG:32643`), grounded against 10 m **ESA WorldCover 2021 v200** annotations.

## Key Results ($64\times 64$ Patches, 10 Bands, $N_{\text{test}} = 2,545$)

| Model               | Paradigm            | OA (%)    | Cohen's $\kappa$ | Macro $F_1$ (%) | Mean IoU (%) | Latency (ms/patch) | GFLOPs   | Peak VRAM  |
|:------------------- |:------------------- |:---------:|:----------------:|:---------------:|:------------:|:------------------:|:--------:|:----------:|
| **Swin-Tiny**       | Hierarchical ViT    | **80.59** | **0.6695**       | **76.11**       | **62.47**    | 13.367             | 8.78     | 2,700 MB   |
| **ResNet-50**       | Residual CNN        | 75.72     | 0.5863           | 69.52           | 54.91        | **5.502**          | 0.71     | 512 MB     |
| **ViT-B/16**        | Isotropic ViT       | 74.97     | 0.5843           | 68.40           | 53.29        | 6.745              | 2.95     | 2,129 MB   |
| **EfficientNet-B0** | Mobile Inverted CNN | 73.60     | 0.5249           | 55.88           | 44.25        | 7.976              | **0.07** | **426 MB** |

### Highlights from Six Controlled Ablations

1. **Spectral Band Selection**: Expanding from 3-band RGB to 10-band MSI (adding Red-Edge `B5–B7`, Narrow NIR `B8A`, and SWIR `B11–B12`) boosts Macro $F_1$ by **+19.03%** on ResNet-50 and **+10.42%** on Swin-Tiny.
2. **Spatial Patch Size**: $64\times 64$ ($640\text{ m}\times 640\text{ m}$) resolves the context-homogeneity trade-off; $32\times 32$ lacks neighborhood context and $128\times 128$ dilutes rare classes.
3. **Weight Initialization**: Random initialization (**81.81% OA**) and domain spectral pretraining (**81.26% OA**) outperform naive 3-to-10-channel ImageNet weight inflation (**80.59% OA**) on Swin-Tiny.
4. **Data Scaling**: Swin-Tiny trained on just **40% of the data** (4,749 patches) achieves **79.02% OA / 73.14% Macro $F_1$**, outperforming ResNet-50 trained on 100% of the dataset.
5. **Class Imbalance (87.8:1 Ratio)**: ResNet-50 peaks under minority oversampling (**72.12% Macro $F_1$**), whereas Swin-Tiny peaks under square-root dampened class-weighted cross-entropy (**76.70% Macro $F_1$**).
6. **Augmentation**: Dihedral $D_4$ flips/rotations add **+5.23% OA** to Swin-Tiny (`81.06%`), while synthetic radiometric noise degrades calibrated Level-2A reflectance accuracy.

## Dataset & Preprocessing Pipeline

- **Satellite Imagery**: Sentinel-2B MSI Level-2A (`S2B_MSIL2A_20261108T051019_N0513_R019_T43PGQ`), 10 bands (`B2, B3, B4, B8` at 10 m; `B5, B6, B7, B8A, B11, B12` bilinearly upsampled from 20 m to 10 m).
- **Cloud Screening**: `MSK_CLDPRB_20m.jp2` ($P_{\text{cloud}} > 20\%$ masked); patches with $>30\%$ invalid pixels discarded.
- **Classes (5 LULC Categories, 16,962 Valid Patches)**: Cropland/Agriculture (55.93%), Sparse Veg./Shrubland (20.87%), Forest/Dense Veg. (17.73%), Urban/Built-up (4.84%), Water Bodies (0.64%). Stratified 70% / 15% / 15% split (`11,873 / 2,544 / 2,545`).

## Repository Structure

```text
BandBridge/
├── code/
│   └── bandbridge-code.ipynb
├── results/
│   └── tables/        # Benchmark, ablation & XAI CSV tables
│   └── plots/         # Comparison plots    
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

## Authors

- Manu Sharma
