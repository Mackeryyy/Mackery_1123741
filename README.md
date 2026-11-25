# Industrial Anomaly Detection Using PatchCore

<p align="center">
  <em>Final Project – Pattern Recognition</em>
</p>

<table align="center">
  <tr><td><b>Student</b></td><td>Mackery</td></tr>
  <tr><td><b>Student ID</b></td><td>1123741</td></tr>
  <tr><td><b>GitHub</b></td><td>Mackery_1123741</td></tr>
  <tr><td><b>Model</b></td><td>PatchCore</td></tr>
</table>

## 1. Project Overview

This project implements an industrial visual anomaly detection system using **PatchCore** and the **MVTec AD Bottle dataset**.

The system identifies abnormal bottle images, including broken bottles and contamination, and generates anomaly maps to locate defective regions.

A web application developed with **Streamlit** allows users to upload images and visualize inspection results.

### Objectives

- Understand and implement the PatchCore anomaly detection method.
- Detect defective bottle images.
- Visualize anomaly maps and defect regions.
- Evaluate image-level and pixel-level performance.
- Develop an interactive inspection application.

### Technologies

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| PyTorch | Deep learning |
| Anomalib 2.7.1 | Anomaly detection framework |
| PatchCore | Detection method |
| Streamlit | Web application |
| NumPy, Pandas | Data processing |
| Matplotlib | Visualization |
| OpenCV | Image processing |

---

## 2. Dataset

This project uses the **Bottle category** of the MVTec Anomaly Detection (MVTec AD) dataset.

- Training images: 209 normal bottles
- Testing images: 83 images
- Defect categories: Broken Large, Broken Small, and Contamination

### Testing Data Distribution

| Category | Number of Images |
|---|---:|
| Good | 20 |
| Broken Large | 20 |
| Broken Small | 22 |
| Contamination | 21 |
| **Total** | **83** |

### Dataset Samples

<table align="center">
  <tr>
    <td align="center">
      <img src="good.png" width="130" alt="Normal bottle"/>
      <br/><b>Good</b>
    </td>
    <td align="center">
      <img src="contamination_1.png" width="130" alt="Contamination"/>
      <br/><b>Contamination</b>
    </td>
    <td align="center">
      <img src="broken_small.png" width="130" alt="Broken small"/>
      <br/><b>Broken Small</b>
    </td>
    <td align="center">
      <img src="broken_large_1.png" width="130" alt="Broken large"/>
      <br/><b>Broken Large</b>
    </td>
  </tr>
</table>

**Dataset source:** [MVTec AD Official Website](https://www.mvtec.com/company/research/datasets/mvtec-ad)

The dataset is not included in this repository. Please download it separately and follow its terms of use.

---

## 3. Methodology

PatchCore is a memory-bank-based anomaly detection method. It learns normal visual patterns from training images and identifies unusual patterns in test images.

### Detection Workflow

1. **Feature Extraction:** A pretrained Wide ResNet-50-2 extracts features from Layer 2 and Layer 3.
2. **Memory Bank Construction:** Normal image features are stored, with coreset sampling used to reduce the memory bank.
3. **Feature Comparison:** Test image features are compared with normal features using nearest-neighbor search.
4. **Anomaly Detection:** Larger feature distances indicate potentially abnormal patterns.
5. **Anomaly Localization:** Anomaly scores generate maps that highlight potential defective regions.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | PatchCore |
| Backbone | Wide ResNet-50-2 |
| Feature Layers | Layer 2 and Layer 3 |
| Pretrained Weights | True |
| Number of Neighbors | 6 |
| Coreset Sampling Ratio | 0.1 |
| Precision | Float32 |
| Hardware | CPU |

PatchCore is implemented through the Anomalib framework. The configuration above represents the settings used in this project.

---

## 4. Project Structure

```text
PatchCore_Project/
│
├── app.py
├── train.py
├── detailed_evaluation.py
├── pixel_evaluation.py
├── evaluate_dataset.py
├── inspect_dataset.py
│
├── training_size_experiment.py
├── threshold_analysis.py
├── compare_threshold_masks.py
├── smoothing_analysis.py
│
├── datasets/
├── results/
├── evaluation_results/
├── detailed_evaluation/
├── training_size_experiments/
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Main files:**

- `app.py` – Streamlit application for image inspection and result visualization.
- `train.py` – Configures and runs PatchCore using Anomalib.
- `detailed_evaluation.py` – Image-level classification evaluation.
- `pixel_evaluation.py` – Pixel-level anomaly localization evaluation.
- `training_size_experiment.py` – Compares different training dataset sizes.
- `threshold_analysis.py` and `smoothing_analysis.py` – Investigate threshold and smoothing effects.

---

## 5. Installation and Usage

### Requirements

- Python 3.12
- CPU-compatible PyTorch installation
- Dependencies listed in `requirements.txt`

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your actual GitHub repository address.

### 2. Create and Activate a Virtual Environment

```bash
python -m venv anomaly_env
```

On Windows PowerShell:

```powershell
.\anomaly_env\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install a compatible CPU version of PyTorch if it is not installed automatically.

### 4. Prepare the Dataset

Download MVTec AD and place the Bottle category in the following structure:

```text
datasets/
└── MVTecAD/
    └── bottle/
        ├── train/
        ├── test/
        └── ground_truth/
```

### 5. Train the Model

```bash
python train.py
```

The trained model checkpoint and related outputs are saved in the `results/` directory.

### 6. Run Evaluation (Optional)

```bash
python detailed_evaluation.py
python pixel_evaluation.py
python training_size_experiment.py
python threshold_analysis.py
python smoothing_analysis.py
```

These scripts reproduce the project's evaluation and experimental analyses. Some application pages depend on their generated outputs.

### 7. Launch the Web Application

```bash
python -m streamlit run app.py
```

Open the application in your browser:

`http://localhost:8501`

Press `Ctrl + C` in the terminal to stop the application.

---

## 6. Web Application

The Streamlit application provides an interactive interface with five main pages:

| Page | Function |
|---|---|
| Overview | Project introduction and system information |
| Image Inspection | Upload images and visualize anomaly detection |
| Inspection History | Review previous inspection records |
| Model Evaluation | View model evaluation metrics |
| Training Data Experiment | Compare performance using different training sizes |

### Image Inspection Example

Users can upload bottle images and view:

- Original image
- Anomaly map and heatmap
- Predicted defect mask
- Final detection result with defect localization

![Image Inspection](screenshots/image_inspection.png)

*Additional screenshots can be found in the `screenshots/` directory.*

---

## 7. Experimental Results

The PatchCore model was evaluated using 83 test images from the MVTec AD Bottle dataset.

### Image-Level Classification

| Metric | Custom Evaluation |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |
| Correct Predictions | 83 / 83 |

### Anomalib Evaluation

| Metric | Result |
|---|---:|
| Image AUROC | 1.0000 |
| Image F1-Score | 0.9920 |
| Pixel AUROC | 0.9856 |
| Pixel F1-Score | 0.7274 |

### Custom Pixel-Level Evaluation

| Defect Category | Pixel AUROC | Pixel F1-Score |
|---|---:|---:|
| Broken Large | 0.9829 | 0.7728 |
| Broken Small | 0.9942 | 0.7373 |
| Contamination | 0.9725 | 0.6416 |
| **Overall** | **0.9855** | **0.7259** |

*Note: Anomalib and custom evaluation metrics are obtained using different evaluation pipelines and are reported separately.*

### Training Data Size Experiment

The experiment compares four independently trained models using different amounts of normal training data.

| Training Data | Images | Image AUROC | Pixel AUROC |
|---|---:|---:|---:|
| 25% | 52 | 1.0000 | 0.9855 |
| 50% | 104 | 1.0000 | 0.9856 |
| 75% | 157 | 1.0000 | 0.9854 |
| 100% | 209 | 1.0000 | 0.9855 |

The results show that performance remained similar across the four training sizes. The 50% experiment had a slightly higher pixel AUROC, but the difference was very small.

### Additional Experiments

- **Threshold Analysis:** An exploratory threshold of 0.5774 produced an F1-score of approximately 0.6971. The threshold was selected using test data, so this result is not an unbiased final performance estimate.
- **Smoothing Analysis:** Gaussian smoothing did not improve anomaly localization in this experiment.

---

## 8. Conclusion and Limitations

The project demonstrated that PatchCore can effectively identify abnormal bottle images in the selected MVTec AD test set.

The custom evaluation achieved 100% image-level classification accuracy, while pixel-level evaluation showed strong AUROC but lower F1-scores, indicating that precise defect localization remains challenging.

### Limitations

- The experiments focus only on the Bottle category.
- Model execution uses CPU hardware.
- Defect boundary localization can be improved.
- Performance on other product categories and real industrial environments has not been evaluated.

### Future Improvements

- Test additional MVTec AD categories.
- Improve defect localization.
- Select thresholds using a separate validation dataset.
- Compare PatchCore with other anomaly detection methods.
- Explore GPU acceleration.

---

## 9. References and Acknowledgement

### References

1. Roth, K. et al. (2022). *Towards Total Recall in Industrial Anomaly Detection*. CVPR. [Research Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
2. Bergmann, P. et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR. [Research Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)
3. [Anomalib GitHub Repository](https://github.com/open-edge-platform/anomalib)
4. [Streamlit Documentation](https://docs.streamlit.io/)

### AI Tools Acknowledgement

ChatGPT (OpenAI) was used as a supporting tool for understanding technical concepts, debugging code, interpreting evaluation metrics, and improving project documentation.

The student reviewed the implementation, conducted the experiments, and verified the reported results.

### Usage and Copyright

This project was developed for academic purposes. PatchCore, Anomalib, and MVTec AD are acknowledged as the work and resources of their respective authors and providers. Please refer to their original publications, repositories, and dataset terms for licensing and usage conditions.
