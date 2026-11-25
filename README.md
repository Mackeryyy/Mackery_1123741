<h1 align="center"> 
  Industrial Anomaly Detection Using PatchCore
</h1>

<p align="center"> 
   <em>Final Project - Pattern Recognition</em>
</p>

<table align="center">
  
 <tr>
   <td><b> Student</b></td>
   <td>Mackery</td>
 </tr>
  
 <tr>
   <td><b>Student ID</b></td>
   <td>1123741</td>
 </tr>

 <tr>
   <td><b>Github</b></td>
   <td>Mackery_1123741</td>
 </tr>

 <tr>
    <td><b>Model</b></td>
    <td>PatchCore</td>
</tr>
</table>

## Project Overview
This project focuses on industrial visual anomaly detection using **PatchCore**, a deep learning-based method for identifying defects in product images.

The system uses the **MVTec AD Bottle dataset** to detect abnormal bottle conditions, including broken bottles and contamination.

A web application developed using **Streamlit** allows users to upload bottle images and visualize anomaly detection results.

## Project Objectives

- Understand the PatchCore anomaly detection method.
- Detect abnormal patterns in industrial bottle images.
- Generate anomaly maps and defect localization masks.
- Evaluate image-level and pixel-level performance.
- Develop an interactive web application for visual inspection.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| PyTorch | Deep learning framework |
| Anomalib 2.7.1 | Anomaly detection framework |
| PatchCore | Anomaly detection model |
| Streamlit | Web application |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Matplotlib | Data visualization |

---


## Dataset

This project uses the **MVTec Anomaly Detection (MVTec AD)** dataset, focusing on the **Bottle** category.

The dataset contains normal bottle images for training and both normal and defective images for testing.

### Dataset Summary

<table align="center">
  <tr>
    <td align="center">
      <h2>209</h2>
      <b>Training Images</b>
      <br>
      <sub>Normal bottles</sub>
    </td>
    <td align="center">
      <h2>83</h2>
      <b>Testing Images</b>
      <br>
      <sub>Normal and defective</sub>
    </td>
    <td align="center">
      <h2>3</h2>
      <b>Defect Types</b>
      <br>
      <sub>Broken and contamination</sub>
    </td>
  </tr>
</table>

### 🧪 Testing Categories

<table align="center">
  <tr>
    <th align="center">Category</th>
    <th align="center">Images</th>
    <th align="center">Description</th>
  </tr>
  <tr>
    <td align="center">🟢 Good</td>
    <td align="center">20</td>
    <td>Normal bottle images</td>
  </tr>
  <tr>
    <td align="center">🔴 Broken Large</td>
    <td align="center">20</td>
    <td>Large broken defects</td>
  </tr>
  <tr>
    <td align="center">🟠 Broken Small</td>
    <td align="center">22</td>
    <td>Small broken defects</td>
  </tr>
  <tr>
    <td align="center">🟡 Contamination</td>
    <td align="center">21</td>
    <td>Contamination defects</td>
  </tr>
</table>

### 🖼️ Dataset Samples

<h3 align="center">Dataset Samples</h3>

<table align="center">
  <tr>
    <td align="center">
      <img src="good.png" width="150" alt="Normal bottle">
      <br><b>Good</b>
    </td>
    <td align="center">
      <img src="contamination_1.png" width="150" alt="Contamination">
      <br><b>Contamination</b>
    </td>
    <td align="center">
      <img src="broken_small.png" width="150" alt="Broken small">
      <br><b>Broken Small</b>
    </td>
    <td align="center">
      <img src="broken_large_1.png" width="150" alt="Broken large">
      <br><b>Broken Large</b>
    </td>
  </tr>
</table>

### 🔗 Dataset Source

[MVTec AD — Official Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

```html
<p align="center">
  <img src="screenshots/dataset_samples.png" width="650">
</p>
<p align="center">
  <em>Examples of normal and defective bottle images</em>
</p>
