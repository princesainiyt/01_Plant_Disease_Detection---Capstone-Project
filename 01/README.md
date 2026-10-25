# Plant Disease Detection Using Deep Learning

A multi-class plant-leaf image classification project using a baseline Convolutional Neural Network (CNN), MobileNetV2 transfer learning, and fine-tuning. A Streamlit web application lets users upload a leaf image and view the model's predicted class and confidence score.

> **Important:** This is an educational/research prototype. Predictions are model estimates, not confirmed agricultural diagnoses. Performance reported on PlantVillage does not guarantee equivalent accuracy on photographs from farms or gardens.

## Project Results

Recorded results from the project's held-out PlantVillage test evaluation:

| Model | Test accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| Baseline CNN | 87.70% | 83.75% | 87.61% |
| Original MobileNetV2 | 96.11% | 95.63% | 96.11% |
| Fine-tuned MobileNetV2 | **98.38%** | **98.05%** | **98.38%** |

- Number of classes: **38**
- Test images in the recorded deployment evaluation: **5,431**
- Selected application model: `final_plant_disease_model.keras`

Results should be checked against the final saved notebook and evaluation files before publication or academic submission. External field-image validation has not yet been established.

## Features

- Upload a leaf image (`.jpg`, `.jpeg`, or `.png`)
- Preview the uploaded image
- Predict one of 38 plant-health categories
- Display the top three class predictions
- Display the model's softmax confidence values
- Evaluate models and inspect class-wise errors using the project notebooks and CSV reports

## Project Structure

The files below are expected in the project directory. Some experiment files may be optional depending on what you retained.

```text
PlantDiseaseDetection/
├── app.py
├── final_plant_disease_model.keras
├── class_names.json
├── best_mobilenetv2.keras
├── best_mobilenetv2_finetuned.keras
├── mobilenetv2_plant_disease.keras
├── train.csv
├── validation.csv
├── test.csv
├── image_index.csv
├── class_distribution.csv
├── model_comparison.csv
├── final_model_results.json
├── final_model_comparison.csv
├── final_classification_report.csv
├── deployment_model_results.json
├── deployment_model_classification_report.csv
├── misclassified_images.csv
├── confusion_pairs.csv
├── class_error_summary.csv
├── requirements.txt
└── README.md
```

The `.keras` model files and CSV files are generated during model development. The actual contents of your folder may differ. Keep the final model and matching class-label mapping together.

## Requirements

- Python compatible with the TensorFlow version used to train/save the model
- TensorFlow
- Streamlit
- NumPy
- Pillow
- pandas and scikit-learn for evaluation scripts
- Matplotlib for plots

For reproducibility, record the Python and package versions from the environment where the model runs successfully.

## Setup and Run Locally

### 1. Open a terminal

In Windows, open Anaconda Prompt, Command Prompt, or PowerShell.

### 2. Change to the project directory

```bat
cd C:\Users\thesh\PlantDiseaseDetection
```

If your project is saved elsewhere, replace the path with your actual project directory.

### 3. (Recommended) Create and activate a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

If activation is blocked by your PowerShell execution policy, use Command Prompt or follow your institution's approved environment setup instructions.

### 4. Install dependencies

If `requirements.txt` is present:

```bat
python -m pip install -r requirements.txt
```

Otherwise, install the core application dependencies:

```bat
python -m pip install streamlit tensorflow numpy pillow
```

Use the TensorFlow version compatible with the environment that created the saved model if you encounter compatibility errors. Avoid blindly upgrading TensorFlow in a working environment.

### 5. Start Streamlit

```bat
python -m streamlit run app.py
```

Streamlit will print a local address, usually:

```text
http://localhost:8501
```

Open that address in your browser.

## Using the Application

1. Open the local Streamlit address.
2. Click **Browse files** and choose a leaf photograph.
3. Review the image preview.
4. Click **Analyze leaf**.
5. Review the predicted class, confidence score, and top-three predictions.

The current application expects the saved model and `class_names.json` to be in the same directory as `app.py`. The app's model path should point to:

```python
MODEL_PATH = PROJECT_DIR / "final_plant_disease_model.keras"
```

The label order in `class_names.json` must match the model's output-index order. Do not reorder the labels independently.

## Model and Preprocessing

The selected model is a fine-tuned MobileNetV2 classifier. The application resizes uploaded images to **224 × 224** RGB pixels and applies MobileNetV2 preprocessing, scaling pixel values from `[0, 255]` to approximately `[-1, 1]`.

Preprocessing in the application must remain consistent with preprocessing used during training. The displayed softmax score is a model confidence value, not a calibrated probability or guarantee of correctness.

## Training and Evaluation Artifacts

The project may include:

- `train.csv`, `validation.csv`, `test.csv`: image paths and labels for the data splits
- `class_names.json`: model output index to class-name mapping
- `model_comparison.csv` / `final_model_comparison.csv`: model metric comparison
- `final_model_results.json`: final evaluation summary
- `final_classification_report.csv`: class-wise metrics for the fine-tuned model
- `misclassified_images.csv`: individual errors from an evaluated model
- `confusion_pairs.csv`: frequently confused class pairs
- `class_error_summary.csv`: class-wise error summary

Check the model name and date associated with each report. Error-analysis files created for the original MobileNetV2 model should not be presented as if they describe the fine-tuned model unless they were regenerated for that model.

## Reproducibility and Academic Reporting

For a reproducible submission:

1. Keep the training notebook (`.ipynb`) with executed outputs.
2. Record Python, TensorFlow, Streamlit, and other dependency versions.
3. Document the train/validation/test split strategy and random seed.
4. Confirm that test images were not used for training or checkpoint selection.
5. Report per-class precision, recall, and F1-score for the selected final model.
6. Preserve the original models and results for comparison.
7. Validate with independent images before making real-world performance claims.

## Limitations

- Performance is reported on a PlantVillage test set; field conditions can differ substantially.
- Lighting, backgrounds, camera characteristics, leaf orientation, symptom severity, and mixed infections may affect predictions.
- The model can be confidently wrong.
- A leaf photograph may not contain enough evidence to determine the plant's actual condition.
- This prototype should not be used as the sole basis for pesticide selection, treatment, or other consequential agricultural decisions.

## Troubleshooting

### `ModuleNotFoundError`
Install the missing dependency in the same Python environment used to launch Streamlit.

```bat
python -m pip install streamlit tensorflow numpy pillow
```

### Model file not found
Confirm `final_plant_disease_model.keras` exists beside `app.py`, and verify that `MODEL_PATH` in `app.py` uses the exact filename.

### Label count does not match model outputs
Use the `class_names.json` file created for this model. Do not substitute a differently ordered list of class names.

### TensorFlow/Keras loading error
Check the TensorFlow/Keras versions used when saving the model. Restore the known-working environment before attempting package upgrades.

### Application does not refresh
Save `app.py`, then use Streamlit's rerun option or stop and restart the app:

```bat
python -m streamlit run app.py
```

## References

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics through machine learning and crowdsourcing*. arXiv:1511.08060.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.
3. TensorFlow documentation: https://www.tensorflow.org/
4. Streamlit documentation: https://docs.streamlit.io/
