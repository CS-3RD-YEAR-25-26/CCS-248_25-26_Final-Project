# Copilot Instructions for CCS-238-25-26-Final-Project

## Project Overview
- This repository contains final projects for CCS 248, focused on hate speech detection using deep learning (BiLSTM) for bilingual (Filipino-English) text.
- The main deliverable is a Jupyter notebook pipeline, supporting scripts, and a trained PyTorch model, with a Tkinter-based GUI for demonstration.

## Key Components
- **Jupyter Notebook**: `CS 3A/Dorado-Montenegro-Pontillas_final-project/HateSpeech_Detection_Complete.ipynb` is the main pipeline (data loading, preprocessing, training, evaluation, and GUI launch).
- **Model Code**: `rnn_model.py` defines the BiLSTM classifier. Model config is set in the notebook and passed to the class.
- **Preprocessing**: `text_preprocessing.py` provides text cleaning, tokenization, and vocabulary management.
- **Dataset Loader**: `load_unified_dataset.py` (not shown here) is used to combine multiple hate speech datasets.
- **GUI App**: `social_media_app.py` launches a Facebook-style Tkinter app for real-time hate speech detection using the trained model and vocabulary.
- **Artifacts**: `best_bilstm_model.pt` (PyTorch model), `vocabulary.pkl` (token mapping), and optionally `model_summary.json`.

## Developer Workflows
- **Run the full pipeline**: Execute all cells in the notebook for end-to-end training and evaluation. The last cell launches the GUI.
- **Retrain model**: Update data or code, then rerun from preprocessing to training cells. The best model is saved as `best_bilstm_model.pt`.
- **Test GUI**: Run the last notebook cell or execute `python social_media_app.py` in the project directory.
- **Add new data**: Update the dataset loader or place new files as needed, then rerun the notebook from the data loading cell.

## Project Conventions
- All model and preprocessing code is modularized for reuse in both the notebook and GUI.
- Model and vocabulary files are expected in the working directory for both training and inference.
- The GUI expects `best_bilstm_model.pt` and `vocabulary.pkl` to be present.
- Use English for code and comments; Filipino may appear in data or sample texts.

## External Dependencies
- PyTorch, scikit-learn, numpy, pandas, matplotlib, seaborn, tqdm, tkinter (standard library)
- All dependencies are standard for deep learning and data science in Python.

## Examples
- To launch the GUI: `python social_media_app.py`
- To retrain: Run all cells in the notebook, ensuring data and code are up to date.

## References
- See the notebook for the full workflow and code examples.
- Key files: `HateSpeech_Detection_Complete.ipynb`, `rnn_model.py`, `text_preprocessing.py`, `social_media_app.py`, `best_bilstm_model.pt`, `vocabulary.pkl`

---
If any section is unclear or missing, please provide feedback for further refinement.
