# Predicting English L2 Proficiency Using Eye-Tracking in Mandarin Chinese-English Bilinguals

## Project Overview:
This project aims to use existing open-source eye-tracking data from Mandarin Chinese-English bilinguals to train a machine learning model to predict L2 English Language Proficiency based on reading behavior.

## Introduction
This project explores the intersection of neuroscience, linguistics, and machine learning by analyzing eye-tracking data when reading. Our goal is to develop predictive models that classify English L2 proficiency levels based on cognitive effort indicators captured through eye movements, such as fixation counts, dwell time, and regression path durations.

## Eye Tracking Data
The project uses data from [Wang et al., 2025] (https://www.nature.com/articles/s41597-025-04628-2) published in *Nature Scientific Data*. The dataset captures bilngual Chinese-English readers processing code-switched and monolingual sentences. The dataset includes four files:
- **Descriptions:** Provides a detailed description of the independent and dependent variables used in the study; outlines the factors manipulated and measured during the experiment.
- **IA_Data:** Eye-tracking data (20,000+ trials) from 80 participants during the eye-tracking study, including fixation, saccade, and regression metrics.
- **Sentences:** Code-switched bilingual sentences used to elicit eye-movement behaviors while reading.
- **Technical Validation:** R scripts validating statistical properties of the dataset.

## Installation Instructions
1. Open Terminal (on Mac/Linux) or Command Prompt (on Windows)
2. Clone the Repository:
```bash
git clone https://github.com/selinah66/NeurotechUSC-Bilingual-Code-Switching.git
```
3. Navigate to Project Directory:
```bash
   cd NeurotechUSC-Bilingual-Code-Switching (or your own project folder's file path)
```
4. Install required Python packages:
```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
Run Python scripts in any IDE (e.g., Visual Studio Code, PyCharm, or Xcode) to preprocess data, engineer features, and train Random Forest models.

## Repository Structure
/config/ - Scripts for configuration settings for each method step, including constants and other parameters.

/data_loader/ - Scripts to load data onto IDE

/preprocessing/ — Scripts for data cleaning and imputation

/feature_eng/ — Scripts for generating new eye-movement features, aggregation, and selecting top features for model

/eda/ - Scripts for splitting the data by L2 proficiency level

/model_experiments/ - Scripts for exploration of model selection process, and comparison of accuracy and evaluation metrics

/model_training/ — Random Forest training, hyperparameter tuning, evaluation

/visualization/ — Scripts for generating figures and model interpretation

## Contributing to the Repository
1. Fork the repository: Click 'Fork' to create a copy in your account
2. Create a new branch labelled with your new update:
```bash
git checkout -b (folder path)
```
3. Commit your changes:
```bash
git commit -m "Describe your update"
```
4. Push to the branch:
```bash
git push (folder path)
```
5. Open a Pull Request by going to the original repository and clicking "Pull requests" -> "New pull request" to submit your changes.

## License
This project is for academic and educational purposes only.

