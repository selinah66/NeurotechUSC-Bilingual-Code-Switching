# Neural Prediction of Bilingual Code-Switching Through Eye-tracking and Machine Learning

## Project Overview:
This project aims to use existing open-source eye-tracking data on code-switching in Bilingual Chinese-English individuals to train a machine learning model to predict bilingual code-switching.

## Introduction
This project explores the intersection of neuroscience, linguistics, and machine learning by leveraging eye-tracking data to predict code-switching behavior in bilingual individuals. Our goal is to develop a predictive model that can help understand how bilingual speakers switch between languages.

## Eye Tracking Data
The project uses data from a research study (https://www.nature.com/articles/s41597-025-04628-2) published in Nature, which provides code-switching data for bilingual Chinese-English individuals. The dataset includes four files:
- Descriptions: Provides a detailed description of the independent and dependent variables used in the study; outlines the factors manipulated and measured during the experiment.
- IA_Data: Excel file contains over 20,000 data points collected from 80 participants during the eye-tracking study; Used for data analysis and preprocessing to understand how bilingual individuals process code-switched sentences.
- Sentences: Excel file lists the bilingual code-switched sentences used in the experimental study to collect eye-tracking data. These sentences were designed to elicit specific linguistic behaviors from participants.
- Technical Validation: R code is used for conducting statistical analysis and validation of the study's findings.

## Installation Instructions
1. Open Terminal (on Mac/Linux) or Command Prompt (on Windows)
2. Clone the Repository:
git clone https://github.com/selinah66/NeurotechUSC-Bilingual-Code-Switching.git
3. Navigate to Project Directory:
   cd NeurotechUSC-Bilingual-Code-Switching (or your own project folder's file path)
4. Install dependencies:
   pip install pandas numpy scikit-learn (and any other packages needed for data analysis, preprocessing and machine learning models)

## Usage
Run the project code in Python with any Python IDE, eg. Visual Studio Code, PyCharm, or XCode, and the R-scripts in RStudio (making sure both RStudio and R are installed)

## Contributing to the Repository
1. Fork the repository: Click 'Fork' to create a copy in your account
2. Create a new branch labelled with your new update: 'git checkout -b (folder path)'
3. Commit your changes: 'git commit -m "Describe your update" '
4. Push to the branch: 'git push (folder path)')
5. Open a Pull Request by going to the original repository and clicking "Pull requests" -> "New pull request" to submit your changes.
