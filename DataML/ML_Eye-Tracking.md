**NOTE:** You are encouraged to experiment with different visualization techniques and quality checks. You can use AI assistants to help troubleshoot issues, explain unfamiliar concepts, or suggest additional analyses.

**AI TOOLS:** ChatGPT, Claude, MetaAI, Poe.com, etc.

---

### **Task 1: Loading, Visualizing, and Quality-Checking Eye Tracking Data**

#### **Step 1: Data Acquisition**

1. **Visit Our repository:**

   - [selinah66/NeurotechUSC-Bilingual-Code... - EyeMovementData](https://github.com/selinah66/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData)

2. **Download the Eye Tracking data files from the EyeMovementData folder:**

   a) **Data Location:** 
      - Navigate to the EyeMovementData folder.

   b) **Data Files:**
      Contains two data files:
      - `IA_data.csv`
      - `Sentences.xslx`

   c) **File Formats:**
      - `IA_data.csv`:
        - 18 columns total, metrics with * indicate recommended metrics for ML model
        - Column 1: Recording Session Label, Label of the data file of participants.
        - Column 2: L2 Proficiency, Participants’ English proficiency (L represents Low, H represents High). 
        - Column 3: Trial Index, Sequential order of the trial in the recording (from 1 to 160).
        - Column 4: Trial Label, Label of the trial, unique number for the sentences.
        - Column 5: Condition, Specific conditions for each sentence (from condition 1 to 8).
        - Column 6: IA_ID, Ordinal ID of the current interest area. ??
        - Column 7: IA_LABEL, Label for the visual form of each word for which the eye-movement measures are calculated.
        - Column 8, 9: IA First & Second Fixation Duration, Provides the 1st & 2nd Fixation Duration (s) for each word.
        - Column 10: IA First Run Dwell Time*, Provides the Gaze Duration for each word. 
        - Column 11: IA Regression Path Duration*, Records the total time spent on regression paths for each word.
        - Column 12: IA Dwell Time, Provides the total dwell time on each word
        - Column 13: IA First Saccade Amplitude, Provides the angular distance the eyes travel during the initial saccadic movement when moving from one point of fixation to another for each word.
        - Column 14: IA Fixation Count*, Counts how long a participant's gaze fixates on each word.
        - Column 15: IA Skip*, Provides the PS1 (Whether a word was skipped during the first pass of reading) for each word (1 = skipped, 0 = not skipped)
        - Column 16: IA First Run Fixation Count, Provides the 1st Fixation Count for each word.
        - Column 17: IA Regression In Count*, Counts how many times a reader's gaze regresses (returns/goes back) into a word while reading
        - Column 18: IA Regression Out Count*, Counts how many times a reader's gaze regresses (moves their eyes backward) out of a word

3. **Store the data either in your Google Drive or on your local machine.**

#### **Step 2: Setting Up the Environment**
   - **Code editor** like PyCharm or VS Code with an appropriate Python environment.
   **AI Assistant Tip:** If you're unsure how to set up your environment or mount Google Drive in Colab, you can ask:
   *"How do I set up a Python environment for data analysis? Can you show me how to mount Google Drive in a Colab notebook?"*

#### **Step 3: Loading the Data**

1. **Import necessary libraries:**

   ```python
   import numpy as np
   import pandas as pd
   import sklearn
   import matplotlib.pyplot as plt
   import seaborn as sb
   ```
    ** Recommended libraries for eye-tracking: SciPy, PyJanitor, DataPrep, PyGaze, PyMovements, etc.

2. **Load the Eye Tracking data:**

   - If your data is in CSV format:

     ```python
     # Adjust the file path as needed, change eye_tracking_data to name of your dataframe
     eye_tracking_data = pd.read_csv('path/to/your/IA_data.csv')
     ```
   **AI Assistant Tip:** If you encounter issues or need help with specific file formats, you can ask:

   *"How do I load eye-tracking data from a CSV file using Python?"*

#### **Step 4: Visualizing the Data**

1. **Visualize Eye-Tracking Data:**

    Eg. Plot Bar Chart Comparing Average First and Second Fixation Duration:

     ```python
     # Calculate average fixation durations, replace 'df' with name of your dataframe
     avg_first_fixation = df.iloc[:, 7].mean()  # Assuming Column 8 is the first fixation duration
     avg_second_fixation = df.iloc[:, 8].mean()
   
     # Plot bar chart
    plt.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
    plt.xlabel('Fixation Type')
    plt.ylabel('Average Duration (s)')
    plt.title('Average Fixation Durations')
    plt.show()
     ```

2. **Generate a Scatter plot for Fixation Count vs. Saccade Amplitude:**

     ```python
     # Plot scatter plot
    plt.scatter(df.iloc[:, 13], df.iloc[:, 12])  # Assuming Column 14 is Fixation Count and Column 13 is First Saccade Amplitude
    plt.xlabel('Fixation Count')
    plt.ylabel('First Saccade Amplitude')
    plt.title('Fixation Count vs. Saccade Amplitude')
    plt.show()
     ```

   **AI Assistant Tip:** To learn more about different visualization techniques, ask:
   *"What are some effective ways to visualize Eye-Tracking data in Python?"*

#### **Step 5: Quality Check**

1. **Check for missing data or anomalies:**

   - Checking missing data:

     ```python
     print(eye_tracking_data.isnull().sum())
     ```
   - If there are missing values, either remove missing values:
   eye_tracking_cleaned = df.dropna()
   
   - OR impute missing values using the mean: (recommended, to not lose sample size/data)
   from sklearn.impute import SimpleImputer
   
   numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns   # isolates columns with numeric values
   numeric_imputer = SimpleImputer(strategy='mean')
   df_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

   * Note: Can also visualize distributions of original data, after cleaning, and after dropping na to compare), ask AI

    - Check for Outliers/Anomalies:
   

2. **Analyze Signal-to-Noise Ratio (SNR):**

   - **Note:** Estimating SNR can be complex; consider using variance or standard deviation as proxies.

   - For MNE:

     ```python
     # This function is not built-in; you'd need to define it or use alternatives
     # Here's a simple way to check signal variance across channels
     import numpy as np
     data = raw.get_data()
     channel_variances = np.var(data, axis=1)
     plt.bar(raw.ch_names, channel_variances)
     plt.xlabel('Channels')
     plt.ylabel('Variance')
     plt.title('Channel Variances')
     plt.show()
     ```

3. **Identify and handle artifacts:**

   - **Detect flat channels:**

     ```python
     flat_channels = mne.preprocessing.find_flat_channels(raw)
     print(f"Flat channels: {flat_channels['flat']}")
     ```

   - **Detect bad segments:**

     ```python
     from mne.preprocessing import annotate_bad_segments
     annotations = annotate_bad_segments(raw, picks='eeg', verbose=True)
     raw.set_annotations(annotations)
     raw.plot()
     ```

   **AI Assistant Tip:** For more advanced quality checks or artifact removal techniques, you can ask:

   *"What are common artifacts in EEG data, and how can I detect and remove them using Python?"*
   *"How can I apply filtering to clean EEG data?"*
---

### **Task 2: Whitening EEG Data Using PCA**

#### **Step 1: Understanding PCA Whitening**

1. **Principal Component Analysis (PCA)** is a dimensionality reduction technique that can also be used for whitening data.

2. **Whitening** transforms data to have unit variance and removes correlations between features.

   **AI Assistant Tip:** If you need a refresher on PCA or whitening, you can ask:

   *"What is PCA and how does it work?"*

   *"How do I perform data whitening using PCA in Python?"*

#### **Step 2: Preprocessing**

1. **Ensure your EEG data is properly formatted:**

   - The data should be in a NumPy array or pandas DataFrame.

2. **Standardize the data:**

   - Remove the mean from each channel.

   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   eeg_data_standardized = scaler.fit_transform(eeg_data)
   ```

#### **Step 3: Applying PCA Whitening**

1. **Import PCA from scikit-learn:**

   ```python
   from sklearn.decomposition import PCA
   ```

2. **Apply PCA with whitening:**

   ```python
   pca = PCA(whiten=True)
   eeg_data_whitened = pca.fit_transform(eeg_data_standardized)
   ```

3. **Inspect the explained variance ratio:**

   ```python
   plt.plot(np.cumsum(pca.explained_variance_ratio_))
   plt.xlabel('Number of Components')
   plt.ylabel('Cumulative Explained Variance')
   plt.title('Explained Variance by PCA Components')
   plt.show()
   ```

   **AI Assistant Tip:** To understand the importance of components, ask:

   *"How many principal components should I keep for my EEG data?"*

   *"What does the explained variance ratio tell me in PCA?"*

#### **Step 4: Saving or Using the Whitened Data**

1. **Save the whitened data for future use:**

   ```python
   np.save('eeg_data_whitened.npy', eeg_data_whitened)
   ```

2. **Proceed to the next task using the whitened data.**

---

### **Task 3: Independent Component Analysis (ICA) on Whitened Data**

#### **Step 1: Understanding ICA**

1. **Independent Component Analysis (ICA)** separates multivariate signals into additive subcomponents that are maximally independent.

   **AI Assistant Tip:** If you need more information on ICA, you can ask:

   *"What is Independent Component Analysis and how is it used in EEG data processing?"*

#### **Step 2: Importing Necessary Libraries**

1. **Use scikit-learn or MNE for ICA:**

   ```python
   from sklearn.decomposition import FastICA
   # or using MNE
   import mne
   ```

#### **Step 3: Applying ICA**

1. **Using scikit-learn's FastICA:**

   ```python
   ica = FastICA(n_components=number_of_components, random_state=0)
   eeg_data_ica = ica.fit_transform(eeg_data_whitened)
   ```

2. **Using MNE's ICA implementation:**

   ```python
   ica = mne.preprocessing.ICA(n_components=number_of_components, random_state=0)
   ica.fit(raw)
   ```

3. **Identify and Remove Artifacts:**

   - **Plot ICA components:**

     ```python
     ica.plot_components()
     ```

   - **Exclude artifact components (e.g., eye blinks, heartbeats):**

     ```python
     ica.exclude = [0, 1]  # Indices of components identified as artifacts
     raw_corrected = raw.copy()
     ica.apply(raw_corrected)
     ```

   **AI Assistant Tip:** To learn how to identify artifacts in ICA components, ask:

   *"How do I identify and remove EOG and ECG artifacts using ICA in EEG data?"*

#### **Step 4: Saving or Using the Cleaned Data**

1. **Save the ICA-cleaned data:**

   - For scikit-learn:

     ```python
     np.save('eeg_data_ica_cleaned.npy', eeg_data_ica)
     ```

   - For MNE:

     ```python
     raw_corrected.save('raw_ica_cleaned.fif', overwrite=True)
     ```

---

### **Task 4: Feature Extraction and Machine Learning Modeling**

#### **Step 1: Preparing the Cleaned Data**

1. **Load the cleaned EEG data from previous steps.**

   ```python
   # For NumPy arrays
   eeg_data_cleaned = np.load('eeg_data_ica_cleaned.npy')

   # For MNE Raw objects
   raw_cleaned = mne.io.read_raw_fif('raw_ica_cleaned.fif', preload=True)
   ```

#### **Step 2: Feature Extraction**

1. **Choose features to extract:**

   - **Spectrograms**
   - **Raw signal segments**
   - **Statistical features (mean, variance, etc.)**
   - **Frequency domain features**

2. **Extract Spectrogram Features:**

   ```python
   from scipy.signal import spectrogram

   f, t, Sxx = spectrogram(eeg_data_cleaned, fs=sampling_rate)
   # fs is the sampling frequency
   ```

3. **Flatten or Reshape Features for Modeling:**

   ```python
   features = Sxx.reshape(Sxx.shape[0], -1)
   ```

   **AI Assistant Tip:** To explore different feature extraction methods, ask:

   *"What are effective feature extraction techniques for EEG data?"*

   *"How do I extract frequency domain features from EEG signals?"*

#### **Step 3: Building Machine Learning Models**

1. **Split the data into training and testing sets:**

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
   ```

   - **Note:** You'll need labels corresponding to your EEG data.

2. **Choose a machine learning model:**

   - **Examples:** Support Vector Machine (SVM), Random Forest, Neural Networks.

3. **Train the Model:**

   ```python
   from sklearn.ensemble import RandomForestClassifier

   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   ```

4. **Evaluate the Model:**

   ```python
   from sklearn.metrics import accuracy_score, classification_report

   y_pred = clf.predict(X_test)
   print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
   print(classification_report(y_test, y_pred))
   ```

   **AI Assistant Tip:** If you need help with model selection or evaluation, ask:

   *"What machine learning models are suitable for EEG classification tasks?"*

   *"How can I evaluate the performance of my EEG classification model?"*

#### **Step 4: Experimentation and Improvement**

1. **Try different models and parameters.**

2. **Perform cross-validation:**

   ```python
   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(clf, features, labels, cv=5)
   print(f"Cross-validation scores: {scores}")
   ```

3. **Tune hyperparameters using Grid Search or Random Search.**
   **AI Assistant Tip:** For guidance on improving your model, ask:
   *"How can I perform hyperparameter tuning for my machine learning model?"*
   *"What techniques can I use to prevent overfitting in my Eye-Tracking data model?"*

#### **Step 5: Documenting and Reporting**
1. **Document your modeling process and results.**
2. **Create visualizations of model performance.**
   - **Confusion Matrix:**

     ```python
     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

     cm = confusion_matrix(y_test, y_pred)
     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
     disp.plot()
     plt.show()
     ```
3. **Summarize findings and suggest next steps.**
---
**Final Notes:**
- **Collaboration:** Feel free to discuss your approaches and findings with your peers.
- **AI Assistance:** Remember to leverage AI tools whenever you encounter challenges or wish to deepen your understanding.
- **Documentation:** Keep thorough notes and comment your code to make it understandable for others and your future self.
---
**Happy Coding!*
