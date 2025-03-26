**NOTE:** You are encouraged to experiment with different visualization techniques and quality checks. You can use AI assistants to help troubleshoot issues, explain unfamiliar concepts, or suggest additional analyses.

**AI TOOLS:** ChatGPT, Perplexity, Claude, MetaAI, Poe.com, etc.

---

### **Task 1: Loading, Visualizing, and Quality-Checking Eye Tracking Data**

#### **Step 1: Data Acquisition**

1. **Visit Our repository:**

   - [selinah66/NeurotechUSC-Bilingual-Code... - EyeTrackingData](https://github.com/selinah66/NeurotechUSC-Bilingual-Code-Switching)

2. **Download the Eye Tracking data files from the EyeTrackingData folder:**

   a) **Data Location:** 
      - Navigate to the EyeTrackingData folder.

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
        
       - If your data is in .XLSX (excel) format:
    
       ```python
        eye_tracking_data = pd.read_excel('path/to/your/IA_data.xlsx')
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

1. **Check for missing data:**

   - Checking missing data:

     ```python
     print(eye_tracking_data.isnull().sum())
     ```
   - Drop columns with excessive missing values:
   
    ```python
    threshold = 20 ## Modify % of missing values
    missing_percentage = (eye_tracking_data.isnull().sum() / len(eye_tracking_data)) * 100 
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    print("Columns dropped:", list(columns_to_drop))  # Print the columns being dropped
    data_cleaned = eye_tracking_data.drop(columns=columns_to_drop)
   ```
     eye_tracking_cleaned = df.dropna()
   
   - Impute remaining numeric columns: (recommended, to not lose sample size/data)
   ```python 
    from sklearn.impute import SimpleImputer

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns   # isolates columns with numeric values
    numeric_imputer = SimpleImputer(strategy='mean')
    df_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    ```
   * Note: Can also visualize distributions of original data, after cleaning (dropping missing values), and after imputing to compare distributions), 
    ask AI how to do this *

2. **Check for and Remove Outliers/Anomalies:**
   (df = insert name of your dataset)

   Example for Regression Path Duration:
   ```python
    def validate_regression_path(regression_durations, min_duration=50, max_duration=5000):
      """
      Check if regression path durations are physiologically plausible.
      Typically between 50ms and 5000ms for reading tasks.
      """
      return (regression_durations >= min_duration) & (regression_durations <= max_duration)
   
    # Apply to your selected eye tracking feature, eg. regression path duration
    df_cleaned = remove_outliers(df, 'IA_REGRESSION_PATH_DURATION', method='iqr')
    print(f"Original data: {len(df)}, After removing outliers: {len(df_cleaned)}")
    ```
   * Note: Can also visualize distributions of original data, before and after cleaning (removing outliers) to compare distributions), ask AI *

3. **Event Specific Filtering:**
(done according to each eye tracking event data, and to physiologically plausible values for each event)

   ```python
    def validate_regression_path(regression_durations, min_duration=50, max_duration=5000):
        """
        Check if all regression path durations are physiologically plausible.
        Typically between 50ms and 5000ms for reading tasks.
        """
        return (regression_durations >= min_duration) & (regression_durations <= max_duration)

    # Apply to your selected eye tracking feature, eg. regression path duration
    is_valid = validate_regression_path(rpd)
    print(f"All durations are valid: {is_valid}")
    ```
    * Note: Can also add code for removing invalid data outside of plausible range, ask AI for help *

### **Task 2: Feature Engineering and Selection**

#### **Step 1: Feature Engineering **

1. **Principal Component Analysis (PCA)** is a dimensionality reduction technique that can also be used for whitening data.

2. **Whitening** transforms data to have unit variance and removes correlations between features.

   **AI Assistant Tip:** If you need a refresher on PCA or whitening, you can ask:

   *"What is PCA and how does it work?"*

   *"How do I perform data whitening using PCA in Python?"*

#### **Step 2: Feature Selection**

1. **Correlation Analysis**

   - The data should be in a NumPy array or pandas DataFrame.

2. **Standardize the data:**

   - Remove the mean from each channel.

   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   etd_standardized = scaler.fit_transform(eye_tracking_data)
   ```

### **Task 3: Feature Extraction and Machine Learning Modeling**

#### **Step 1: Feature Extraction**

1. **Choose features to extract:**

   - **Fixation Duration Ratio**
   - **Raw signal segments**
   - **Statistical features (mean, variance, etc.)**
   - **Frequency domain features**

2. **Extract Fixation Duration Ratio:**

   ```python
   etd_cleaned['Fixation Duration Ratio'] = etd_cleaned['IA Second Fixation Duration'] / etd_cleaned['IA First Fixation Duration']
   
   ```

3. **Flatten or Reshape Features for Modeling:**

   ```python
   features = Sxx.reshape(Sxx.shape[0], -1)
   ```

   **AI Assistant Tip:** To explore different feature extraction methods, ask:

   *"What are effective feature extraction techniques for Eye Tracking data?"*

#### **Step 3: Building Machine Learning Models**

1. **Split the data into training and testing sets:**

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
   ```

   - **Note:** You'll need labels corresponding to your eye tracking data, eg 1: code switching, 0: no code switching.

2. **Choose a machine learning model:**

   - **Examples:** Support Vector Machine (SVM), Random Forest, Neural Networks.

3. **Train the Model:**

   ```python
   from sklearn.ensemble import RandomForestClassifier

   clf = RandomForestClassifier(n_estimators=200, random_state=42)
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

   *"What machine learning models are suitable for bilingual code switching eye tracking classification tasks?"*

   *"How can I evaluate the performance of my eye tracking classification model?"*

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