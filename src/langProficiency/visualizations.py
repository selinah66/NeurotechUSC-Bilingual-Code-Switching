import matplotlib.pyplot as plt
import seaborn as sns
from config_lang import skipGraphs
from eda import split_by_proficiency

def doGraphs(df):
    """
    Generate visualizations comparing eye-tracking metrics between low and high L2 proficiency groups

    Parameters:
    -----------
    data : DataFrame
        Preprocessed eye-tracking data with L2 PROFICIENCY column
    skipGraphs : bool, default=False
        Option to skip graph generation
    """
    if skipGraphs:
        return

    # Split data by proficiency level using the dedicated function
    low_prof, high_prof = split_by_proficiency(df)

    # Plot 1. Scatter Plot: Fixation Count vs. First Saccade Amplitude
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    # Plot for low proficiency
    ax1.scatter(low_prof['IA_FIXATION_COUNT'], low_prof['IA_FIRST_SACCADE_AMPLITUDE'],
                alpha=0.5, edgecolors='w', linewidth=0.5, c='blue')
    ax1.set_title("Low L2 Proficiency")
    ax1.set_xlabel("Fixation Count")
    ax1.set_ylabel("First Saccade Amplitude")

    # Plot for high proficiency
    ax2.scatter(high_prof['IA_FIXATION_COUNT'], high_prof['IA_FIRST_SACCADE_AMPLITUDE'],
                alpha=0.5, edgecolors='w', linewidth=0.5, c='red')
    ax2.set_title("High L2 Proficiency")
    ax2.set_xlabel("Fixation Count")

    plt.suptitle("Fixation Count vs. First Saccade Amplitude by L2 Proficiency", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot 2. Bar Chart: Average fixation durations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    # Low proficiency
    avg_first_fixation = low_prof['IA_FIRST_FIXATION_DURATION'].mean()
    avg_second_fixation = low_prof['IA_SECOND_FIXATION_DURATION'].mean()
    ax1.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
    ax1.set_title("Low L2 Proficiency")
    ax1.set_xlabel('Fixation Type')
    ax1.set_ylabel('Average Duration (ms)')

    # High proficiency
    avg_first_fixation = high_prof['IA_FIRST_FIXATION_DURATION'].mean()
    avg_second_fixation = high_prof['IA_SECOND_FIXATION_DURATION'].mean()
    ax2.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
    ax2.set_title("High L2 Proficiency")
    ax2.set_xlabel('Fixation Type')

    plt.suptitle("Average Fixation Durations by L2 Proficiency", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot 3. Bar Chart: Gaze behavior comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    # Low proficiency
    regression_path_time = low_prof['IA_REGRESSION_PATH_DURATION'].sum()
    first_run_dwell_time = low_prof['IA_FIRST_RUN_DWELL_TIME'].sum()
    ax1.bar(['Regression Paths', 'First Run Dwell Time'],
            [regression_path_time, first_run_dwell_time])
    ax1.set_title("Low L2 Proficiency")
    ax1.set_xlabel('Gaze Behavior')
    ax1.set_ylabel('Total Time (ms)')

    # High proficiency
    regression_path_time = high_prof['IA_REGRESSION_PATH_DURATION'].sum()
    first_run_dwell_time = high_prof['IA_FIRST_RUN_DWELL_TIME'].sum()
    ax2.bar(['Regression Paths', 'First Run Dwell Time'],
            [regression_path_time, first_run_dwell_time])
    ax2.set_title("High L2 Proficiency")
    ax2.set_xlabel('Gaze Behavior')

    plt.suptitle("Gaze Behavior Comparison by L2 Proficiency", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Optional: Add distribution plot for IA_DWELL_TIME by proficiency
    plt.figure(figsize=(10, 6))
    sns.kdeplot(low_prof['IA_DWELL_TIME'], label='Low L2 Proficiency', color='blue')
    sns.kdeplot(high_prof['IA_DWELL_TIME'], label='High L2 Proficiency', color='red')
    plt.title('IA_DWELL_TIME Distribution by L2 Proficiency')
    plt.xlabel('IA_DWELL_TIME (ms)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_preprocessing_stages(processing_steps, column_to_plot):
    """
    Creates a single combined plot for preprocessing stages excluding standardization

    Parameters:
    -----------
    processing_steps : dict
        Dictionary returned by preprocess_pipeline(return_steps=True)
    column_to_plot : str
        Column name to visualize
    """
    # Create transformation of imputed data (for comparison)
    trans_df = handle_data_distribution(
        processing_steps['imputed'].copy(),
        [column_to_plot]
    )

    # Plot all stages before standardization in one figure
    plt.figure(figsize=(12, 7))

    # Add each processing stage with different color/style
    stages_to_plot = [
        ('filtered', 'green', '-', 'After Filtering/Cleaning'),
        ('imputed', 'orange', '-', 'After Imputation'),
        (trans_df, 'purple', '-', 'After Transformation')
    ]

    for stage_data, color, style, label in stages_to_plot:
        # Handle both DataFrame and string cases
        if isinstance(stage_data, str):
            if column_to_plot in processing_steps[stage_data].columns:
                data = processing_steps[stage_data][column_to_plot].dropna()
                sns.kdeplot(
                    data,
                    label=label,
                    color=color,
                    linestyle=style
                )
        else:
            # For the transformation DataFrame
            if column_to_plot in stage_data.columns:
                sns.kdeplot(
                    stage_data[column_to_plot].dropna(),
                    label=label,
                    color=color,
                    linestyle=style
                )

    plt.title(f'{column_to_plot} Distribution Through Processing Stages')
    plt.xlabel(f'{column_to_plot} (original scale)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print summary statistics for each stage
    print("\nSummary Statistics:")
    stages_for_stats = ['filtered', 'imputed', 'processed']
    for stage in stages_for_stats:
        if column_to_plot in processing_steps[stage].columns:
            data = processing_steps[stage][column_to_plot].dropna()
            print(f"\n{stage.capitalize()} Stage:")

def doMorePlots(etd, etd_cleaned):
    if skipGraphs:
        return
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=etd_cleaned[['IA_REGRESSION_PATH_DURATION']])
    plt.title('Box Plot of Eye Tracking Metrics')
    plt.show()
