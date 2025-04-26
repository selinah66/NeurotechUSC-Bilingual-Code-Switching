import matplotlib.pyplot as plt
import seaborn as sns
from config_lang import MAX_DEPTH
from eda import split_by_proficiency
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import textwrap
import io
from PIL import Image


# Generate comparative visualizations for eye-tracking metrics between proficiency groups
def doGraphs(df, skipGraphs=False):
    """
    Generate comparative visualizations for eye-tracking metrics between proficiency groups

    Parameters:
    df : DataFrame
        Preprocessed eye-tracking data with L2 PROFICIENCY column
    skipGraphs : bool, default=False
        Option to skip graph generation
    """
    if skipGraphs:
        return

    # Split data by proficiency level
    low_prof, high_prof = split_by_proficiency(df)

    # 5 most important features based on feature importance
    top_features = [
        'IA_DWELL_TIME',
        'IA_FIXATION_COUNT',
        'IA_REGRESSION_PATH_DURATION',
        'IA_FIRST_FIXATION_DURATION',
        'fixation_density'
    ]

    # Check which columns exist
    available_features = df.columns.tolist()

    # Define top features to plot
    potential_features = [
        'IA_DWELL_TIME',
        'IA_FIXATION_COUNT',
        'IA_REGRESSION_PATH_DURATION',
        'IA_FIRST_FIXATION_DURATION',
        'IA_FIRST_SACCADE_AMPLITUDE'
    ]

    # Only use features that exist
    top_features = [f for f in potential_features if f in available_features]

    if 'fixation_density' in available_features:
        top_features.append('fixation_density')

    # Print available columns for debugging
    print(f"Available columns: {available_features}")
    print(f"Using features: {top_features}")

    # Plot 1: Violin plots
    plt.figure(figsize=(8, 6))

    for i, feature in enumerate(top_features[:2], 1): 
        plt.subplot(1, 2, i)

        sns.violinplot(
            x='L2 PROFICIENCY',
            y=feature,
            data=df,
            hue='L2 PROFICIENCY',
            legend=False
        )

        plt.title(feature.replace('IA_', '').replace('_', ' ').title(), fontsize=10)
        plt.xlabel('L2 Proficiency')
        
        if feature == 'IA_DWELL_TIME':
            plt.ylabel('Dwell Time (ms)')
        else:
            plt.ylabel('Fixation Count')

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.88)
    plt.suptitle("Distribution of Top 2 Features by L2 Proficiency Group", fontsize=14, y=0.98)

    plt.show()

    # Plot 2: Radar plot
    # Create radar chart
    def radar_factory(num_vars, frame='polygon'):
        import numpy as np
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        from matplotlib.spines import Spine
        from matplotlib.path import Path
        from matplotlib.transforms import Affine2D
        from matplotlib.patches import Circle, RegularPolygon

        # Calculate evenly-spaced axis angles
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        # Create radar transform
        class RadarTransform(PolarAxes.PolarTransform):
            def transform_path_non_affine(self, path):
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        # Create radar axes
        class RadarAxes(PolarAxes):
            name = 'radar'
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)
                return lines

            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    spine = Spine(axes=self, spine_type='circle',
                                  path=Path.unit_regular_polygon(num_vars))
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

    # Prepare data for radar plot
    feature_labels = [f.replace('IA_', '').replace('_', ' ').title() for f in top_features]

    # Calculate means for each feature by proficiency group
    # Scale features to 0-1 range for fair comparison
    scaled_features = {}
    for feature in top_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        range_val = max_val - min_val
        # Avoid division by zero
        if range_val == 0:
            range_val = 1
        scaled_features[feature] = (df[feature] - min_val) / range_val

    low_means = [scaled_features[feature][low_prof.index].mean() for feature in top_features]
    high_means = [scaled_features[feature][high_prof.index].mean() for feature in top_features]

    # Create radar plot
    theta = radar_factory(len(top_features), frame='polygon')

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    ax.plot(theta, low_means, color='blue', label='Low L2 Proficiency')
    ax.fill(theta, low_means, facecolor='blue', alpha=0.25)
    ax.plot(theta, high_means, color='red', label='High L2 Proficiency')
    ax.fill(theta, high_means, facecolor='red', alpha=0.25)

    ax.set_varlabels(feature_labels)
    ax.set_rlabel_position(180)  # Move radial labels to the left side
    ax.set_theta_offset(np.pi/2)  # Rotate the plot to start from the top
    ax.set_theta_direction(-1)  # Make the plot go clockwise
    
    for label, angle in zip(ax.get_xticklabels(), theta):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    
    plt.title('Mean Dwell Time by L2 Proficiency', size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# Plot preprocessing stages by feature
def plot_preprocessing_stages(processing_steps, column_to_plot, return_steps=True):
    """
    Creates a single combined plot for preprocessing stages

    Parameters:
    processing_steps : dict
        Dictionary returned by preprocess_pipeline(return_steps=True)
    column_to_plot : str
        Column name to visualize
    """
    
    plt.figure(figsize=(12, 7))

    # Add each processing stage with different color/style
    stages_to_plot = [
        ('filtered', 'green', '-', 'After Filtering/Cleaning'),
        ('imputed', 'orange', '-', 'After Imputation'),
        ('processed', 'purple', '-', 'After Processing')
    ]

    for stage_name, color, style, label in stages_to_plot:
        if column_to_plot in processing_steps[stage_name].columns:
            data = processing_steps[stage_name][column_to_plot].dropna()
            sns.kdeplot(
                data,
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
    for stage_name, _, _, label in stages_to_plot:
        if column_to_plot in processing_steps[stage_name].columns:
            data = processing_steps[stage_name][column_to_plot].dropna()
            print(f"\n{label}:")
            print(f"Mean: {data.mean():.2f}")
            print(f"Std: {data.std():.2f}")
            print(f"Min: {data.min():.2f}")
            print(f"Max: {data.max():.2f}")
            print(f"Count: {len(data)}")

# Plot feature correlation matrix
def plot_feature_correlation(X, selected_features):
    """
    Plot correlation matrix for selected numerical features

    Parameters:
    -----------
    X : DataFrame
        Feature matrix (should contain only numerical features)
    selected_features : list
        List of selected numerical feature names
    """
    # Validate input features
    missing_features = [f for f in selected_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Features missing from X: {missing_features}")

    non_numeric = X[selected_features].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features selected: {non_numeric}")

    # Calculate correlation matrix
    corr_matrix = X[selected_features].corr()

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot feature importance analysis
def plot_feature_importance_analysis(X, y, selected_features):
    """
    Plot feature importance analysis with clear explanation of features used in training
    
    Parameters:
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    selected_features : list
        List of selected feature names
    """

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3)
    
    # Plot 1: All available features
    ax1 = fig.add_subplot(gs[0, 0])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    all_importances = pd.Series(rf.feature_importances_, index=X.columns)
    all_importances.sort_values().plot(kind='barh', ax=ax1, color='lightblue')
    ax1.set_title('Available Features by\nImportance Score')
    ax1.set_xlabel('Importance Score')
    ax1.set_ylabel('Features')
    ax1.tick_params(axis='y', rotation=45)
    
    # Plot 2: Selected features importance
    ax2 = fig.add_subplot(gs[0, 1])
    selected_importances = all_importances[selected_features]
    selected_importances.sort_values().plot(kind='barh', ax=ax2, color='green')
    ax2.set_title('Selected Features by\nImportance Score')
    ax2.set_xlabel('Importance Score')
    ax2.set_ylabel('Features')
    ax2.tick_params(axis='y', rotation=50)
    
    # Plot 3: Selected features correlation with target
    ax3 = fig.add_subplot(gs[0, 2])
    correlations = X[selected_features].apply(lambda x: x.corr(y))
    correlations.sort_values().plot(kind='barh', ax=ax3, color='purple')
    ax3.set_title('Selected Features by\nCorrelation with Target')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Features')
    ax3.tick_params(axis='y', rotation=50)

    # Process y-tick labels with all transformations
    def process_label(label):
        cleaned = label.replace('IA_', '').replace('_', ' ')
        wrapped = textwrap.wrap(cleaned, width=15)
        return '\n'.join(wrapped)

    for ax in [ax1, ax2, ax3]:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        new_labels = [process_label(label) for label in labels]
        ax.set_yticklabels(new_labels, rotation=50)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.5, wspace=1.2)
    plt.suptitle('Feature Selection: Comparison of Feature Importance, Target Correlations', 
                fontsize=16, y=1.05)
    plt.show()
    
    # Print detailed feature information
    print("\nFeature Analysis")
    print("\nAll Available Features:")
    for i, (feature, importance) in enumerate(all_importances.sort_values(ascending=False).items(), 1):
        status = "SELECTED" if feature in selected_features else ""
        print(f"{i}. {feature}: {importance:.3f} {status}")
    
    print("\nSelected Features for Training:")
    for i, feature in enumerate(selected_features, 1):
        importance = all_importances[feature]
        correlation = correlations[feature]
        print(f"{i}. {feature}")
        print(f"   Importance: {importance:.3f}")
        print(f"   Correlation with target: {correlation:.3f}")
        print(f"   Description: {get_feature_description(feature)}")

# Get feature description
def get_feature_description(feature):
    descriptions = {
        'IA_FIRST_FIXATION_DURATION': 'Duration of the first fixation on a word',
        'IA_REGRESSION_PATH_DURATION': 'Time spent in regression movements',
        'IA_DWELL_TIME': 'Total time spent looking at a word',
        'IA_FIXATION_COUNT': 'Number of fixations on a word',
        'IA_FIRST_SACCADE_AMPLITUDE': 'Distance of the first eye movement',
        'regression_dwell_ratio': 'Ratio of regression time to total dwell time',
        'fixation_density': 'Number of fixations per unit time',
        'log_first_fixation': 'Log-transformed first fixation duration',
        'saccade_speed': 'Speed of the first eye movement'
    }
    return descriptions.get(feature, "No description available")

# Plot decision trees
def visualize_decision_trees(model, feature_names, max_depth=3, n_trees=3):
    """
    Visualize 3 decision trees in a Random Forest using Graphviz

    Parameters:
    model : RandomForestClassifier
        Trained Random Forest model
    feature_names : list
        Names of the features used in the model
    max_depth : int, optional
        Maximum depth of trees to visualize (default=3)
    n_trees : int, optional
        Number of trees to visualize (default=3)
    """
    # Visualize the first n_trees
    for i in range(min(n_trees, len(model.estimators_))):
        tree = model.estimators_[i]

        # Export the tree to a dot file
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=['Low', 'High'],
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=MAX_DEPTH
        )

        # Create and show the graph using GraphViz & matplotlib
        graph = graphviz.Source(dot_data)
        png_bytes = graph.pipe(format='png')
        image = Image.open(io.BytesIO(png_bytes))
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Decision Tree {i+1}')
        plt.show()

        # Print tree statistics
        n_leaves = sum(1 for node in range(tree.tree_.node_count)
                      if tree.tree_.children_left[node] == tree.tree_.children_right[node])
        print(f"\nTree {i+1} Statistics:")
        print(f"Number of nodes: {tree.tree_.node_count}")
        print(f"Max depth: {tree.tree_.max_depth}")
        print(f"Number of leaves: {n_leaves}")

def plot_descriptive_stats_comparison(processing_steps, numeric_cols):
    """
    Plot box and whisker plot of regression path duration across preprocessing stages
    
    Parameters:
    -----------
    processing_steps : dict
        Dictionary returned by preprocess_pipeline(return_steps=True)
    numeric_cols : list
        List of numeric column names to analyze
    """
    # Define stages to compare
    stages = [
        ('filtered', 'After Filtering/Cleaning'),
        ('imputed', 'After Imputation'),
        ('processed', 'After Processing')
    ]
    
    # Prepare data for box plot
    box_data = []
    labels = []
    
    for stage_name, stage_label in stages:
        df = processing_steps[stage_name]
        if 'IA_REGRESSION_PATH_DURATION' in df.columns:
            data = df['IA_REGRESSION_PATH_DURATION'].dropna()
            box_data.append(data)
            labels.append(stage_label)
    
    # Create figure with two subplots: one with original scale, one with log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original scale
    box1 = ax1.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    ax1.set_title('Regression Path Duration (Original Scale)')
    ax1.set_ylabel('Duration (ms)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    box2 = ax2.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    ax2.set_yscale('log')
    ax2.set_title('Regression Path Duration (Log Scale)')
    ax2.set_ylabel('Duration (ms) - Log Scale')
    ax2.grid(True, alpha=0.3)
    
    # Customize box colors for both plots
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for box in [box1, box2]:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nRegression Path Duration Statistics")
    for data, label in zip(box_data, labels):
        print(f"\n{label}:")
        print(f"Mean: {data.mean():.2f} ms")
        print(f"Median: {data.median():.2f} ms")
        print(f"Std: {data.std():.2f} ms")
        print(f"Min: {data.min():.2f} ms")
        print(f"Max: {data.max():.2f} ms")
        print(f"Q1: {data.quantile(0.25):.2f} ms")
        print(f"Q3: {data.quantile(0.75):.2f} ms")
        print(f"Count: {len(data)}")
        
        # Print IQR and range
        iqr = data.quantile(0.75) - data.quantile(0.25)
        print(f"IQR: {iqr:.2f} ms")
        print(f"Range: {data.max() - data.min():.2f} ms")
