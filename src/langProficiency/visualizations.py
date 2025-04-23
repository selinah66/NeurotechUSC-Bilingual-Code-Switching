import matplotlib.pyplot as plt
import seaborn as sns
from config_lang import skipGraphs
from eda import split_by_proficiency
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
import os
import numpy as np
import textwrap


def doGraphs(df, skipGraphs=False):
    """
    Generate comparative visualizations for eye-tracking metrics between proficiency groups

    Parameters:
    -----------
    df : DataFrame
        Preprocessed eye-tracking data with L2 PROFICIENCY column
    skipGraphs : bool, default=False
        Option to skip graph generation
    """
    if skipGraphs:
        return

    # Split data by proficiency level
    low_prof, high_prof = split_by_proficiency(df)

    # The 5 most important features based on feature importance
    top_features = [
        'IA_DWELL_TIME',
        'IA_FIXATION_COUNT',
        'IA_REGRESSION_PATH_DURATION',
        'IA_FIRST_FIXATION_DURATION',
        'fixation_density'
    ]

    # 1. First, check which columns actually exist in the dataframe
    available_features = df.columns.tolist()

    # Define the top features you want to plot
    potential_features = [
        'IA_DWELL_TIME',
        'IA_FIXATION_COUNT',
        'IA_REGRESSION_PATH_DURATION',
        'IA_FIRST_FIXATION_DURATION',
        'IA_FIRST_SACCADE_AMPLITUDE'
    ]

    # Only use features that exist in the dataframe
    top_features = [f for f in potential_features if f in available_features]

    if 'fixation_density' in available_features:
        top_features.append('fixation_density')

    # Print available columns for debugging
    print(f"Available columns: {available_features}")
    print(f"Using features: {top_features}")

    # 2. VIOLIN PLOTS - Fixed to use hue instead of palette directly
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        if i <= 6:  # Limit to 6 subplots maximum
            plt.subplot(2, 3, i)

            # FIXED VERSION - Using hue instead of palette directly
            sns.violinplot(
                x='L2 PROFICIENCY',
                y=feature,
                data=df,
                hue='L2 PROFICIENCY',  # Use hue instead of palette
                legend=False  # Hide legend since it's redundant
            )

            # Clean up labels
            plt.title(feature.replace('IA_', '').replace('_', ' ').title())
            plt.xlabel('L2 Proficiency')
            plt.ylabel('')

    plt.tight_layout()
    plt.suptitle("Distribution of Top Features by Proficiency Group", fontsize=16, y=1.02)
    plt.show()

    # 2. SCATTER PLOT: Top 2 features colored by proficiency
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=top_features[0],
        y=top_features[1],
        hue='L2 PROFICIENCY',
        data=df,
        palette=['blue', 'red'],
        alpha=0.7
    )
    plt.title(
        f'Scatter Plot: {top_features[0].replace("IA_", "").replace("_", " ").title()} vs {top_features[1].replace("IA_", "").replace("_", " ").title()}')
    plt.xlabel(top_features[0].replace('IA_', '').replace('_', ' ').title())
    plt.ylabel(top_features[1].replace('IA_', '').replace('_', ' ').title())
    plt.legend(title='L2 Proficiency', labels=['Low', 'High'])
    plt.tight_layout()
    plt.show()

    # 3. RADAR PLOT
    # First implement the radar factory function
    def radar_factory(num_vars, frame='polygon'):
        """Create a radar chart with `num_vars` axes."""
        import numpy as np
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        from matplotlib.spines import Spine
        from matplotlib.path import Path
        from matplotlib.transforms import Affine2D
        from matplotlib.patches import Circle, RegularPolygon

        # Calculate evenly-spaced axis angles
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):
            def transform_path_non_affine(self, path):
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

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

    # Create the radar plot
    # Prepare the data
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

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.plot(theta, low_means, color='blue', label='Low Proficiency')
    ax.fill(theta, low_means, facecolor='blue', alpha=0.25)
    ax.plot(theta, high_means, color='red', label='High Proficiency')
    ax.fill(theta, high_means, facecolor='red', alpha=0.25)

    ax.set_varlabels(feature_labels)
    plt.title('Mean Feature Values by Proficiency Group', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def plot_preprocessing_stages(processing_steps, column_to_plot, return_steps=True):
    """
    Creates a single combined plot for preprocessing stages

    Parameters:
    -----------
    processing_steps : dict
        Dictionary returned by preprocess_pipeline(return_steps=True)
    column_to_plot : str
        Column name to visualize
    """
    # Plot all stages in one figure
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

def plot_feature_importance_analysis(X, y, selected_features):
    """
    Plot feature importance analysis with clear explanation of features used in training
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    selected_features : list
        List of selected feature names
    """
    # Create a figure with three subplots
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3)
    
    # Plot 1: All available features
    ax1 = fig.add_subplot(gs[0, 0])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    all_importances = pd.Series(rf.feature_importances_, index=X.columns)
    all_importances.sort_values().plot(kind='barh', ax=ax1, color='lightblue')
    ax1.set_title('All Available Features\n(Importance Score)')
    ax1.set_xlabel('Importance Score')
    ax1.tick_params(axis='y', rotation=45)
    
    # Plot 2: Selected features importance
    ax2 = fig.add_subplot(gs[0, 1])
    selected_importances = all_importances[selected_features]
    selected_importances.sort_values().plot(kind='barh', ax=ax2, color='green')
    ax2.set_title('Selected Features for Training\n(Importance Score)')
    ax2.set_xlabel('Importance Score')
    ax2.tick_params(axis='y', rotation=50)
    
    # Plot 3: Selected features correlation with target
    ax3 = fig.add_subplot(gs[0, 2])
    correlations = X[selected_features].apply(lambda x: x.corr(y))
    correlations.sort_values().plot(kind='barh', ax=ax3, color='purple')
    ax3.set_title('Selected Features\n(Correlation with Target)')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.tick_params(axis='y', rotation=50)

    def process_label(label):
        """Process y-tick labels with all transformations"""
        # Remove 'IA_' prefix and replace underscores with spaces
        cleaned = label.replace('IA_', '').replace('_', ' ')
        # Wrap labels longer than 15 characters into 2 lines
        wrapped = textwrap.wrap(cleaned, width=15)
        return '\n'.join(wrapped)

    for ax in [ax1, ax2, ax3]:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        new_labels = [process_label(label) for label in labels]
        ax.set_yticklabels(new_labels, rotation=50)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.5, wspace=1.2)
    plt.show()
    
    # Print detailed feature information
    print("\n=== Feature Analysis ===")
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

def get_feature_description(feature):
    """Return a human-readable description of each feature"""
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

def visualize_decision_trees(model, feature_names, max_depth=3, n_trees=3):
    """
    Visualize the first few decision trees in a Random Forest using Graphviz

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    feature_names : list
        Names of the features used in the model
    max_depth : int, optional
        Maximum depth of trees to visualize (default=3)
    n_trees : int, optional
        Number of trees to visualize (default=3)
    """

    # Create output directory if it doesn't exist
    os.makedirs('tree_visualizations', exist_ok=True)

    # Visualize the first n_trees
    for i in range(min(n_trees, len(model.estimators_))):
        tree = model.estimators_[i]

        # Export the tree to a dot file
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=['Low', 'High'],  # Assuming binary classification
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth
        )

        # Define the output directory as a variable for reuse
        output_dir = "/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/outputfigures/tree_visualizations"

        # Create and save the graph
        graph = graphviz.Source(dot_data)
        graph.render(
            filename=f"{output_dir}/tree_{i + 1}.png",
            format='png',
            cleanup=True
        )
        print(f"\nTree {i+1} visualization saved as '{output_dir}/tree_{i+1}.png'")

        # Print tree statistics
        n_leaves = sum(1 for node in range(tree.tree_.node_count)
                      if tree.tree_.children_left[node] == tree.tree_.children_right[node])

        print(f"\nTree {i+1} Statistics:")
        print(f"Number of nodes: {tree.tree_.node_count}")
        print(f"Max depth: {tree.tree_.max_depth}")
        print(f"Number of leaves: {n_leaves}")
