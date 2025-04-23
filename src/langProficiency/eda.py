import pandas as pd

def split_by_proficiency(df: pd.DataFrame):
    """Split the DataFrame into low and high L2 proficiency subsets."""
    low_prof = df[df['L2 PROFICIENCY'] == 'L']
    high_prof = df[df['L2 PROFICIENCY'] == 'H']
    return low_prof, high_prof

