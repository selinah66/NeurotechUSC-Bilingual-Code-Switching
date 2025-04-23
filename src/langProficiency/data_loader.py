import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    print("Initial shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    return df