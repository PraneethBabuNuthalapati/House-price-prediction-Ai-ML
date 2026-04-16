import pandas as pd

DATA_PATH = "data/raw/zhvi.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def process_data(df):
    df = df.rename(columns={
        "RegionID": "ZipCode",
        "StateName": "State"
    })
    
    date_cols = df.columns[5:]
    df["LatestPrice"] = df[date_cols].iloc[:, -1]
    
    df = df.dropna(subset=["LatestPrice"])
    
    df = df[["ZipCode", "State", "LatestPrice"]]
    
    return df

if __name__ == "__main__":
    df = load_data()
    df = process_data(df)
    
    print(df.head())