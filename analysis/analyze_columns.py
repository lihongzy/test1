import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")

df = pd.read_csv(DATA_PATH)
print("所有列名:")
print(df.columns.tolist())
print()
print("前5行数据:")
print(df.head())
print()
print("数据类型:")
print(df.dtypes)
