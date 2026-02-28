import pandas as pd

DATA_PATH = r"d:\postgraduate\20260207\workplace\data\[张永平]XiaMen2024-共享单车、电单车.csv"

df = pd.read_csv(DATA_PATH)
print("所有列名:")
print(df.columns.tolist())
print()
print("前5行数据:")
print(df.head())
print()
print("数据类型:")
print(df.dtypes)
