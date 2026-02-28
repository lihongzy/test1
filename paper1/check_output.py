# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv(r'd:\postgraduate\20260207\workplace\paper1\output\processed_trips.csv', nrows=5)
print("列名:", df.columns.tolist())
print("\n前5行:")
print(df)
