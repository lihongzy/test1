# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "paper1", "output", "processed_trips.csv"), nrows=5)
print("列名:", df.columns.tolist())
print("\n前5行:")
print(df)
