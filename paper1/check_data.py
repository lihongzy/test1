# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")
df = pd.read_csv(data_path)
print(f"总记录数: {len(df)}")
print(f"\n车辆类型分布:")
print(df['bike_type'].value_counts())
print(f"\n城市分布:")
print(df['city_name'].value_counts())
print(f"\n用户数量: {df['user_guid'].nunique()}")
