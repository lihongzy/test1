# -*- coding: utf-8 -*-
"""
计算出行特征变量的描述性统计

计算以下变量：
1. Cycling days - 每月骑行天数
2. Cycling trips - 每月骑行次数  
3. Cycling distance - 平均骑行距离
4. ClusterCount - 常用借还车位置数量
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRIPS_PATH = os.path.join(BASE_DIR, "paper1", "output", "processed_trips.csv")
PRIMARY_LOCATIONS_PATH = os.path.join(BASE_DIR, "paper1", "output", "user_primary_locations.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "paper1", "output", "travel_characteristics.csv")

print("=" * 60)
print("计算出行特征变量的描述性统计")
print("=" * 60)

print("\n正在加载数据...")
trips_df = pd.read_csv(TRIPS_PATH)
primary_locations_df = pd.read_csv(PRIMARY_LOCATIONS_PATH)

print(f"出行记录数: {len(trips_df)}")
print(f"用户数: {trips_df['user_guid'].nunique()}")

print("\n计算各变量...")

cycling_days_stats = trips_df.groupby('user_guid')['usage_date'].nunique().reset_index()
cycling_days_stats.columns = ['user_guid', 'cycling_days']

cycling_trips_stats = trips_df.groupby('user_guid').size().reset_index(name='cycling_trips')

cycling_distance_stats = trips_df.groupby('user_guid')['straight_distance'].mean().reset_index()
cycling_distance_stats.columns = ['user_guid', 'cycling_distance']

cluster_count_df = primary_locations_df[['user_guid', 'total_clusters']].copy()
cluster_count_df.columns = ['user_guid', 'cluster_count']

user_stats = cycling_days_stats.merge(cycling_trips_stats, on='user_guid')
user_stats = user_stats.merge(cycling_distance_stats, on='user_guid')
user_stats = user_stats.merge(cluster_count_df, on='user_guid', how='left')

user_stats['cycling_distance_km'] = user_stats['cycling_distance'] / 1000

user_stats = user_stats.fillna(0)

print(f"有效用户数: {len(user_stats)}")

print("\n" + "=" * 60)
print("描述性统计结果")
print("=" * 60)

print("\n1. Cycling days (骑行天数)")
print(f"   均值: {user_stats['cycling_days'].mean():.3f}")
print(f"   标准差: {user_stats['cycling_days'].std():.3f}")
print(f"   最小值: {user_stats['cycling_days'].min():.3f}")
print(f"   最大值: {user_stats['cycling_days'].max():.3f}")

print("\n2. Cycling trips (骑行次数)")
print(f"   均值: {user_stats['cycling_trips'].mean():.3f}")
print(f"   标准差: {user_stats['cycling_trips'].std():.3f}")
print(f"   最小值: {user_stats['cycling_trips'].min():.3f}")
print(f"   最大值: {user_stats['cycling_trips'].max():.3f}")

print("\n3. Cycling distance (骑行距离, km)")
print(f"   均值: {user_stats['cycling_distance_km'].mean():.3f}")
print(f"   标准差: {user_stats['cycling_distance_km'].std():.3f}")
print(f"   最小值: {user_stats['cycling_distance_km'].min():.3f}")
print(f"   最大值: {user_stats['cycling_distance_km'].max():.3f}")

print("\n4. ClusterCount (常用位置数量)")
print(f"   均值: {user_stats['cluster_count'].mean():.3f}")
print(f"   标准差: {user_stats['cluster_count'].std():.3f}")
print(f"   最小值: {user_stats['cluster_count'].min():.3f}")
print(f"   最大值: {user_stats['cluster_count'].max():.3f}")

print("\n" + "=" * 60)
print("与论文对比")
print("=" * 60)

print("\n变量           | 论文均值 | 我们的均值 | 论文标准差 | 我们的标准差")
print("-" * 70)
print(f"Cycling days   | 13.193   | {user_stats['cycling_days'].mean():>8.3f} | 4.078     | {user_stats['cycling_days'].std():>8.3f}")
print(f"Cycling trips  | 19.657   | {user_stats['cycling_trips'].mean():>8.3f} | 9.949     | {user_stats['cycling_trips'].std():>8.3f}")
print(f"Cycling dist   | 2.996    | {user_stats['cycling_distance_km'].mean():>8.3f} | 1.572     | {user_stats['cycling_distance_km'].std():>8.3f}")
print(f"ClusterCount   | 2.465    | {user_stats['cluster_count'].mean():>8.3f} | 0.893     | {user_stats['cluster_count'].std():>8.3f}")

print("\n" + "=" * 60)
print("保存结果")
print("=" * 60)

user_stats_output = user_stats[['user_guid', 'cycling_days', 'cycling_trips', 
                                 'cycling_distance_km', 'cluster_count']]
user_stats_output.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"出行特征数据已保存至: {OUTPUT_PATH}")

print("\n数据前10行预览:")
print(user_stats_output.head(10))

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)
