# -*- coding: utf-8 -*-
"""
用户分析模块
功能：分析共享单车数据中的用户特征和行为模式
"""

import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis")

def load_data():
    """加载数据"""
    print("正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，共 {len(df):,} 条骑行记录")
    return df

def analyze_users(df):
    """
    用户分析主函数
    
    分析内容：
    1. 用户数量统计
    2. 用户活跃度分析
    3. 用户骑行偏好
    4. Top用户排名
    """
    print("\n" + "="*50)
    print("开始用户分析...")
    print("="*50)
    
    # 1. 基本统计：用户数量和骑行次数
    print("\n【1. 用户基本统计】")
    total_users = df['user_guid'].nunique()
    total_rides = len(df)
    print(f"  - 总用户数: {total_users:,} 人")
    print(f"  - 总骑行次数: {total_rides:,} 次")
    print(f"  - 人均骑行次数: {total_rides/total_users:.2f} 次")
    
    # 2. 用户骑行次数分布
    print("\n【2. 用户骑行次数分布】")
    user_ride_counts = df.groupby('user_guid').size()
    
    # 一次性用户
    one_time_users = (user_ride_counts == 1).sum()
    # 活跃用户（骑行≥2次）
    active_users = (user_ride_counts >= 2).sum()
    # 频繁用户（骑行≥10次）
    frequent_users = (user_ride_counts >= 10).sum()
    # 核心用户（骑行≥50次）
    core_users = (user_ride_counts >= 50).sum()
    
    print(f"  - 一次性用户（仅1次）: {one_time_users:,} 人 ({one_time_users/total_users*100:.1f}%)")
    print(f"  - 活跃用户（≥2次）: {active_users:,} 人 ({active_users/total_users*100:.1f}%)")
    print(f"  - 频繁用户（≥10次）: {frequent_users:,} 人 ({frequent_users/total_users*100:.1f}%)")
    print(f"  - 核心用户（≥50次）: {core_users:,} 人 ({core_users/total_users*100:.1f}%)")
    
    # 3. 用户骑行次数分段统计
    print("\n【3. 用户骑行次数分段统计】")
    ride_bins = [1, 2, 5, 10, 20, 50, 100, float('inf')]
    ride_labels = ['1次', '2-4次', '5-9次', '10-19次', '20-49次', '50-99次', '100次+']
    user_ride_df = user_ride_counts.reset_index()
    user_ride_df['ride_group'] = pd.cut(
        user_ride_df[0], 
        bins=ride_bins, 
        labels=ride_labels, 
        right=False
    )
    ride_group_counts = user_ride_df['ride_group'].value_counts().sort_index()
    for group, count in ride_group_counts.items():
        print(f"  - {group}: {count:,} 人")
    
    # 4. 用户骑行偏好分析（单车 vs 助力车）
    print("\n【4. 用户骑行偏好分析】")
    user_bike_type = df.groupby(['user_guid', 'bike_type']).size().unstack(fill_value=0)
    
    # 纯单车用户
    single_only = ((user_bike_type['单车'] > 0) & (user_bike_type.get('助力车', 0) == 0)).sum()
    # 纯助力车用户
    ebike_only = ((user_bike_type.get('助力车', 0) > 0) & (user_bike_type['单车'] == 0)).sum()
    # 混合用户
    mixed = ((user_bike_type['单车'] > 0) & (user_bike_type.get('助力车', 0) > 0)).sum()
    
    print(f"  - 纯单车用户: {single_only:,} 人 ({single_only/total_users*100:.1f}%)")
    print(f"  - 纯助力车用户: {ebike_only:,} 人 ({ebike_only/total_users*100:.1f}%)")
    print(f"  - 混合用户（两种都用）: {mixed:,} 人 ({mixed/total_users*100:.1f}%)")
    
    # 5. 用户平均骑行距离和时间
    print("\n【5. 用户平均骑行数据】")
    user_stats = df.groupby('user_guid').agg({
        'ride_dis': 'mean',      # 平均骑行距离
        'ride_time': 'mean',     # 平均骑行时间
        'ride_id': 'count'       # 骑行次数
    }).rename(columns={'ride_id': 'ride_count'})
    
    avg_distance = user_stats['ride_dis'].mean()
    avg_time = user_stats['ride_time'].mean()
    print(f"  - 用户平均骑行距离: {avg_distance:.2f} 米")
    print(f"  - 用户平均骑行时间: {avg_time:.2f} 分钟")
    
    # 6. Top 10 骑行次数最多的用户
    print("\n【6. Top 10 骑行次数最多的用户】")
    top_ride_users = user_ride_counts.nlargest(10).reset_index()
    top_ride_users.columns = ['user_guid', 'ride_count']
    for i, row in top_ride_users.iterrows():
        print(f"  {i+1}. 用户 {row['user_guid'][:8]}... : {row['ride_count']} 次")
    
    # 7. Top 10 累计骑行距离最远的用户
    print("\n【7. Top 10 累计骑行距离最远的用户】")
    user_total_distance = df.groupby('user_guid')['ride_dis'].sum()
    top_distance_users = user_total_distance.nlargest(10).reset_index()
    top_distance_users.columns = ['user_guid', 'total_distance']
    for i, row in top_distance_users.iterrows():
        print(f"  {i+1}. 用户 {row['user_guid'][:8]}... : {row['total_distance']/1000:.2f} 公里")
    
    # 8. 构建返回结果
    result = {
        'total_users': int(total_users),
        'total_rides': int(total_rides),
        'avg_rides_per_user': float(total_rides/total_users),
        'one_time_users': int(one_time_users),
        'active_users': int(active_users),
        'frequent_users': int(frequent_users),
        'core_users': int(core_users),
        'single_only_users': int(single_only),
        'ebike_only_users': int(ebike_only),
        'mixed_users': int(mixed),
        'avg_ride_distance': float(avg_distance),
        'avg_ride_time': float(avg_time),
        'ride_distribution': {str(k): int(v) for k, v in ride_group_counts.items()},
        'top_users_by_rides': [
            {'user_guid': str(row['user_guid']), 'ride_count': int(row['ride_count'])} 
            for _, row in top_ride_users.iterrows()
        ],
        'top_users_by_distance': [
            {'user_guid': str(row['user_guid']), 'total_distance_km': round(float(row['total_distance'])/1000, 2)} 
            for _, row in top_distance_users.iterrows()
        ]
    }
    
    return result

def save_results(result):
    """保存分析结果到JSON文件"""
    output_path = os.path.join(OUTPUT_DIR, "user_analysis_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存到: {output_path}")

# 程序入口
if __name__ == "__main__":
    # 1. 加载数据
    df = load_data()
    
    # 2. 分析用户
    result = analyze_users(df)
    
    # 3. 保存结果
    save_results(result)
