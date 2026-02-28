import pandas as pd
import numpy as np
from datetime import datetime
import os

DATA_PATH = r"d:\postgraduate\20260207\workplace\data\[张永平]XiaMen2024-共享单车、电单车.csv"
OUTPUT_DIR = r"d:\postgraduate\20260207\workplace\paper1"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据列: {df.columns.tolist()}")
    return df

def analyze_bike_type_distribution(df):
    print("\n=== 车辆类型分布 ===")
    bike_type_counts = df['bike_type'].value_counts()
    print(bike_type_counts)
    bike_type_pct = df['bike_type'].value_counts(normalize=True) * 100
    print("\n车辆类型占比(%):")
    print(bike_type_pct)
    return bike_type_counts, bike_type_pct

def analyze_user_preferences(df):
    print("\n=== 用户偏好分析 ===")
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['stop_time'] = pd.to_datetime(df['stop_time'])
    df['hour'] = df['start_time'].dt.hour
    
    bike_type_prefs = df.groupby('bike_type').agg({
        'ride_id': 'count',
        'ride_dis': 'mean',
        'ride_time': 'mean',
    }).rename(columns={'ride_id': 'ride_count'})
    
    print("\n各车辆类型统计:")
    print(bike_type_prefs)
    
    return bike_type_prefs

def analyze_usage_patterns(df):
    print("\n=== 使用模式分析 ===")
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['hour'] = df['start_time'].dt.hour
    df['day_of_week'] = df['start_time'].dt.dayofweek
    
    hourly_usage = df.groupby(['hour', 'bike_type']).size().unstack(fill_value=0)
    print("\n每小时使用量:")
    print(hourly_usage)
    
    daily_usage = df.groupby(['day_of_week', 'bike_type']).size().unstack(fill_value=0)
    print("\n每周使用量:")
    print(daily_usage)
    
    return hourly_usage, daily_usage

def analyze_spatial_patterns(df):
    print("\n=== 空间使用模式 ===")
    
    start_locations = df.groupby('bike_type').agg({
        'start_lat': ['mean', 'std'],
        'start_lng': ['mean', 'std'],
    })
    print("\n起点位置统计:")
    print(start_locations)
    
    end_locations = df.groupby('bike_type').agg({
        'end_lat': ['mean', 'std'],
        'end_lng': ['mean', 'std'],
    })
    print("\n终点位置统计:")
    print(end_locations)
    
    return start_locations, end_locations

def analyze_trip_characteristics(df):
    print("\n=== 出行特征分析 ===")
    
    trip_stats = df.groupby('bike_type').agg({
        'ride_dis': ['mean', 'median', 'std', 'min', 'max'],
        'ride_time': ['mean', 'median', 'std', 'min', 'max'],
    })
    print("\n骑行距离和时间统计:")
    print(trip_stats)
    
    df['speed'] = df['ride_dis'] / (df['ride_time'] / 60)
    speed_stats = df.groupby('bike_type')['speed'].agg(['mean', 'median', 'std'])
    print("\n骑行速度(米/分钟):")
    print(speed_stats)
    
    return trip_stats, speed_stats

def analyze_user_behavior(df):
    print("\n=== 用户行为分析 ===")
    
    user_stats = df.groupby('user_guid').agg({
        'ride_id': 'count',
        'bike_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
        'ride_dis': 'mean',
        'ride_time': 'mean',
    }).rename(columns={
        'ride_id': 'total_rides',
        'bike_type': 'preferred_bike_type',
        'ride_dis': 'avg_distance',
        'ride_time': 'avg_duration'
    })
    
    print(f"\n用户总数: {len(user_stats)}")
    print(f"用户平均骑行次数: {user_stats['total_rides'].mean():.2f}")
    
    preferred_type = user_stats['preferred_bike_type'].value_counts()
    print("\n用户偏好车辆类型分布:")
    print(preferred_type)
    
    return user_stats

def main():
    print("开始复现论文分析...")
    print("=" * 50)
    
    df = load_data()
    
    analyze_bike_type_distribution(df)
    analyze_user_preferences(df)
    analyze_usage_patterns(df)
    analyze_spatial_patterns(df)
    analyze_trip_characteristics(df)
    analyze_user_behavior(df)
    
    print("\n" + "=" * 50)
    print("分析完成!")

if __name__ == "__main__":
    main()
