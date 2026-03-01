# -*- coding: utf-8 -*-
"""
厦门共享单车/电单车数据预处理模块

本模块实现以下数据处理步骤：
1. 异常数据剔除：地域筛选、时间筛选、距离筛选
2. 同一用户ID识别：匹配DBS和EBS用户，用户筛选
3. 用户类型分类：根据DBS使用比例分类用户

数据来源: 厦门共享单车/电单车数据
车辆类型: 单车(DBS)、助力车(EBS)
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "paper1", "output")

DBS_TYPE = "单车"
EBS_TYPE = "助力车"

MIN_TRIP_DURATION = 1
MAX_TRIP_DURATION = 120
MIN_TRIP_DISTANCE = 100

MIN_DAYS = 3
MIN_TRIPS = 12

XIAMEN_LAT_MIN = 24.40
XIAMEN_LAT_MAX = 24.60
XIAMEN_LNG_MIN = 117.90
XIAMEN_LNG_MAX = 118.30

DBS_DOMINANT_THRESHOLD = 0.66
EBS_DOMINANT_THRESHOLD = 0.33


def haversine_distance(lat1, lng1, lat2, lng2):
    """
    计算两点之间的直线距离（单位：米）
    使用Haversine公式
    """
    R = 6371000
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)

    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def step1_remove_abnormal_data(df):
    """
    步骤1：异常数据剔除

    处理内容：
    1. 地域筛选：剔除借还车地理位置在厦门主城区以外的记录
    2. 字段清洗：剔除缺失值和异常字段记录
    3. 行程时间筛选：保留行程时长在1-120分钟之间的记录
    4. 行程距离筛选：剔除直线距离小于100米的记录

    参数:
        df: 原始数据DataFrame

    返回:
        清洗后的DataFrame
    """
    print("=" * 60)
    print("步骤1：异常数据剔除")
    print("=" * 60)
    original_count = len(df)

    df = df.copy()

    df = df.dropna(subset=['user_guid', 'bike_type', 'start_time', 'stop_time',
                           'start_lat', 'start_lng', 'end_lat', 'end_lng'])
    print(f"缺失值清洗后: {len(df)} 条 (剔除 {original_count - len(df)} 条)")

    df = df[df['city_name'].str.contains('厦门', na=False)]
    print(f"地域筛选后: {len(df)} 条")

    mask_in_area = (
        (df['start_lat'] >= XIAMEN_LAT_MIN) & (df['start_lat'] <= XIAMEN_LAT_MAX) &
        (df['start_lng'] >= XIAMEN_LNG_MIN) & (df['start_lng'] <= XIAMEN_LNG_MAX) &
        (df['end_lat'] >= XIAMEN_LAT_MIN) & (df['end_lat'] <= XIAMEN_LAT_MAX) &
        (df['end_lng'] >= XIAMEN_LNG_MIN) & (df['end_lng'] <= XIAMEN_LNG_MAX)
    )
    df = df[mask_in_area]
    print(f"厦门主城区筛选后: {len(df)} 条")

    df['start_time_dt'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['stop_time_dt'] = pd.to_datetime(df['stop_time'], errors='coerce')
    df = df.dropna(subset=['start_time_dt', 'stop_time_dt'])

    df['trip_duration_min'] = (df['stop_time_dt'] - df['start_time_dt']).dt.total_seconds() / 60
    df = df[(df['trip_duration_min'] >= MIN_TRIP_DURATION) & (df['trip_duration_min'] <= MAX_TRIP_DURATION)]
    print(f"行程时间筛选(1-120分钟)后: {len(df)} 条")

    df['straight_distance'] = df.apply(
        lambda row: haversine_distance(
            row['start_lat'], row['start_lng'],
            row['end_lat'], row['end_lng']
        ), axis=1
    )
    df = df[df['straight_distance'] >= MIN_TRIP_DISTANCE]
    print(f"行程距离筛选(>=100米)后: {len(df)} 条")

    print(f"\n步骤1完成: 从 {original_count} 条记录清洗为 {len(df)} 条")
    return df


def step2_identify_same_user(df):
    """
    步骤2：同一用户ID识别

    处理内容：
    1. 根据用户ID匹配DBS和EBS出行记录
    2. 统计每个用户使用DBS和EBS的天数和次数
    3. 筛选使用天数>=3天且出行次数>=12次的用户

    参数:
        df: 步骤1处理后的DataFrame

    返回:
        筛选后的DataFrame和用户统计信息
    """
    print("\n" + "=" * 60)
    print("步骤2：同一用户ID识别")
    print("=" * 60)

    df = df.copy()
    df['usage_date'] = df['start_time_dt'].dt.date

    df['is_dbs'] = (df['bike_type'] == DBS_TYPE).astype(int)
    df['is_ebs'] = (df['bike_type'] == EBS_TYPE).astype(int)

    user_stats = df.groupby('user_guid').agg({
        'is_dbs': 'sum',
        'is_ebs': 'sum',
        'usage_date': lambda x: x.nunique()
    }).rename(columns={
        'is_dbs': 'dbs_trips',
        'is_ebs': 'ebs_trips',
        'usage_date': 'usage_days'
    })

    user_stats['total_trips'] = user_stats['dbs_trips'] + user_stats['ebs_trips']

    qualified_users = user_stats[
        (user_stats['usage_days'] >= MIN_DAYS) &
        (user_stats['total_trips'] >= MIN_TRIPS)
    ]

    print(f"筛选条件: 使用天数 >= {MIN_DAYS} 天 且 出行次数 >= {MIN_TRIPS} 次")
    print(f"符合条件用户数: {len(qualified_users)}")

    qualified_user_ids = qualified_users.index.tolist()
    df_filtered = df[df['user_guid'].isin(qualified_user_ids)]

    total_trips = len(df_filtered)
    ebs_trips = df_filtered[df_filtered['bike_type'] == EBS_TYPE].shape[0]
    ebs_ratio = ebs_trips / total_trips * 100 if total_trips > 0 else 0

    print(f"筛选后出行记录总数: {total_trips}")
    print(f"  - DBS(单车)记录: {total_trips - ebs_trips}")
    print(f"  - EBS(助力车)记录: {ebs_trips} ({ebs_ratio:.2f}%)")

    user_stats_filtered = user_stats.loc[qualified_user_ids]

    print(f"\n步骤2完成: 识别出 {len(qualified_users)} 名有效用户")
    return df_filtered, user_stats_filtered


def step3_classify_users(df, user_stats):
    """
    步骤3：用户类型分类

    处理内容：
    根据DBS出行占总出行的比例，将用户分为三类：
    1. DBS主导型用户：DBS占比 (0.66, 1]
    2. 均衡型用户：DBS占比 [0.33, 0.66]
    3. EBS主导型用户：DBS占比 [0, 0.33)

    参数:
        df: 步骤2处理后的DataFrame
        user_stats: 用户统计数据

    返回:
        添加用户类型标签的DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤3：用户类型分类")
    print("=" * 60)

    user_stats = user_stats.copy()
    user_stats['dbs_ratio'] = user_stats['dbs_trips'] / user_stats['total_trips']

    def classify_user(ratio):
        if ratio > DBS_DOMINANT_THRESHOLD:
            return 'DBS主导型'
        elif ratio >= EBS_DOMINANT_THRESHOLD:
            return '均衡型'
        else:
            return 'EBS主导型'

    user_stats['user_type'] = user_stats['dbs_ratio'].apply(classify_user)

    user_type_counts = user_stats['user_type'].value_counts()
    print(f"用户类型分布:")
    for user_type in ['DBS主导型', '均衡型', 'EBS主导型']:
        count = user_type_counts.get(user_type, 0)
        ratio = count / len(user_stats) * 100
        print(f"  - {user_type}: {count} 人 ({ratio:.2f}%)")

    user_type_mapping = user_stats['user_type'].to_dict()
    df['user_type'] = df['user_guid'].map(user_type_mapping)

    print(f"\n步骤3完成: 完成用户分类")
    return df, user_stats


def save_results(df, user_stats, output_dir):
    """
    保存处理结果到文件

    参数:
        df: 处理后的出行记录DataFrame
        user_stats: 用户统计数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    trip_output_path = os.path.join(output_dir, "processed_trips.csv")
    df.to_csv(trip_output_path, index=False, encoding='utf-8-sig')
    print(f"\n出行记录已保存至: {trip_output_path}")

    user_output_path = os.path.join(output_dir, "user_statistics.csv")
    user_stats.to_csv(user_output_path, encoding='utf-8-sig')
    print(f"用户统计已保存至: {user_output_path}")


def main():
    """
    主函数：执行完整的数据处理流程
    """
    print("厦门共享单车/电单车数据预处理")
    print(f"数据来源: {DATA_PATH}")
    print("-" * 60)

    print("\n正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {len(df)} 条记录")

    df_cleaned = step1_remove_abnormal_data(df)

    df_filtered, user_stats = step2_identify_same_user(df_cleaned)

    df_final, user_stats_final = step3_classify_users(df_filtered, user_stats)

    save_results(df_final, user_stats_final, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("数据处理完成!")
    print("=" * 60)

    return df_final, user_stats_final


if __name__ == "__main__":
    df_result, user_stats_result = main()


# 厦门共享单车/电单车数据预处理
# 数据来源: d:\postgraduate\20260207\workplace\data\[张永平]XiaMen2024-共享单车、电单车.csv 
# ------------------------------------------------------------

# 正在加载数据...
# 原始数据: 457264 条记录
# ============================================================
# 步骤1：异常数据剔除
# ============================================================
# 缺失值清洗后: 457264 条 (剔除 0 条)
# 地域筛选后: 457264 条
# 厦门主城区筛选后: 401501 条
# 行程时间筛选(1-120分钟)后: 396562 条
# 行程距离筛选(>=100米)后: 375496 条

# 步骤1完成: 从 457264 条记录清洗为 375496 条

# ============================================================
# 步骤2：同一用户ID识别
# ============================================================
# 筛选条件: 使用天数 >= 3 天 且 出行次数 >= 12 次
# 符合条件用户数: 5331
# 筛选后出行记录总数: 88580
#   - DBS(单车)记录: 73939
#   - EBS(助力车)记录: 14641 (16.53%)

# 步骤2完成: 识别出 5331 名有效用户

# ============================================================
# 步骤3：用户类型分类
# ============================================================
# 用户类型分布:
#   - DBS主导型: 4255 人 (79.82%)
#   - 均衡型: 27 人 (0.51%)
#   - EBS主导型: 1049 人 (19.68%)

# 步骤3完成: 完成用户分类

# 出行记录已保存至: d:\postgraduate\20260207\workplace\paper1\output\processed_trips.csv    
# 用户统计已保存至: d:\postgraduate\20260207\workplace\paper1\output\user_statistics.csv

# ============================================================
# 数据处理完成!
# ============================================================
