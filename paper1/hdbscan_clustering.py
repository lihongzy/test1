# -*- coding: utf-8 -*-
"""
厦门共享单车/电单车用户高频活动位置聚类分析模块

本模块使用HDBSCAN聚类算法识别用户的频繁借还车位置：
1. 量化借还位置密度关系
2. 构建层次聚类树
3. 根据出行频率调整聚类参数
4. 确定用户主要活动位置

数据来源: 经过预处理的用户出行数据
"""

import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_TRIPS_PATH = r"d:\postgraduate\20260207\workplace\paper1\output\processed_trips.csv"
INPUT_USER_STATS_PATH = r"d:\postgraduate\20260207\workplace\paper1\output\user_statistics.csv"
OUTPUT_DIR = r"d:\postgraduate\20260207\workplace\paper1\output"

K_NEIGHBORS = 5

TRIP_FREQUENCY_INTERVALS = [
    (0, 24, 3),
    (24, 48, 4),
    (48, 96, 6),
    (96, 192, 8),
    (192, 384, 10),
    (384, float('inf'), 12)
]


def get_trip_frequency_category(total_trips):
    """
    步骤3：根据出行次数确定频率分类和对应的min_cluster_size
    """
    for min_trips, max_trips, min_cluster_size in TRIP_FREQUENCY_INTERVALS:
        if total_trips < max_trips:
            return min_cluster_size
    return TRIP_FREQUENCY_INTERVALS[-1][2]


def cluster_user_locations(user_df):
    """
    对单个用户的借还位置进行HDBSCAN聚类
    """
    borrow_locations = user_df[['start_lat', 'start_lng']].values
    return_locations = user_df[['end_lat', 'end_lng']].values
    all_locations = np.vstack([borrow_locations, return_locations])

    if len(all_locations) < 3:
        return None, None, None

    total_trips = len(user_df)
    min_cluster_size = get_trip_frequency_category(total_trips)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    cluster_labels = clusterer.fit_predict(all_locations)

    return cluster_labels, all_locations, clusterer


def identify_user_primary_locations(user_df):
    """
    步骤4：识别用户的主要活动位置

    从聚类结果中，选择点数量最多的簇作为用户的主要位置
    同时最小化点到簇中心的平均距离（论文步骤4）
    """
    cluster_labels, all_locations, clusterer = cluster_user_locations(user_df)

    if cluster_labels is None or len(all_locations) < 3:
        return None

    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)

    if len(unique_labels) == 0:
        return None

    label_counts = {}
    avg_distances = {}
    cluster_centroids = {}

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_points = all_locations[mask]
        centroid = cluster_points.mean(axis=0)
        distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))

        label_counts[label] = len(cluster_points)
        avg_distances[label] = distances.mean()
        cluster_centroids[label] = centroid

    max_count = max(label_counts.values())
    candidate_labels = [label for label, count in label_counts.items() if count == max_count]

    if len(candidate_labels) > 1:
        primary_label = min(candidate_labels, key=lambda x: avg_distances[x])
    else:
        primary_label = candidate_labels[0]

    primary_mask = cluster_labels == primary_label
    primary_locations = all_locations[primary_mask]

    primary_lat = float(primary_locations[:, 0].mean())
    primary_lng = float(primary_locations[:, 1].mean())

    noise_count = int(np.sum(cluster_labels == -1))

    stability = 0
    if hasattr(clusterer, 'cluster_persistence_'):
        stability = float(np.mean(clusterer.cluster_persistence_))

    result = {
        'primary_lat': primary_lat,
        'primary_lng': primary_lng,
        'primary_cluster_size': label_counts[primary_label],
        'total_clusters': len(unique_labels),
        'num_locations': len(all_locations),
        'min_cluster_size': get_trip_frequency_category(len(user_df)),
        'noise_points': noise_count,
        'cluster_stability': stability,
        'avg_distance_to_centroid': avg_distances[primary_label]
    }

    return result


def step3_adjust_parameters(user_trip_stats):
    """
    步骤3：根据出行频率调整聚类参数
    为不同出行频率的用户组设置不同的min_cluster_size
    """
    print("\n" + "=" * 60)
    print("步骤3：根据出行频率调整聚类参数")
    print("=" * 60)

    print("出行频率分组与min_cluster_size设置:")
    for min_trips, max_trips, min_cluster_size in TRIP_FREQUENCY_INTERVALS:
        if max_trips == float('inf'):
            print(f"  - 高频用户 (>= {min_trips} 次): min_cluster_size = {min_cluster_size}")
        else:
            print(f"  - {min_trips}-{max_trips} 次: min_cluster_size = {min_cluster_size}")


def step4_determine_primary_locations(df, user_stats):
    """
    步骤4：确定用户主要活动位置
    对每个用户进行HDBSCAN聚类，选择点数量最多的簇作为主要位置
    """
    print("\n" + "=" * 60)
    print("步骤4：确定用户主要活动位置")
    print("=" * 60)

    user_ids = df['user_guid'].unique()
    print(f"开始聚类分析，共 {len(user_ids)} 名用户...")

    results = []
    processed = 0

    for user_id in user_ids:
        user_df = df[df['user_guid'] == user_id]
        result = identify_user_primary_locations(user_df)

        if result:
            result['user_guid'] = user_id
            results.append(result)

        processed += 1
        if processed % 500 == 0:
            print(f"  已处理 {processed}/{len(user_ids)} 用户...")

    results_df = pd.DataFrame(results)
    print(f"成功识别主要位置的用户数: {len(results_df)}")

    return results_df


def calculate_cluster_quality_metrics(primary_locations_df):
    """
    计算聚类质量指标
    """
    print("\n" + "=" * 60)
    print("计算聚类质量指标")
    print("=" * 60)

    if 'noise_points' in primary_locations_df.columns:
        total_noise = primary_locations_df['noise_points'].sum()
        total_locations = primary_locations_df['num_locations'].sum()
        noise_ratio = total_noise / total_locations * 100 if total_locations > 0 else 0
        print(f"噪声点比例: {noise_ratio:.2f}%")

        avg_clusters = primary_locations_df['total_clusters'].mean()
        print(f"平均簇数量: {avg_clusters:.2f}")

        avg_primary_size = primary_locations_df['primary_cluster_size'].mean()
        print(f"主要位置平均点数: {avg_primary_size:.2f}")

        avg_stability = primary_locations_df['cluster_stability'].mean()
        print(f"平均聚类稳定性: {avg_stability:.4f}")

    if 'avg_distance_to_centroid' in primary_locations_df.columns:
        avg_dist = primary_locations_df['avg_distance_to_centroid'].mean()
        print(f"主要位置平均点到中心距离: {avg_dist:.6f}")

    return primary_locations_df


def save_clustering_results(primary_locations_df, output_dir):
    """
    保存聚类结果
    """
    os.makedirs(output_dir, exist_ok=True)

    primary_path = os.path.join(output_dir, "user_primary_locations.csv")
    primary_locations_df.to_csv(primary_path, index=False, encoding='utf-8-sig')
    print(f"\n用户主要位置已保存至: {primary_path}")


def main():
    """
    主函数：执行HDBSCAN聚类分析流程
    """
    print("=" * 60)
    print("HDBSCAN聚类分析 - 识别用户高频活动位置")
    print("=" * 60)

    print("\n正在加载数据...")
    df = pd.read_csv(INPUT_TRIPS_PATH)
    user_stats = pd.read_csv(INPUT_USER_STATS_PATH, index_col=0)
    print(f"出行记录数: {len(df)}")
    print(f"用户数: {df['user_guid'].nunique()}")

    step3_adjust_parameters(user_stats)

    primary_locations_df = step4_determine_primary_locations(df, user_stats)

    calculate_cluster_quality_metrics(primary_locations_df)

    save_clustering_results(primary_locations_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("聚类分析完成!")
    print("=" * 60)

    return primary_locations_df


if __name__ == "__main__":
    primary_locations = main()
