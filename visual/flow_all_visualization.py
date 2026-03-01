# -*- coding: utf-8 -*-
"""
厦门共享单车骑行流动轨迹可视化
功能：读取共享单车数据，生成OD（起点-终点）聚合矩阵，并生成交互式地图
"""

import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visual")

def load_and_process_data():
    """加载并处理共享单车骑行数据"""
    print("正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 删除经纬度为空的数据行
    df = df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
    
    print(f"起点纬度范围: {df['start_lat'].min():.4f} - {df['start_lat'].max():.4f}")
    print(f"起点经度范围: {df['start_lng'].min():.4f} - {df['start_lng'].max():.4f}")
    
    # 网格大小，用于OD聚合（约1km x 1km的网格）
    grid_size = 0.01
    
    def create_od_matrix(data, bike_type):
        """创建OD（起点-终点）聚合矩阵"""
        data = data.copy()
        
        # 计算每个骑行记录所属的网格坐标（向下取整）
        data['start_grid_lat'] = (data['start_lat'] // grid_size) * grid_size
        data['start_grid_lng'] = (data['start_lng'] // grid_size) * grid_size
        data['end_grid_lat'] = (data['end_lat'] // grid_size) * grid_size
        data['end_grid_lng'] = (data['end_lng'] // grid_size) * grid_size
        
        # 创建网格唯一标识符
        data['start_key'] = data['start_grid_lat'].astype(str) + '_' + data['start_grid_lng'].astype(str)
        data['end_key'] = data['end_grid_lat'].astype(str) + '_' + data['end_grid_lng'].astype(str)
        
        # 按起点-终点网格对进行分组聚合
        od = data.groupby(['start_key', 'end_key']).agg({
            'start_grid_lat': 'mean',
            'start_grid_lng': 'mean',
            'end_grid_lat': 'mean',
            'end_grid_lng': 'mean',
            'ride_id': 'count',
            'ride_dis': 'mean'
        }).reset_index()
        
        # 过滤掉起点和终点在同一网格的记录
        od = od[od['start_key'] != od['end_key']]
        
        # 将聚合结果转换为前端需要的格式
        flow_data = []
        for _, row in od.iterrows():
            # 计算网格中心点坐标
            start_lat = float(row['start_grid_lat']) + grid_size / 2
            start_lng = float(row['start_grid_lng']) + grid_size / 2
            end_lat = float(row['end_grid_lat']) + grid_size / 2
            end_lng = float(row['end_grid_lng']) + grid_size / 2
            
            # 数据有效性检查：过滤异常坐标（厦门合理范围：纬度>20，经度>100）
            if start_lat > 20 and start_lng > 100 and end_lat > 20 and end_lng > 100:
                flow_data.append({
                    'bike_type': bike_type,
                    'start': [start_lat, start_lng],
                    'end': [end_lat, end_lng],
                    'count': int(row['ride_id']),
                    'avg_distance': float(row['ride_dis'])
                })
        
        return flow_data
    
    print("正在处理单车OD矩阵...")
    # 筛选单车数据并创建OD矩阵
    df_single = df[df['bike_type'] == '单车']
    single_flow = create_od_matrix(df_single, '单车')
    
    print("正在处理助力车OD矩阵...")
    # 筛选助力车数据并创建OD矩阵
    df_ebike = df[df['bike_type'] == '助力车']
    ebike_flow = create_od_matrix(df_ebike, '助力车')
    
    # 计算地图中心点坐标
    center_lat = (df['start_lat'].mean() + df['end_lat'].mean()) / 2
    center_lng = (df['start_lng'].mean() + df['end_lng'].mean()) / 2
    
    # 整合所有数据
    result = {
        'center': [center_lat, center_lng],
        'single_flow': single_flow,
        'ebike_flow': ebike_flow,
        'total_rides': len(df),
        'single_count': len(df_single),
        'ebike_count': len(df_ebike),
        'single_od_count': len(single_flow),
        'ebike_od_count': len(ebike_flow)
    }
    
    # 保存JSON文件
    output_json = os.path.join(OUTPUT_DIR, "flow_all_data.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"单车OD对数: {len(single_flow)}")
    print(f"助力车OD对数: {len(ebike_flow)}")
    print(f"数据已保存到: {output_json}")
    return result

def create_html_map():
    """生成交互式HTML地图（使用Leaflet.js）"""
    print("正在生成HTML地图...")
    
    # 读取处理后的JSON数据
    output_json = os.path.join(OUTPUT_DIR, "flow_all_data.json")
    with open(output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    single_flow = data['single_flow']
    ebike_flow = data['ebike_flow']
    
    # 将Python列表转换为JavaScript代码中的JSON字符串
    single_json_str = json.dumps(single_flow)
    ebike_json_str = json.dumps(ebike_flow)
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>厦门共享单车骑行流动轨迹图（全部数据）</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        
        .control-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            min-width: 200px;
        }}
        
        .control-panel h3 {{ 
            margin-bottom: 15px; 
            color: #333;
            font-size: 16px;
            text-align: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        
        .btn {{
            display: block;
            width: 100%;
            padding: 12px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            color: white;
            transition: all 0.3s;
            font-weight: bold;
        }}
        
        .btn:hover {{ 
            opacity: 0.9; 
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .btn-single {{ 
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }}
        
        .btn-ebike {{ 
            background: linear-gradient(135deg, #3498db, #2980b9);
        }}
        
        .btn-all-show {{ 
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
        }}
        
        .btn-all-hide {{ 
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        }}
        
        .legend {{
            position: fixed;
            bottom: 30px;
            left: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 220px;
        }}
        
        .legend h4 {{ 
            margin-bottom: 15px; 
            font-size: 14px;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        
        .legend-line {{
            width: 40px;
            height: 4px;
            margin-right: 12px;
            border-radius: 2px;
        }}
        
        .legend-text {{
            font-size: 13px;
            color: #555;
        }}
        
        .title {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            text-align: center;
            color: white;
        }}
        
        .title h1 {{ font-size: 20px; margin-bottom: 5px; }}
        .title p {{ font-size: 12px; opacity: 0.9; }}
        
        .stats {{
            position: fixed;
            top: 100px;
            left: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 13px;
        }}
        
        .stats-value {{
            font-weight: bold;
            color: #333;
        }}
        
        .info-box {{
            position: fixed;
            top: 200px;
            left: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 12px;
            color: #666;
            max-width: 220px;
        }}
        
        .slider-container {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            min-width: 200px;
        }}
        
        .slider-container label {{
            font-size: 13px;
            color: #333;
            display: block;
            margin-bottom: 8px;
        }}
        
        .slider-container input {{
            width: 100%;
        }}
        
        .slider-value {{
            font-size: 12px;
            color: #666;
            text-align: center;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="title">
        <h1>🚲 厦门共享单车骑行流动轨迹图</h1>
        <p>全部数据 · 聚合显示</p>
    </div>
    
    <div class="stats">
        <div><span class="stats-value">{data['total_rides']:,}</span> 条总骑行</div>
        <div><span class="stats-value">{data['single_count']:,}</span> 条单车</div>
        <div><span class="stats-value">{data['ebike_count']:,}</span> 条助力车</div>
    </div>
    
    <div class="info-box">
        <b>聚合统计:</b><br>
        🟢 单车OD对: {data['single_od_count']:,}<br>
        🔵 助力车OD对: {data['ebike_od_count']:,}<br><br>
        💡 线条粗细表示流量大小<br>
        💡 点击线条查看详情
    </div>
    
    <div class="control-panel">
        <h3>🗂️ 图层控制</h3>
        <button class="btn btn-single" id="btnSingle" onclick="toggleLayer('single')">✅ 显示/隐藏 单车</button>
        <button class="btn btn-ebike" id="btnEbike" onclick="toggleLayer('ebike')">✅ 显示/隐藏 助力车</button>
        <button class="btn btn-all-show" onclick="showAll()">👁️ 显示全部</button>
        <button class="btn btn-all-hide" onclick="hideAll()">🔒 隐藏全部</button>
    </div>
    
    <div class="legend">
        <h4>📖 图例说明</h4>
        <div class="legend-item">
            <div class="legend-line" style="background: linear-gradient(90deg, #2ecc71, #27ae60);"></div>
            <span class="legend-text">单车轨迹</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: linear-gradient(90deg, #3498db, #2980b9);"></div>
            <span class="legend-text">助力车轨迹</span>
        </div>
        <div class="legend-item">
            <span class="legend-text">线条粗细 = 流量大小</span>
        </div>
    </div>
    
    <div class="slider-container">
        <label>流量阈值过滤</label>
        <input type="range" id="flowThreshold" min="1" max="500" value="1" onchange="updateThreshold()">
        <div class="slider-value">显示流量 ≥ <span id="thresholdValue">1</span> 的轨迹</div>
    </div>
    
    <div id="map"></div>
    
    <script>
        // 全局变量：图层显示状态和图层组
        var singleLayerVisible = true;
        var ebikeLayerVisible = true;
        var singleLayerGroup = null;
        var ebikeLayerGroup = null;
        var currentThreshold = 1;
        
        // 从Python传入的JSON数据
        var singleData = {single_json_str};
        var ebikeData = {ebike_json_str};
        
        // 调试信息：输出数据量到浏览器控制台
        console.log("单车数据:", singleData.length);
        console.log("助力车数据:", ebikeData.length);
        console.log("单车样本:", singleData[0]);
        
        // 初始化地图，中心点设为厦门市中心
        var map = L.map('map', {{
            center: [{data['center'][0]}, {data['center'][1]}],
            zoom: 12
        }});
        
        // 添加OpenStreetMap底图
        var osmTileLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap',
            maxZoom: 19
        }}).addTo(map);
        
        // 添加CartoDB浅色底图（可选）
        var cartoDBTileLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '© CartoDB',
            maxZoom: 19
        }});
        
        // 底图选项
        var baseMaps = {{
            "OpenStreetMap": osmTileLayer,
            "CartoDB 浅色": cartoDBTileLayer
        }};
        
        // 根据流量计算线条粗细，流量越大，线条越粗
        function getLineWeight(count, maxCount) {{
            var minWeight = 1;
            var maxWeight = 8;
            return minWeight + (count / maxCount) * (maxWeight - minWeight);
        }}
        
        // 根据流量计算透明度，流量越大，透明度越高
        function getLineOpacity(count, maxCount) {{
            return 0.3 + (count / maxCount) * 0.7;
        }}
        
        // 创建流量轨迹图层函数
        // data: 流量数据数组, color: 轨迹线颜色, threshold: 流量阈值过滤
        function createFlowLines(data, color, threshold) {{
            var layerGroup = L.layerGroup();
            
            // 找出最大流量值，用于归一化
            var maxCount = Math.max(...data.map(d => d.count));
            // 根据阈值过滤数据
            var filteredData = data.filter(d => d.count >= threshold);
            
            console.log(`创建图层: ${{data.length}} 条数据, 过滤后 ${{filteredData.length}} 条`);
            
            // 遍历每条流量记录，绘制轨迹线
            filteredData.forEach(function(flow) {{
                // 计算线条粗细和透明度
                var weight = getLineWeight(flow.count, maxCount);
                var opacity = getLineOpacity(flow.count, maxCount);
                
                // 绘制起点到终点的连线
                var polyline = L.polyline([flow.start, flow.end], {{
                    color: color,
                    weight: weight,
                    opacity: opacity,
                    smoothFactor: 1
                }});
                
                // 绑定弹出信息框
                var popupContent = `<b>骑行类型:</b> ${{flow.bike_type}}<br>` +
                                   `<b>流量:</b> ${{flow.count}} 次<br>` +
                                   `<b>平均距离:</b> ${{flow.avg_distance ? flow.avg_distance.toFixed(0) : 'N/A'}} 米<br>` +
                                   `<b>起点:</b> (${{flow.start[0].toFixed(4)}}, ${{flow.start[1].toFixed(4)}})<br>` +
                                   `<b>终点:</b> (${{flow.end[0].toFixed(4)}}, ${{flow.end[1].toFixed(4)}})`;
                
                polyline.bindPopup(popupContent);
                polyline.addTo(layerGroup);
                
                // 绘制起点标记
                var startMarker = L.circleMarker(flow.start, {{
                    radius: Math.max(2, weight),
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.8,
                    weight: 1
                }});
                startMarker.addTo(layerGroup);
                
                // 绘制终点标记
                var endMarker = L.circleMarker(flow.end, {{
                    radius: Math.max(2, weight),
                    color: '#333',
                    fillColor: '#333',
                    fillOpacity: 0.8,
                    weight: 1
                }});
                endMarker.addTo(layerGroup);
            }});
            
            return layerGroup;
        }}
        
        // 刷新图层：根据当前显示状态重新绘制
        function refreshLayers() {{
            // 先移除已有图层
            if (singleLayerGroup) map.removeLayer(singleLayerGroup);
            if (ebikeLayerGroup) map.removeLayer(ebikeLayerGroup);
            
            // 根据显示状态添加图层
            if (singleLayerVisible) {{
                singleLayerGroup = createFlowLines(singleData, '#2ecc71', currentThreshold);
                singleLayerGroup.addTo(map);
            }}
            
            if (ebikeLayerVisible) {{
                ebikeLayerGroup = createFlowLines(ebikeData, '#3498db', currentThreshold);
                ebikeLayerGroup.addTo(map);
            }}
        }}
        
        // 更新流量阈值并刷新图层
        function updateThreshold() {{
            var slider = document.getElementById('flowThreshold');
            var value = parseInt(slider.value);
            document.getElementById('thresholdValue').textContent = value;
            currentThreshold = value;
            refreshLayers();
        }}
        
        // 延迟500ms后初始化图层（等待地图加载完成）
        setTimeout(function() {{
            refreshLayers();
        }}, 500);
        
        // 添加图层控制面板
        var overlayMaps = {{
            "🟢 单车轨迹": L.layerGroup(),
            "🔵 助力车轨迹": L.layerGroup()
        }};
        
        L.control.layers(baseMaps, overlayMaps, {{collapsed: false}}).addTo(map);
        
        // 切换单个图层的显示/隐藏
        function toggleLayer(type) {{
            if (type === 'single') {{
                singleLayerVisible = !singleLayerVisible;
                if (singleLayerVisible) {{
                    refreshLayers();
                    document.getElementById('btnSingle').innerHTML = '✅ 显示/隐藏 单车';
                    document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
                }} else {{
                    if (singleLayerGroup) map.removeLayer(singleLayerGroup);
                    document.getElementById('btnSingle').innerHTML = '❌ 显示/隐藏 单车';
                    document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }} else {{
                ebikeLayerVisible = !ebikeLayerVisible;
                if (ebikeLayerVisible) {{
                    refreshLayers();
                    document.getElementById('btnEbike').innerHTML = '✅ 显示/隐藏 助力车';
                    document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
                }} else {{
                    if (ebikeLayerGroup) map.removeLayer(ebikeLayerGroup);
                    document.getElementById('btnEbike').innerHTML = '❌ 显示/隐藏 助力车';
                    document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }}
        }}
        
        // 显示所有图层
        function showAll() {{
            singleLayerVisible = true;
            ebikeLayerVisible = true;
            refreshLayers();
            
            document.getElementById('btnSingle').innerHTML = '✅ 显示/隐藏 单车';
            document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
            document.getElementById('btnEbike').innerHTML = '✅ 显示/隐藏 助力车';
            document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
        }}
        
        // 隐藏所有图层
        function hideAll() {{
            singleLayerVisible = false;
            ebikeLayerVisible = false;
            if (singleLayerGroup) map.removeLayer(singleLayerGroup);
            if (ebikeLayerGroup) map.removeLayer(ebikeLayerGroup);
            
            document.getElementById('btnSingle').innerHTML = '❌ 显示/隐藏 单车';
            document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
            document.getElementById('btnEbike').innerHTML = '❌ 显示/隐藏 助力车';
            document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
        }}
        
        // 添加比例尺
        L.control.scale({{imperial: false, metric: true}}).addTo(map);
    </script>
</body>
</html>'''
    
    # 保存HTML文件
    html_path = os.path.join(OUTPUT_DIR, 'flow_map_all.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 全部数据轨迹地图已保存到: {html_path}")
    print("请在浏览器中打开该文件查看")

# 程序入口
if __name__ == "__main__":
    # 1. 加载并处理数据
    load_and_process_data()
    # 2. 生成HTML地图
    create_html_map()
