import pandas as pd
import numpy as np
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "[张永平]XiaMen2024-共享单车、电单车.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visual")

def load_and_process_data():
    print("正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    sample_size = 2000
    
    df_single = df[df['bike_type'] == '单车'].sample(n=min(sample_size, len(df[df['bike_type'] == '单车'])), random_state=42)
    df_eBike = df[df['bike_type'] == '助力车'].sample(n=min(sample_size, len(df[df['bike_type'] == '助力车'])), random_state=42)
    
    def process_flow_data(data, bike_type):
        flow_data = []
        for _, row in data.iterrows():
            if pd.notna(row['start_lat']) and pd.notna(row['start_lng']) and \
               pd.notna(row['end_lat']) and pd.notna(row['end_lng']):
                flow_data.append({
                    'bike_type': bike_type,
                    'start': [row['start_lng'], row['start_lat']],
                    'end': [row['end_lng'], row['end_lat']],
                    'distance': row.get('ride_dis', 0),
                    'duration': row.get('ride_time', 0)
                })
        return flow_data
    
    print("正在处理单车数据...")
    single_flow = process_flow_data(df_single, '单车')
    
    print("正在处理助力车数据...")
    ebike_flow = process_flow_data(df_eBike, '助力车')
    
    center_lat = (df['start_lat'].mean() + df['end_lat'].mean()) / 2
    center_lng = (df['start_lng'].mean() + df['end_lng'].mean()) / 2
    
    result = {
        'center': [center_lat, center_lng],
        'single_flow': single_flow,
        'ebike_flow': ebike_flow,
        'total_rides': len(df),
        'single_count': len(df[df['bike_type'] == '单车']),
        'ebike_count': len(df[df['bike_type'] == '助力车'])
    }
    
    output_json = os.path.join(OUTPUT_DIR, "flow_data.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"单车轨迹数: {len(single_flow)}")
    print(f"助力车轨迹数: {len(ebike_flow)}")
    print(f"数据已保存到: {output_json}")
    return result

def create_html_map():
    print("正在生成HTML地图...")
    
    output_json = os.path.join(OUTPUT_DIR, "flow_data.json")
    with open(output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    single_flow = data['single_flow']
    ebike_flow = data['ebike_flow']
    
    def create_geojson_lines(flow_data, color, bike_type):
        features = []
        for i, flow in enumerate(flow_data):
            coords = [flow['start'], flow['end']]
            feature = {
                "type": "Feature",
                "properties": {
                    "id": i,
                    "bike_type": bike_type,
                    "distance": flow['distance'],
                    "duration": flow['duration']
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }
            features.append(feature)
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    single_geojson = create_geojson_lines(single_flow, '#2ecc71', '单车')
    ebike_geojson = create_geojson_lines(ebike_flow, '#e74c3c', '助力车')
    
    single_geojson_path = os.path.join(OUTPUT_DIR, "single_flow.geojson")
    ebike_geojson_path = os.path.join(OUTPUT_DIR, "ebike_flow.geojson")
    
    with open(single_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(single_geojson, f, ensure_ascii=False)
    
    with open(ebike_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(ebike_geojson, f, ensure_ascii=False)
    
    single_json_str = json.dumps(single_flow)
    ebike_json_str = json.dumps(ebike_flow)
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>厦门共享单车骑行流动轨迹图</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
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
            min-width: 200px;
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
        
        .title h1 {{ font-size: 22px; margin-bottom: 5px; }}
        .title p {{ font-size: 13px; opacity: 0.9; }}
        
        .stats {{
            position: fixed;
            top: 100px;
            left: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
        }}
        
        .stats-item {{
            margin: 8px 0;
            font-size: 13px;
            color: #555;
        }}
        
        .stats-value {{
            font-weight: bold;
            color: #333;
        }}
        
        .info-box {{
            position: fixed;
            top: 180px;
            left: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 12px;
            color: #666;
            max-width: 200px;
        }}
    </style>
</head>
<body>
    <div class="title">
        <h1>🚲 厦门共享单车骑行流动轨迹图</h1>
        <p>从起点到终点的流动方向可视化</p>
    </div>
    
    <div class="stats">
        <div class="stats-item">
            <span class="stats-value">{data['total_rides']:,}</span> 条总骑行记录
        </div>
        <div class="stats-item">
            <span class="stats-value">{data['single_count']:,}</span> 条单车骑行
        </div>
        <div class="stats-item">
            <span class="stats-value">{data['ebike_count']:,}</span> 条助力车骑行
        </div>
    </div>
    
    <div class="info-box">
        💡 当前显示: 每种类型随机采样 {len(single_flow)} 条轨迹<br><br>
        🟢 绿色 = 单车<br>
        🔵 蓝色 = 助力车<br><br>
        轨迹从起点指向终点
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
            <span class="legend-text">单车轨迹 (绿色)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line" style="background: linear-gradient(90deg, #3498db, #2980b9);"></div>
            <span class="legend-text">助力车轨迹 (蓝色)</span>
        </div>
        <div class="legend-item">
            <span class="legend-text">→ 起点指向终点</span>
        </div>
    </div>
    
    <div id="map"></div>
    
    <script>
        var singleLayerVisible = true;
        var ebikeLayerVisible = true;
        var singleLayerGroup = null;
        var ebikeLayerGroup = null;
        
        var singleData = {single_json_str};
        var ebikeData = {ebike_json_str};
        
        var map = L.map('map', {{
            center: {json.dumps(data['center'])},
            zoom: 12
        }});
        
        var osmTileLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        var cartoDBTileLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '© CartoDB',
            maxZoom: 19
        }});
        
        var baseMaps = {{
            "OpenStreetMap": osmTileLayer,
            "CartoDB 浅色": cartoDBTileLayer
        }};
        
        function createFlowLines(data, color) {{
            var layerGroup = L.layerGroup();
            
            data.forEach(function(flow, index) {{
                if (index > 500) return;
                
                var polyline = L.polyline([flow.start, flow.end], {{
                    color: color,
                    weight: 1.5,
                    opacity: 0.6,
                    smoothFactor: 1
                }});
                
                var popupContent = `<b>骑行类型:</b> ${{flow.bike_type}}<br>` +
                                   `<b>骑行距离:</b> ${{flow.distance ? flow.distance.toFixed(0) : 'N/A'}} 米<br>` +
                                   `<b>骑行时间:</b> ${{flow.duration ? flow.duration.toFixed(1) : 'N/A'}} 分钟<br>` +
                                   `<b>起点:</b> (${{flow.start[1].toFixed(4)}}, ${{flow.start[0].toFixed(4)}})<br>` +
                                   `<b>终点:</b> (${{flow.end[1].toFixed(4)}}, ${{flow.end[0].toFixed(4)}})`;
                
                polyline.bindPopup(popupContent);
                polyline.addTo(layerGroup);
                
                var startMarker = L.circleMarker(flow.start, {{
                    radius: 3,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.8,
                    weight: 1
                }});
                startMarker.addTo(layerGroup);
                
                var endMarker = L.circleMarker(flow.end, {{
                    radius: 3,
                    color: '#333',
                    fillColor: '#333',
                    fillOpacity: 0.8,
                    weight: 1
                }});
                endMarker.addTo(layerGroup);
            }});
            
            return layerGroup;
        }}
        
        singleLayerGroup = createFlowLines(singleData, '#2ecc71');
        ebikeLayerGroup = createFlowLines(ebikeData, '#3498db');
        
        if (singleLayerVisible) singleLayerGroup.addTo(map);
        if (ebikeLayerVisible) ebikeLayerGroup.addTo(map);
        
        var overlayMaps = {{
            "🟢 单车轨迹": singleLayerGroup,
            "🔵 助力车轨迹": ebikeLayerGroup
        }};
        
        L.control.layers(baseMaps, overlayMaps, {{collapsed: false}}).addTo(map);
        
        function toggleLayer(type) {{
            if (type === 'single') {{
                singleLayerVisible = !singleLayerVisible;
                if (singleLayerVisible) {{
                    singleLayerGroup.addTo(map);
                    document.getElementById('btnSingle').innerHTML = '✅ 显示/隐藏 单车';
                    document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
                }} else {{
                    map.removeLayer(singleLayerGroup);
                    document.getElementById('btnSingle').innerHTML = '❌ 显示/隐藏 单车';
                    document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }} else {{
                ebikeLayerVisible = !ebikeLayerVisible;
                if (ebikeLayerVisible) {{
                    ebikeLayerGroup.addTo(map);
                    document.getElementById('btnEbike').innerHTML = '✅ 显示/隐藏 助力车';
                    document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
                }} else {{
                    map.removeLayer(ebikeLayerGroup);
                    document.getElementById('btnEbike').innerHTML = '❌ 显示/隐藏 助力车';
                    document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }}
        }}
        
        function showAll() {{
            singleLayerVisible = true;
            ebikeLayerVisible = true;
            if (!map.hasLayer(singleLayerGroup)) singleLayerGroup.addTo(map);
            if (!map.hasLayer(ebikeLayerGroup)) ebikeLayerGroup.addTo(map);
            
            document.getElementById('btnSingle').innerHTML = '✅ 显示/隐藏 单车';
            document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
            document.getElementById('btnEbike').innerHTML = '✅ 显示/隐藏 助力车';
            document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
        }}
        
        function hideAll() {{
            singleLayerVisible = false;
            ebikeLayerVisible = false;
            if (map.hasLayer(singleLayerGroup)) map.removeLayer(singleLayerGroup);
            if (map.hasLayer(ebikeLayerGroup)) map.removeLayer(ebikeLayerGroup);
            
            document.getElementById('btnSingle').innerHTML = '❌ 显示/隐藏 单车';
            document.getElementById('btnSingle').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
            document.getElementById('btnEbike').innerHTML = '❌ 显示/隐藏 助力车';
            document.getElementById('btnEbike').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
        }}
        
        L.control.scale({{imperial: false, metric: true}}).addTo(map);
    </script>
</body>
</html>'''
    
    html_path = os.path.join(OUTPUT_DIR, 'flow_map.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 流动轨迹地图已保存到: {html_path}")
    print("请在浏览器中打开该文件查看交互式地图")

if __name__ == "__main__":
    load_and_process_data()
    create_html_map()
