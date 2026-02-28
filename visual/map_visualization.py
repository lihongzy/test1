import pandas as pd
import numpy as np
import json
import os

DATA_PATH = r"d:\postgraduate\20260207\workplace\data\[张永平]XiaMen2024-共享单车、电单车.csv"
OUTPUT_DIR = r"d:\postgraduate\20260207\workplace\visual"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "map_data.json")

def load_and_process_data():
    print("正在加载数据...")
    df = pd.read_csv(DATA_PATH)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    grid_size = 0.005
    
    df_start = df[['start_lat', 'start_lng']].dropna()
    df_end = df[['end_lat', 'end_lng']].dropna()
    
    print("正在计算起点网格...")
    df_start['lat_grid'] = (df_start['start_lat'] // grid_size) * grid_size
    df_start['lng_grid'] = (df_start['start_lng'] // grid_size) * grid_size
    start_agg = df_start.groupby(['lat_grid', 'lng_grid']).size().reset_index(name='count')
    start_agg['lat'] = start_agg['lat_grid'] + grid_size / 2
    start_agg['lng'] = start_agg['lng_grid'] + grid_size / 2
    start_data = start_agg[['lat', 'lng', 'count']].values.tolist()
    
    print("正在计算终点网格...")
    df_end['lat_grid'] = (df_end['end_lat'] // grid_size) * grid_size
    df_end['lng_grid'] = (df_end['end_lng'] // grid_size) * grid_size
    end_agg = df_end.groupby(['lat_grid', 'lng_grid']).size().reset_index(name='count')
    end_agg['lat'] = end_agg['lat_grid'] + grid_size / 2
    end_agg['lng'] = end_agg['lng_grid'] + grid_size / 2
    end_data = end_agg[['lat', 'lng', 'count']].values.tolist()
    
    center_lat = (df['start_lat'].mean() + df['end_lat'].mean()) / 2
    center_lng = (df['start_lng'].mean() + df['end_lng'].mean()) / 2
    
    result = {
        'center': [center_lat, center_lng],
        'start_data': start_data,
        'end_data': end_data,
        'total_rides': len(df),
        'start_count': len(df_start),
        'end_count': len(df_end)
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"起点网格数: {len(start_data)}")
    print(f"终点网格数: {len(end_data)}")
    print(f"数据已保存到: {OUTPUT_JSON}")
    return result

def create_html_map():
    print("正在生成HTML地图...")
    
    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>厦门共享单车骑行位置分布图</title>
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
            min-width: 180px;
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
        
        .btn-start {{ 
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }}
        
        .btn-end {{ 
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }}
        
        .btn-all-show {{ 
            background: linear-gradient(135deg, #3498db, #2980b9);
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
            min-width: 180px;
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
        
        .legend-color {{
            width: 30px;
            height: 15px;
            margin-right: 12px;
            border-radius: 3px;
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
    </style>
</head>
<body>
    <div class="title">
        <h1>🚲 厦门共享单车骑行位置分布图</h1>
        <p>基于GPS轨迹数据的起点终点热力图分析</p>
    </div>
    
    <div class="stats">
        <div class="stats-item">
            <span class="stats-value">{data['total_rides']:,}</span> 条骑行记录
        </div>
        <div class="stats-item">
            <span class="stats-value">{data['start_count']:,}</span> 个起点位置
        </div>
        <div class="stats-item">
            <span class="stats-value">{data['end_count']:,}</span> 个终点位置
        </div>
    </div>
    
    <div class="control-panel">
        <h3>🗂️ 图层控制</h3>
        <button class="btn btn-start" id="btnStart" onclick="toggleLayer('start')">✅ 显示/隐藏 起点</button>
        <button class="btn btn-end" id="btnEnd" onclick="toggleLayer('end')">✅ 显示/隐藏 终点</button>
        <button class="btn btn-all-show" onclick="showAll()">👁️ 显示全部</button>
        <button class="btn btn-all-hide" onclick="hideAll()">🔒 隐藏全部</button>
    </div>
    
    <div class="legend">
        <h4>📖 图例说明</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);"></div>
            <span class="legend-text">起点 (低→高密度)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(90deg, #3498db, #9b59b6, #e74c3c);"></div>
            <span class="legend-text">终点 (低→高密度)</span>
        </div>
    </div>
    
    <div id="map"></div>
    
    <script>
        var startLayerVisible = true;
        var endLayerVisible = true;
        var startHeatLayer = null;
        var endHeatLayer = null;
        
        var startData = {json.dumps(data['start_data'])};
        var endData = {json.dumps(data['end_data'])};
        
        var map = L.map('map', {{
            center: {json.dumps(data['center'])},
            zoom: 12,
            layers: []
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
        
        function getHeatOptions(data, color) {{
            return {{
                minOpacity: 0.4,
                maxZoom: 18,
                radius: 30,
                blur: 25,
                gradient: color === 'start' 
                    ? {{0.2: '#2ecc71', 0.4: '#f1c40f', 0.6: '#e67e22', 0.8: '#e74c3c', 1: '#c0392b'}}
                    : {{0.2: '#3498db', 0.4: '#9b59b6', 0.6: '#e74c3c', 0.8: '#c0392b', 1: '#8e44ad'}}
            }};
        }}
        
        function createHeatLayer(data, color) {{
            return L.heatLayer(data, getHeatOptions(data, color));
        }}
        
        startHeatLayer = createHeatLayer(startData, 'start');
        endHeatLayer = createHeatLayer(endData, 'end');
        
        if (startLayerVisible) startHeatLayer.addTo(map);
        if (endLayerVisible) endHeatLayer.addTo(map);
        
        var overlayMaps = {{
            "🟢 起点热力图": startHeatLayer,
            "🔴 终点热力图": endHeatLayer
        }};
        
        L.control.layers(baseMaps, overlayMaps, {{collapsed: false}}).addTo(map);
        
        function toggleLayer(type) {{
            if (type === 'start') {{
                startLayerVisible = !startLayerVisible;
                if (startLayerVisible) {{
                    startHeatLayer.addTo(map);
                    document.getElementById('btnStart').innerHTML = '✅ 显示/隐藏 起点';
                    document.getElementById('btnStart').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
                }} else {{
                    map.removeLayer(startHeatLayer);
                    document.getElementById('btnStart').innerHTML = '❌ 显示/隐藏 起点';
                    document.getElementById('btnStart').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }} else {{
                endLayerVisible = !endLayerVisible;
                if (endLayerVisible) {{
                    endHeatLayer.addTo(map);
                    document.getElementById('btnEnd').innerHTML = '✅ 显示/隐藏 终点';
                    document.getElementById('btnEnd').style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
                }} else {{
                    map.removeLayer(endHeatLayer);
                    document.getElementById('btnEnd').innerHTML = '❌ 显示/隐藏 终点';
                    document.getElementById('btnEnd').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
                }}
            }}
        }}
        
        function showAll() {{
            startLayerVisible = true;
            endLayerVisible = true;
            if (!map.hasLayer(startHeatLayer)) startHeatLayer.addTo(map);
            if (!map.hasLayer(endHeatLayer)) endHeatLayer.addTo(map);
            
            document.getElementById('btnStart').innerHTML = '✅ 显示/隐藏 起点';
            document.getElementById('btnStart').style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
            document.getElementById('btnEnd').innerHTML = '✅ 显示/隐藏 终点';
            document.getElementById('btnEnd').style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
        }}
        
        function hideAll() {{
            startLayerVisible = false;
            endLayerVisible = false;
            if (map.hasLayer(startHeatLayer)) map.removeLayer(startHeatLayer);
            if (map.hasLayer(endHeatLayer)) map.removeLayer(endHeatLayer);
            
            document.getElementById('btnStart').innerHTML = '❌ 显示/隐藏 起点';
            document.getElementById('btnStart').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
            document.getElementById('btnEnd').innerHTML = '❌ 显示/隐藏 终点';
            document.getElementById('btnEnd').style.background = 'linear-gradient(135deg, #95a5a6, #7f8c8d)';
        }}
        
        L.control.scale({{imperial: false, metric: true}}).addTo(map);
    </script>
</body>
</html>'''
    
    html_path = os.path.join(OUTPUT_DIR, 'bike_sharing_map.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 地图HTML已保存到: {html_path}")
    print("请在浏览器中打开该文件查看交互式地图")

if __name__ == "__main__":
    load_and_process_data()
    create_html_map()
