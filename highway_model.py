
import pandas as pd
import geopandas as gpd
import transbigdata as tbd
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

def reconstruct_highway(highway_centerline,station,add_link = True):
    '''
    提取简化高速公路边信息并构建相关网络

    输入:
    highway_centerline (GeoDataFrame): 包含高速公路中心线的地理空间数据框（GeoDataFrame），其中应至少包含以下列：
        - 'geometry': 包含道路中心线几何形状的列
    station (GeoDataFrame): 包含收费站信息的地理空间数据框（GeoDataFrame），其中应至少包含以下列：
        - 'geometry': 包含收费站位置几何形状的列
        - 'id': 收费站的唯一标识

    输出:
    edge (GeoDataFrame): 包含生成的道路网络边的地理空间数据框（GeoDataFrame），其中包括以下列：
        - 'geometry': 包含道路边界几何形状的列
        - 'edge_id': 道路边的唯一标识
        - 'attr': 道路属性（默认为 'road'）

    node (DataFrame): 包含生成的道路网络节点的数据框（DataFrame），其中包括以下列：
        - 'lon': 节点经度
        - 'lat': 节点纬度
        - 'geometry': 包含节点位置几何形状的列
        - 'id': 节点的唯一标识

    station2node (DataFrame): 包含收费站节点与网络节点的对应关系的数据框（DataFrame），其中包括以下列：
        - 'station_id': 收费站的唯一标识
        - 'node_id': 对应的网络节点的唯一标识

    步骤:
    1. 提取简化高速公路边信息，构建道路边数据框 edge。
    2. 生成反向边信息，构建道路反向边数据框 edge_inverse。
    3. 合并正反向边信息，得到完整的道路边数据框 edge。
    4. 进行道路匹配，生成道路网络。
    5. 构建节点表 node，包括道路网络的起始节点和终点。
    6. 生成收费站节点对应网络节点的编号 station2node。
    7. 为道路边添加起始点和终点节点信息，计算边的长度。
    8. 返回道路边、节点、和收费站节点对应关系的数据框。
    '''

    def nearest(station,edges):
        # 为每个点找到最近的边
        results = []
        for i, point in station.iterrows():
            min_dist = float("inf")
            nearest_edge_info = None
            
            for j, edge in edges.iterrows():
                dist = point['geometry'].distance(edge['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_edge_info = {
                        'lon1': point['lon1'],
                        'lat1': point['lat1'],
                        'geometry_x': point['geometry'],
                        'dist': min_dist,
                        'index': j,
                        'lon': edge['lon'],
                        'lat': edge['lat'],
                        'geometry_y': edge['geometry']
                    }
            
            results.append(nearest_edge_info)

        # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)
        return result_df

    def get_matched_edge(edge,station):
        '''
        将收费站匹配到最近的道路边并调整网络

        输入:
        edge (GeoDataFrame): 包含道路边界的地理空间数据框（GeoDataFrame），其中应至少包含以下列：
            - 'geometry': 包含道路边界几何形状的列
            - 'edge_id': 道路边的唯一标识

        station (GeoDataFrame): 包含收费站信息的地理空间数据框（GeoDataFrame），其中应至少包含以下列：
            - 'geometry': 包含收费站位置几何形状的列
            - 'id': 收费站的唯一标识

        输出:
        edge (GeoDataFrame): 更新后的道路边界的地理空间数据框（GeoDataFrame），包含匹配到的收费站和额外的节点信息。
        
        matched_table_2 (DataFrame): 包含未匹配到的收费站信息的数据框（DataFrame），其中包括以下列：
            - 'edge_id': 道路边的唯一标识
            - 'id': 未匹配到的收费站的唯一标识
        '''
        # 投影、偏移、匹配至最近的边
        dist = 50
        # 偏移100米的边
        edge_offset = edge.copy()
        edge_offset.crs = 'EPSG:4326'
        edge_offset_proj = edge_offset.to_crs('EPSG:4525') #CGCS2000 / 3-degree Gauss Kruger zone 37
        edge_offset_proj['geometry'] = edge_offset_proj['geometry'].apply(lambda x:x.parallel_offset(dist))
        edge_offset_proj = edge_offset_proj.reset_index(drop=True)
        edge_offset_proj = edge_offset_proj[-edge_offset_proj['geometry'].is_empty]

        station.crs = 'EPSG:4326'
        station_proj = station.to_crs('EPSG:4525')
        station_proj_match = nearest(station_proj,edge_offset_proj[-edge_offset_proj['geometry'].is_empty][['edge_id','geometry']])

        # 对于只匹配到一个点的边
        matched_table_1 = station_proj_match.drop_duplicates(subset=['edge_id'],keep='first')[['edge_id','id']]
        matched_table_2 = station_proj_match[-station_proj_match['id'].isin(matched_table_1['id'])]

        # 基于额外节点调整网络
        for i in range(len(matched_table_1)):
            r = matched_table_1.iloc[i]
            nodeid = r['id']
            edgeid = r['edge_id']
            nodegeometry = station[station['id']==nodeid]['geometry'].iloc[0]
            edgegeometry = edge[edge['edge_id']==edgeid]['geometry'].iloc[0]
            #找到edge上距离node最近的点
            projectdist = edgegeometry.project(nodegeometry)
            projectpoint = edgegeometry.interpolate(projectdist)
            #如果在两端，则不需要切分
            #增加端点到节点的线段
            add_edges = []
            add_edges_attr = []
            if (projectdist==edgegeometry.length)|(projectdist==0):
                if add_link:
                    line3 = LineString([nodegeometry,projectpoint])
                    line4 = LineString([projectpoint,nodegeometry])
                    add_edges.append(line3)
                    add_edges_attr.append('link')
                    add_edges.append(line4)
                    add_edges_attr.append('link')
            else: #如果在中间，则需要切分，将原始边切分为两段，再加上端点到节点的线段
                from shapely.geometry import Point
                edge_coords = pd.DataFrame(edgegeometry.coords)
                edge_coords['proj'] = edge_coords.apply(lambda r:edgegeometry.project(Point([r[0],r[1]])),axis = 1)

                # 由中间端点切分边为两段
                line1 = LineString(edge_coords[edge_coords['proj']<projectdist][[0,1]].apply(lambda r:Point([r[0],r[1]]),axis = 1).tolist()+[projectpoint])
                add_edges.append(line1)
                add_edges_attr.append('road')
                line2 = LineString([projectpoint] + edge_coords[edge_coords['proj']>projectdist][[0,1]].apply(lambda r:Point([r[0],r[1]]),axis = 1).tolist())
                add_edges.append(line2)
                add_edges_attr.append('road')
                if add_link:
                    # 添加中间端点到收费站的线段
                    line3 = LineString([nodegeometry,projectpoint])
                    line4 = LineString([projectpoint,nodegeometry])
                    add_edges.append(line3)
                    add_edges_attr.append('link')
                    add_edges.append(line4)
                    add_edges_attr.append('link')
                # 此时需要删除原有的边
                edge = edge[edge['edge_id']!=edgeid]
            # 将新的边加入
            add_edges = gpd.GeoDataFrame(geometry=add_edges)
            add_edges['attr'] = add_edges_attr
            edge = pd.concat([edge,add_edges])
        edge = edge[edge.length>0]
        edge['edge_id'] = range(len(edge))
        return edge,matched_table_2

    #提取简化高速公路边信息
    edge = highway_centerline[['geometry']]
    edge ['edge_id'] = range (len (edge))
    #生成反向边信息
    edge_inverse = edge.copy ()
    from shapely.geometry import LineString
    edge_inverse['geometry'] = edge_inverse['geometry'].apply(lambda x:LineString(list(x.coords)[::-1]))
    edge_inverse['edge_id'] +=len (edge)
    #合并正反向边信息
    edge = pd.concat([edge, edge_inverse])

    edge['attr'] = 'road'
    edge,matched_table_2 = get_matched_edge(edge,station)
    while len(matched_table_2)>0:
        edge,matched_table_2 = get_matched_edge(edge,station[station['id'].isin(matched_table_2['id'])])

    # 构建节点表
    edge['slon'] = edge['geometry'].apply(lambda r:r.coords[0][0])
    edge['slat'] = edge['geometry'].apply(lambda r:r.coords[0][1])
    edge['elon'] = edge['geometry'].apply(lambda r:r.coords[-1][0])
    edge['elat'] = edge['geometry'].apply(lambda r:r.coords[-1][1])

    # 提取简化高速公路节点信息
    node = pd.concat([edge[['slon','slat']].rename(columns = {'slon':'lon','slat':'lat'}),
                    edge[['elon','elat']].rename(columns = {'elon':'lon','elat':'lat'})]).drop_duplicates()
    node['geometry'] = gpd.points_from_xy(node['lon'],node['lat'])
    node['id'] = range(len(node)) 
    node = gpd.GeoDataFrame(node)
    # 生成收费站节点对应网络节点的编号
    station2node = tbd.ckdnearest_point(station,node)[['id_x','id_y']].rename(columns = {'id_x':'station_id','id_y':'node_id'})

    # 为边添加起终点接节点信息
    ## 添加起点信息
    node_tmp = node[['lon','lat','id']]
    node_tmp.columns = ['slon','slat','u']
    edge = pd.merge(edge,node_tmp,on = ['slon','slat'],how = 'left')
    ## 添加终点信息
    node_tmp = node[['lon','lat','id']]
    node_tmp.columns = ['elon','elat','v']
    edge = pd.merge(edge,node_tmp,on = ['elon','elat'],how = 'left')
    edge = gpd.GeoDataFrame(edge,geometry = 'geometry')
    edge.crs = 'EPSG:4326'
    edge['length'] = edge.to_crs('EPSG:4525').length
    edge = edge[edge['length']>0]
    
    return edge,node,station2node

# 高速公路边向右平移一定距离形成面
def generate_plane(edge, dist=100):
    '''
    高速公路边向右平移一定距离形成面

    输入:
    edge (GeoDataFrame): 包含高速公路边界的地理空间数据框（GeoDataFrame），其中应至少包含以下列：
        - 'geometry': 包含道路几何形状的列
        - 'crs': 坐标参考系统信息，应为 'EPSG:4326'。

    dist (float): 高速公路边界向右平移的距离，默认为 100 米。

    输出:
    edge_plane (GeoDataFrame): 包含生成的平移后道路面的地理空间数据框（GeoDataFrame），其中包括以下列：
        - 'geometry': 包含生成的道路面几何形状的列
        - 'crs': 坐标参考系统信息，为 'EPSG:4326'。

    步骤:
    1. 将输入的高速公路边界数据框的坐标参考系统转换为 'EPSG:4326'。
    2. 计算每个边界段的长度，并筛选出长度大于 50 米的边界段。
    3. 将筛选后的边界段数据框转换为 'EPSG:4525' 坐标系。
    4. 计算新的边界段长度。
    5. 生成单方向偏移，将每个边界段向右平移指定距离（默认为 100 米）。
    6. 将平移后的边界段数据框转换回 'EPSG:4326' 坐标系。
    7. 返回包含生成的道路面的地理空间数据框。
    '''
    # 转换为投影坐标系
    edge.crs = 'EPSG:4326'
    edge['length'] = edge.to_crs('EPSG:4525').length

    # 筛选较长的边界段
    edge = edge[edge['length'] > 50]
    edge_plane = edge.to_crs('EPSG:4525')

    # 重新计算长度
    edge['length'] = edge_plane['geometry'].length

    # 生成单方向偏移
    from shapely.geometry import Polygon
    edge_plane['geometry'] = edge_plane['geometry'].apply(lambda x: Polygon(list(x.coords) + list(x.parallel_offset(dist).coords)))
    edge_plane = edge_plane.to_crs('EPSG:4326')

    return edge_plane



def get_od_path(edge,node,station2node_dict,gdtollnode):
    """
    获取OD路径和长度信息的函数。

    输入参数：
    - edge: 包含路段信息的DataFrame，包括 'u'（起点节点ID）、'v'（终点节点ID）、'length'（路段长度）、'edge_id'（路段ID）等列。
    - node: 包含节点信息的DataFrame，至少包括 'id'（节点ID）列。
    - station2node_dict: 字典，将站点ID映射到节点ID的关系。
    - gdtollnode: 包含站点信息的DataFrame，至少包括 'id' 列。

    输出结果：
    - od_dis_table: 包含每个OD对应的路段信息的DataFrame，包括 'station_id_x'（起点站点ID）、'station_id_y'（终点站点ID）、'edge_id'（路段ID）、'cumsumlength'（经过路段的累计长度）等列。
    - od_length: 包含每个OD对应的路径总长度的DataFrame，包括 'station_id_x'（起点站点ID）、'station_id_y'（终点站点ID）、'length'（路径总长度）等列。

    注意事项：
    - 函数中使用了外部库NetworkX来计算最短路径，请确保已经正确安装并导入NetworkX库。
    - 输入的DataFrame需要按照函数要求包含相应的列名和数据。
    - 输出结果为包含路径和长度信息的DataFrame，可以根据需要进一步分析和处理。

    """
    # 获取最短路径的函数（已优化，使用缓存）
    path_cache = {}  # 缓存已计算的最短路径
    def get_shortest_path(start_station, end_station):
        if (start_station, end_station) not in path_cache:
            start_node = station2node_dict[start_station]
            end_node = station2node_dict[end_station]
            shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
            path_cache[(start_station, end_station)] = list(map(int, shortest_path))
        return path_cache[(start_station, end_station)]

    def get_path_dis_table(row):
        shortest_path = row['path']
        return [{'u': int(u), 'v': int(v), 'id': i,
                'station_id_x': row['station_id_x'], 'station_id_y': row['station_id_y']}
                for i, (u, v) in enumerate(zip(shortest_path[:-1], shortest_path[1:]))]

    G_edges = edge[['u','v','length']].values
    G_nodes = list(node['id'])

    #先创建一个有向图
    G = nx.DiGraph()
    #添加节点
    G.add_nodes_from(G_nodes) 
    #添加边
    G.add_weighted_edges_from(G_edges)

    # 创建OD表
    o = gdtollnode[['id']]
    o.columns = ['station_id']
    o['flag'] = 1
    d = o.copy()
    od = pd.merge(o, d, on='flag')[['station_id_x', 'station_id_y']]
    od = od[od['station_id_x'] != od['station_id_y']]


    # 获取OD的出行路径，使用缓存的路径数据
    print('获取OD的出行路径')
    od['path'] = od.parallel_apply(lambda r: get_shortest_path(r['station_id_x'], r['station_id_y']), axis=1)

    # 对OD编号
    od['odid'] = range(len(od))

    # 使用并行处理来提高效率
    print('整理OD的出行路径')
    od_details = od.parallel_apply(lambda r: get_path_dis_table(r), axis=1)

    # 合并得到的路径信息，避免多层循环
    od1_tmp = pd.DataFrame([item for sublist in od_details for item in sublist])

    # 合并边的信息
    od1_tmp = pd.merge(od1_tmp, edge[['u', 'v', 'length', 'edge_id']], on=['u', 'v'])

    # 转换为整型并排序
    od1_tmp['length'] = od1_tmp['length'].astype(int)
    od1_tmp = od1_tmp.sort_values(by=['station_id_x', 'station_id_y', 'id'])

    # 计算累计长度
    od1_tmp['cumsumlength'] = od1_tmp.groupby(['station_id_x', 'station_id_y'])['length'].cumsum()

    od_dis_table = od1_tmp[['station_id_x','station_id_y','edge_id','cumsumlength']]

    # od_dis_table存储了每个OD对应的路段信息
    # 其中，station_id_x、station_id_y为OD的起点、终点
    # edge_id为经过路段的id，cumsumlength为经过路段的累计长度
    # 例如，station_id_x=0，station_id_y=1，edge_id=47784，cumsumlength=68.596574，表示OD为0-1的出行路径，需要经过id为47784的路段，路径走完这一路段时，所经过的长度为68.596574米
    od_length = od_dis_table.groupby(['station_id_x','station_id_y'])['cumsumlength'].max().rename('length').reset_index()

    return od_dis_table,od_length

def od_merge_table_reconstruct(od_dis_table):
    """
    重新构造OD合并表格的函数。

    输入参数：
    - od_dis_table: 包含每个OD对应的路段信息的DataFrame，至少包括 'station_id_x'（起点站点ID）、'station_id_y'（终点站点ID）、'edge_id'（路段ID）、'cumsumlength'（经过路段的累计长度）等列。

    输出结果：
    - od_merge_table: 重新构造的OD合并表格的DataFrame，包括 'station_id_x'（起点站点ID）、'station_id_y'（终点站点ID）、'edge_id'（路段ID列表）、'cumsumlength'（经过路段的累计长度差值列表）、'cumsumlength2'（经过路段的原始累计长度列表）等列。

    注意事项：
    - 输入的DataFrame需要按照函数要求包含相应的列名和数据。
    - 函数使用了并行处理来计算累计长度的差值，提高了计算效率。
    - 输出结果为重新构造的OD合并表格，包含了路段ID列表和累计长度差值列表，可以根据需要进一步分析和处理。

    """
    # 合并列表
    od_merge_table = od_dis_table.groupby(['station_id_x', 'station_id_y']).agg(
        {'edge_id': list, 'cumsumlength': list}).reset_index()
    def calculate_differences(lst):
        differences = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
        return [lst[0]] + differences
    od_merge_table['cumsumlength2'] = od_merge_table['cumsumlength']
    od_merge_table['cumsumlength'] = od_merge_table['cumsumlength'].parallel_apply(calculate_differences)
    return od_merge_table



def get_edge_to_charge(edge,node,charge_station):
    """
    查找最近充电站点信息的函数。

    输入参数：
    - edge: 包含路段信息的DataFrame，至少包括 'u'（起点节点ID）、'v'（终点节点ID）、'length'（路段长度）、'edge_id'（路段ID）等列。
    - node: 包含节点信息的DataFrame，至少包括 'id'（节点ID）列。
    - charge_station: 包含充电站点信息的DataFrame，至少包括 '场站名称'、'充电车位数'、'node_id' 等列。

    输出结果：
    - edge_to_charge: 包含路段到最近充电站点的关联信息的DataFrame，包括 'edge_id'（路段ID）、'charge_node_id'（最近充电站点ID）、'charge_node_distance'（距离最近充电站点的距离）、'场站名称'、'充电车位数'等列。

    注意事项：
    - 输入的DataFrame需要按照函数要求包含相应的列名和数据。
    - 函数使用了NetworkX库来计算最近充电站点，需要确保正确安装并导入NetworkX库。
    - 输出结果为包含路段到最近充电站点的关联信息的DataFrame，可以根据需要进一步分析和处理。

    """
    # find the nearest charge node
    def get_nearest_charge_station(G,node_id, radius = 1000):
        # create subgraph
        subgraph = nx.ego_graph (G, node_id, radius=radius,distance = 'weight')
        distances = []
        stationid = None
        stationdistance = None
        for node in subgraph.nodes() :
            if (node in list(charge_station ['node_id'])):# if it is charge station
                distance = nx.shortest_path_length(subgraph,source=node_id, target=node,weight='weight')
                distances.append ( [node, distance])
        distances = pd.DataFrame (distances, columns=['stationid', 'distance']) .sort_values (by='distance')
        # if there is a charge staion
        if len (distances)>0:
            stationid = int (distances ['stationid'].iloc [0])
            stationdistance = int (distances ['distance'].iloc[0])
        return stationid, stationdistance

    # find the nearest charge node
    # 构建网络
    G_edges = edge[['u','v','length']].values
    G_nodes = list(node['id'])
    import networkx as nx
    #先创建一个有向图
    G = nx.DiGraph()
    #添加节点
    G.add_nodes_from(G_nodes) 
    #添加边
    G.add_weighted_edges_from(G_edges)

    G_nodes_df = pd. DataFrame (G_nodes,columns = ['node_id'])
    a = G_nodes_df['node_id'].parallel_apply(lambda node_id:get_nearest_charge_station(G,node_id,radius = 1000))
    G_nodes_df['charge_node_id'] = a.apply(lambda r:r [0])
    G_nodes_df['charge_node_distance'] = a.apply (lambda r:r [1])

    edge_to_charge = pd.merge(edge[['edge_id', 'v']].rename(columns={'v': 'node_id'}),
                            G_nodes_df)[['edge_id', 'charge_node_id', 'charge_node_distance']]
    edge_to_charge = pd.merge(edge_to_charge,
                            charge_station[['场站名称', '充电车位数', "node_id"]].rename(
                                columns={'node_id': 'charge_node_id'})).sort_values(by='edge_id')
    return edge_to_charge
