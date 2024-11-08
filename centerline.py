import pandas as pd
import shapely
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString


def get_centerline(road,interpolation_distance,minlength,simplify_tolerance = 0.0001,densify_interval = 0.0001):
    '''
    根据提供的道路数据生成中心线。该函数执行以下步骤：
    1. 对道路数据进行缓冲区分析。
    2. 计算缓冲区的泰森多边形（Voronoi图）。
    3. 从泰森多边形中提取位于道路缓冲区内的中心线段。
    4. 合并这些线段，并清除不符合条件的死路。
    5. 对结果中的中心线进行简化。

    参数:
    - road (Shapely.geometry): 道路数据的几何对象。
    - interpolation_distance (float): 缓冲区分析和细分顶点时的距离参数。
    - minlength (float): 被视为死路的最小长度阈值，短于此长度的孤立线段将被清除。
    - simplify_tolerance (float): 几何简化时的公差值，默认为0.0001。

    返回:
    - centerline_merged (GeoDataFrame): 包含合并后的中心线的GeoDataFrame，坐标系与输入的道路数据相同。

    注意:
    - 输入的道路数据应为Shapely的几何对象，如LineString或MultiLineString。
    - 输出的GeoDataFrame的每个元素都是一个中心线的几何对象。
    - 函数内部会打印出处理过程中的中间道路数量，以便跟踪进度。
    - 该函数依赖于shapely, geopandas, pandas, networkx等Python库。

    示例:
    ```python
    from shapely.geometry import LineString
    import geopandas as gpd

    # 示例道路的LineString
    road_geo = LineString([(0, 0), (1, 1), (2, 2)])

    # 计算中心线
    centerline = get_centerline(road_geo, interpolation_distance=5, minlength=10)

    # 结果是一个GeoDataFrame，可以直接用于进一步的空间分析或绘图
    '''

    def densify_line(line, interval):
        """
        增密每个LineString对象，按指定的间隔在其上插入新的节点。
        :param line: shapely.geometry.LineString 对象
        :param interval: 节点间隔
        :return: 增密后的LineString对象
        """
        line_length = line.length
        num_points = int(line_length / interval)
        points = [line.interpolate(i * interval) for i in range(num_points + 1)]
        return LineString(points)

    def get_start_end_point(centerline_merged):
        '''
        提取道路中心线的起点和终点坐标，并为每条中心线增加起终点的x和y坐标字段。

        输入:
        - centerline_merged (GeoDataFrame): 包含道路中心线几何数据的GeoDataFrame。

        输出:
        - centerline_merged (GeoDataFrame): 经过处理，新增了起点和终点坐标字段的GeoDataFrame。

        功能说明:
        1. 对每条道路中心线，提取起点和终点的坐标。
        2. 将提取的坐标分别作为新的字段（'sx', 'sy'表示起点，'ex', 'ey'表示终点）添加到输入的GeoDataFrame中。
        3. 剔除起点和终点坐标相同的道路中心线，这些通常是闭合环的表示。
        '''
        centerline_merged = centerline_merged.copy()
        #判断起始点是否连通多个路径
        sp = centerline_merged['geometry'].apply(lambda linestring:linestring.coords[0])
        ep = centerline_merged['geometry'].apply(lambda linestring:linestring.coords[-1])
        centerline_merged['sx'] = sp.apply(lambda x:x[0])
        centerline_merged['sy'] = sp.apply(lambda x:x[1])
        centerline_merged['ex'] = ep.apply(lambda x:x[0])
        centerline_merged['ey'] = ep.apply(lambda x:x[1])
        centerline_merged = centerline_merged[~((centerline_merged['sx']==centerline_merged['ex'])&(centerline_merged['sy']==centerline_merged['ey']))]
        return centerline_merged

    def clean_endlane(centerline_merged,minlength):
        '''
        清除孤立且长度不足的道路末端，以维护道路网络的连通性。此函数的目标是剔除那些只连接一个节点且长度小于指定阈值的道路末端。

        输入:
        - centerline_merged (GeoDataFrame): 含有几何对象的GeoDataFrame，表示待处理的道路中心线。
        - minlength (float): 用于判定道路末端是否足够长而应被保留的长度阈值。

        输出:
        - centerline_merged (GeoDataFrame): 清除了孤立且短的道路末端后的中心线GeoDataFrame。
        '''
        #判断起始点是否连通多个路径
        centerline_merged = get_start_end_point(centerline_merged)
        p1 = centerline_merged[['sx','sy']]
        p1.columns = ['x','y']
        p2 = centerline_merged[['ex','ey']]
        p2.columns = ['x','y']
        p_count = pd.concat([p1,p2]).groupby(['x','y']).size().rename('count').reset_index()
        p_count.columns = ['sx','sy','s_count']
        centerline_merged = pd.merge(centerline_merged,p_count)
        p_count.columns = ['ex','ey','e_count']
        centerline_merged = pd.merge(centerline_merged,p_count)

        # 剔除不承担连通作用且短的断头路
        centerline_merged = centerline_merged[~(((centerline_merged['s_count']==1)|(centerline_merged['e_count']==1))&(centerline_merged.length<=minlength))]
        centerline_merged = shapely.line_merge(MultiLineString(list(centerline_merged['geometry'])))
        centerline_merged = gpd.GeoDataFrame(geometry=list(centerline_merged.geoms))
        return centerline_merged

    def clean_duplicate_paths(centerline_merged,minlength):
        '''
        清除重复的路径段，以生成更加简洁的中心线。此函数的目标是识别并移除长度低于特定阈值的重复线段，
        保留较长的线段，并合并其余的路径以形成连续的中心线。

        输入:
        - centerline_merged (GeoDataFrame): 含有几何对象的GeoDataFrame，表示当前所有中心线段。
        - minlength (float): 用于识别短路径的长度阈值，小于或等于此长度的重复路径会被移除。

        输出:
        - centerline_merged (GeoDataFrame): 清除了重复路径之后的中心线GeoDataFrame。
        '''
        # 起终点剔除重复路径
        centerline_merged = centerline_merged.copy()
        centerline_merged['length'] = centerline_merged.length
        centerline_merged_short = centerline_merged[centerline_merged['length']<=minlength]
        centerline_merged_long = centerline_merged[centerline_merged['length']>minlength]
        centerline_merged_short = get_start_end_point(centerline_merged_short).sort_values(by = 'length').drop_duplicates(subset= ['sx','sy','ex','ey'],keep = 'first')[['geometry','length']]
        centerline_merged = pd.concat([centerline_merged_short,centerline_merged_long])
        centerline_merged = shapely.line_merge(MultiLineString(list(centerline_merged['geometry'])))
        centerline_merged = gpd.GeoDataFrame(geometry=list(centerline_merged.geoms))
        return centerline_merged


    def get_spanning_tree(centerline_merged,minlength):
        '''
        构建道路中心线的最小生成树，忽略长度小于等于指定阈值的边。

        输入:
        - centerline_merged (GeoDataFrame): 包含道路中心线几何数据的GeoDataFrame。
        - minlength (float): 长度阈值，小于等于此阈值的边将不会被包含在生成的最小生成树中。

        输出:
        - centerline_merged (GeoDataFrame): 构建完成的最小生成树，其中包含了所有满足长度条件的中心线。

        功能说明:
        1. 提取输入中心线的起始点和终点坐标，构成边的集合。
        2. 构建节点集合，并为每个节点分配唯一ID。
        3. 创建边的集合，并为边的起始和终点匹配节点ID。
        4. 根据边的长度，分离出短边和长边。
        5. 使用短边集合构建一个无向图，并计算最小生成树。
        6. 将最小生成树中的边和长边集合合并，以形成完整的道路网络。
        7. 将合并后的边集合中的几何数据合并为一个GeoDataFrame。

        注释:
        - 输入的GeoDataFrame应保证每个元素为LineString几何类型。
        - 使用NetworkX库构建无向图和最小生成树。
        - 函数返回的是一个包含了最小生成树中所有边的GeoDataFrame。
        '''
        # 获取道路中心线的起终点坐标
        edge = get_start_end_point(centerline_merged)

        # 提取并去重道路节点信息，为后续建图准备
        node = pd.concat([edge[['sx','sy']].rename(columns={'sx':'x','sy':'y'}),
                        edge[['ex','ey']].rename(columns={'ex':'x','ey':'y'})]).drop_duplicates()
        # 为节点生成几何信息，并分配唯一ID
        node['geometry'] = gpd.points_from_xy(node['x'], node['y'])
        node['id'] = range(len(node))

        # 为边添加起终点对应的节点ID
        ## 添加起点ID
        node_tmp = node[['x', 'y', 'id']]
        node_tmp.columns = ['sx', 'sy', 'u']
        edge = pd.merge(edge, node_tmp, on=['sx', 'sy'], how='left')
        ## 添加终点ID
        node_tmp = node[['x', 'y', 'id']]
        node_tmp.columns = ['ex', 'ey', 'v']
        edge = pd.merge(edge, node_tmp, on=['ex', 'ey'], how='left')
        edge = gpd.GeoDataFrame(edge, geometry='geometry')

        # 分离出短边和长边
        edge['length'] = edge.length
        edge_short = edge[edge['length'] <= minlength]
        edge_long = edge[edge['length'] > minlength]

        # 准备用于构建图的边集合
        G_edges = edge_short[['u', 'v', 'length']].values
        G_nodes = list(node['id'])

        # 使用NetworkX构建无向图
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(G_nodes)  # 添加节点
        G.add_weighted_edges_from(G_edges)  # 添加边

        # 计算最小生成树
        T = nx.minimum_spanning_tree(G)

        # 获取生成树的边，并与原有的长边集合合并
        T_edges1 = pd.DataFrame(list(T.edges), columns=['u', 'v'])
        T_edges2 = pd.DataFrame(list(T.edges), columns=['v', 'u'])
        T_edges = pd.concat([T_edges1, T_edges2])
        edge_short = pd.merge(edge_short, T_edges)
        edges = pd.concat([edge_short, edge_long])
        edges = edges[['geometry']]

        # 合并边集合中的几何信息
        centerline_merged = shapely.line_merge(MultiLineString(list(edges['geometry'])))
        centerline_merged = gpd.GeoDataFrame(geometry=list(centerline_merged.geoms)).reset_index()
        
        return centerline_merged
    # 道路做buffer
    road_polygon = road.buffer(interpolation_distance).unary_union

    # 提取泰森多边形顶点
    borders = road_polygon.segmentize(interpolation_distance * 0.5)  # 节点更加密集
    voronoied = shapely.voronoi_polygons(borders, only_edges=True)  # 生成泰森多边形

    # 保留在多边形内的边
    centerlines = gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms))\
        .sjoin(gpd.GeoDataFrame(geometry=gpd.GeoSeries(road_polygon)), predicate="within")

    # 合并提取的中心线
    centerline_merged = shapely.line_merge(centerlines.unary_union)
    centerline_merged = gpd.GeoDataFrame(geometry=list(centerline_merged.geoms)).reset_index()
    roadnum = len(centerline_merged)
    print('初步提取后路段数：', roadnum)

    # 循环处理，直至新旧道路数目一致
    while True:
        centerline_merged = clean_endlane(centerline_merged, minlength)
        centerline_merged = get_spanning_tree(centerline_merged, minlength)
        centerline_merged = clean_duplicate_paths(centerline_merged, minlength)
        newroadnum = len(centerline_merged)
        print('修整后路段数：', newroadnum)
        if newroadnum == roadnum:
            break
        else:
            roadnum = newroadnum

    # 再次简化几何线型
    centerline_merged['geometry'] = centerline_merged['geometry'].simplify(simplify_tolerance)
    centerline_merged.crs = road.crs

    # 增密中心线（如果指定了 densify_interval）
    if densify_interval:
        centerline_merged['geometry'] = centerline_merged['geometry'].apply(lambda line: densify_line(line, densify_interval))
    
    return centerline_merged

