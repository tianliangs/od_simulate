import pandas as pd
import geopandas as gpd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def highway_chargenum(province,stations_vis):
    stations= pd.read_csv(f'中间数据/network/{province}/stations.csv')
    stations_limitation = stations[['lon','lat','limatation']]
    stations_vis=pd.read_csv(stations_vis)
    stations_vis["hour"]=stations_vis["time"].dt.hour
    stations_vis['car_count'] = stations_vis['current_car_count'] + stations_vis['waiting_car_count']
    stations_vis = pd.merge(stations_vis,stations_limitation,on=['lon','lat'],how='left')
    station=stations_vis.groupby(["charge_node_id"])["car_count"].quantile(0.84).reset_index()
    station=station.sort_values(by="car_count",ascending=False)
    station["chargenum"] = min(station["chargenum"],station["limatation"])
    station["chargenum"] = station["chargenum"].apply(lambda x:2 if x<=2 else x)

    station2node = pd.read_csv(f'中间数据/network/{province}/{province}收费站节点对应网络节点的编号.csv')
    charge_station = stations[stations['flag']!='收费站'].rename(columns={'id':'station_id'})
    station_result = station[["charge_node_id","chargenum"]]
    result = pd.merge(charge_station, station2node.rename(columns={'node_id': 'charge_node_id'}), on=['station_id']).drop_duplicates(subset=['charge_node_id'])
    result = result[["charge_node_id","场站名称","充电车位数",'pname','lon','lat']]
    result = result[result['pname'] == province]
    result = pd.merge(result, station_result, on=['charge_node_id'], how='left')

    result.to_csv('station_decesion_result/station_decesion_result.csv',index=False)
    return 'success'

