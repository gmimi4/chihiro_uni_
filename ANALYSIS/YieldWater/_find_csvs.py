# -*- coding: utf-8 -*-
"""
find csv file within polygon

@author: chihiro
"""
import os
import pandas as pd
import geopandas as gpd


## You need vars↓ with the same name:
    #gdf_extent,
    #gdf_A_dic,
    #gdf_region, 
    #gdf_grid,
    #list_palm, 
    #var_csv_dir
# def find_page(gdfpoi, gdf_extent):
#     for i,row in gdf_extent.iterrows():
#         grid = row.geometry
#         if gdfpoi.within(grid).values[0]: #if point is on the line, it's false
#             page = row.PageName
#         else:
#             if gdfpoi.intersects(grid).values[0]: #if point is on the line, it's false
#                 page = row.PageName
#     return page
def find_page(gdfpoi, gdf_extent):
    # for i,row in gdf_extent.iterrows():
    for A,row in extent_polygons.items():
        # grid = row.geometry
        grid = row.geometry.values[0]
        if gdfpoi.within(grid).values[0]: #if point is on the line, it's false
            page = row.PageName.values[0]
        else:
            # if gdfpoi.intersects(grid).values[0]: #if point is on the line, it's false
            #     page = row.PageName.values[0]
            continue
    return page

def find_index(gdfpoi, pagenum, gdf_A_dic):
    gdf_page = gdf_A_dic[pagenum]
    gdf_page_intersecting = gdf_page[gdf_page.intersects(gdfpoi.geometry.values[0])]
    index_want = gdf_page_intersecting.raster_val.values[0]
    return int(index_want)


def main(regi_poly, gdf_region, gdf_grid, gdf_extent, gdf_A_dic, list_palm, var_csv_dir):
    # regi_poly = gdf_region.loc[0,:].geometry
    
    gdf_regi = gpd.GeoDataFrame({"geometry":[regi_poly]}).set_crs(gdf_region.crs) #multipolygon
    
    """ # select target grids"""
    ## grids which intersect with region polygon
    gdf_tar_grid = gdf_grid[gdf_grid.intersects(regi_poly)] #index is target tif filename
    ### select grid id within palm 2
    gdf_tar_grid = gdf_tar_grid[gdf_tar_grid['raster_val'].isin(list_palm)] #palm 2002
    
    ## convert to point
    gdf_tar_grid['centroid'] = gdf_tar_grid.geometry.centroid
    gdf_centroids = gdf_tar_grid.copy()
    gdf_centroids['geometry'] = gdf_centroids['centroid']
    gdf_centroids = gdf_centroids.drop(columns=['centroid'])
    
    csvlist = []
    for poi in gdf_centroids.geometry:
        gdfp = gpd.GeoDataFrame({"geometry":[poi]}).set_crs(gdf_region.crs)
        A = find_page(gdfp, gdf_extent)
        index_target = find_index(gdfp, A, gdf_A_dic)
        csvfile = var_csv_dir + os.sep + A + os.sep + f"{index_target}.csv"
        csvlist.append(csvfile)
    
    return csvlist