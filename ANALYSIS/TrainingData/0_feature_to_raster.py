# -*- coding: utf-8 -*-
"""
# convert point and lines to training polygon, then raster
# 10: terrace edge
# 1: others
"""

import os
import arcpy
import geopandas as gpd

line_shp = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\1_shp\trainingline.shp"
label_pointshp = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\1_shp\labelpoints.shp"
extent_shp = r"E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\1_shp\extent.shp"
out_dir = r'E:\Malaysia\01_Blueprint\Pegah_san\03_UNet\2_retraining\0_preparation\2_convert'

tmp_dir = out_dir + os.sep + "_tmp"
os.makedirs(tmp_dir,exist_ok=True)

epsg_code = 32647

#Feature to Polygon
outfile = tmp_dir + os.sep + "to_polygon.shp"
arcpy.management.FeatureToPolygon([line_shp,extent_shp], outfile, "", "", label_pointshp)

## 以下諦め
# # add field area_Ha
# inputfile = outfile
# arcpy.management.AddFields(inputfile, field_description="area_ha DOUBLE # # # #", template=None)
# #calc area ha
# arcpy.management.CalculateGeometryAttributes(inputfile, geometry_property="AREA AREA",
#                                              area_unit="HECTARES", coordinate_format="SAME_AS_INPUT")

## clip polygon by extent
input = outfile
outfile = tmp_dir + os.sep + "to_polygon_clip.shp"
arcpy.analysis.Clip(input, extent_shp, outfile)

## convert background classval to 1 (terrace edge is 10)
input = outfile
gdf = gpd.read_file(input)
back_idx = gdf.index[gdf['Id'] != 10].tolist() 
gdf.loc[back_idx, 'Id'] = 1

outfile = out_dir + os.sep + "polygon_fin.shp"
gdf.to_file(outfile)

## convert to raster
inputfile = outfile
outfile = out_dir + os.sep + "polygon_ras02.tif"
arcpy.conversion.PolygonToRaster(inputfile, "Id", outfile, "CELL_CENTER", "", 0.2)
