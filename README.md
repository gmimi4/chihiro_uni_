# chihiro_uni_
# Replanting Blueprint
This program was developed on Windows.
You may need to change code to set path to files depending on your work environment.
This progarm utilizes arcpy in creating slope, curvature and vectorizing line images. These steps will replace arcpy with open source libraries in the future.
This program sets the spatial coordinate system as WGS 84/UTM zone 48N (epsg:32648)

## Creation of CS image (_01_CreateCSimage)
Please run codes in "run_CSimg.py"
The followings provide small notes for each code in the steps.

You need DEM for initiating the program
- _01_Gaussian.py
You need process DEM with Gausian filter for generating a curvature image.
- _02_Slope_Curvature.py
Use original DEM for slope image, and Gaussian filtered DEM for curvature image.
- _03_CSMap_export.py

## Terrace detection by deep learning (_02_TerraceDetection)
This program provides a trained model. However, please note that the trained model still needs improvement.
You will see insufficient segmentation result by the current model. In such case, you need to train a new model or apply other detection methods.

After terraces were segmented, please run Please run codes in "run_CSimg.py"
The followings provide small notes for each code in the steps.

- _00_dilation.py
- _01_vectorize_centerlines.py
You need to set the path to arcpy and path to "_01_vectorize_centerlines_arctool.py" in the code.
- _02_filtering_by_intersects.py
You can change the angle threshold "angle_thre" (default 45).
- _03_cut_intersects.py
You can change the cut distance "buff_distance" (default 8).
- _04_cut_intersects_2lines.py
You can change the cut distance "buff_distance" (default 8).
You can change the angle threshold "angle_thre" (default 45).
- _05_connect_nearlines.py
You can change the distance threshold "linestring.length" (default 5).
You can change the angle threshold "angle_thre" (default 45).
- _06_erase_by_roads.py
- _07_cut_intersects_pairing.py
You can change the cut distance "buff_distance" (default 1).
- _08_cut_intersects_2lines_pairing.py
You can change the cut distance "buff_distance" (default 1).
You can change the angle threshold "angle_thre" (default 45).
- _99_devide_line_roads.py
- _99_devide_lines.py

run_pairing_terrace.py
run_point_generation.py
run_terrace_detection.py


_03_PairingTerraces
_04_Point_generation
