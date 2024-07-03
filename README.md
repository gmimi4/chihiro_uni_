# chihiro_uni_
# Replanting Blueprint
This program was developed on Windows.
You may need to change code to set path to files depending on your work environment.
This progarm utilizes arcpy in creating slope, curvature and vectorizing line images. These steps will replace arcpy with open source libraries in the future.

The followings provide small notes for each code in the steps.

## Creation of CS image (_01_CreateCSimage)
Please run "run_CSimg.py"
You need DEM for initiating the program
### _01_Gaussian.py
You need process DEM with Gausian filter for generating a curvature image
### _02_Slope_Curvature.py
Use original DEM for slope image, and Gaussian filtered DEM for curvature image
### _03_CSMap_export.py

## Terrace detection by deep learning (_02_TerraceDetection)
This program provides a trained model. However, please note that the trained model still needs improvement.
You will see insufficient segmentation result by the current model. In such case, you need to train a new model or apply other detection methods.

After terraces were segmented, please run Please run "run_CSimg.py"


run_pairing_terrace.py
run_point_generation.py
run_terrace_detection.py


_03_PairingTerraces
_04_Point_generation
