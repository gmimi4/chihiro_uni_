# -*- coding: utf-8 -*-
"""
# clustering pixels by k-ean
"""
import os
import numpy as np
import rasterio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


tiffile = '/Volumes/PortableSSD/Malaysia/ENSO/04_mins/_mosaic/mosaic_min_rain.tif'
out_dir = '/Volumes/PortableSSD/Malaysia/ENSO/05_kclustering'

with rasterio.open(tiffile) as src:
    raster_data = src.read(1)  # Read the first band
    profile = src.profile

# Flatten the raster data and mask no-data values
raster_flat = raster_data.flatten()
valid_pixels = raster_flat[raster_flat != profile['nodata']]  # Exclude no-data values

# Perform K-means clustering
n_clusters = 5  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(valid_pixels.reshape(-1, 1))
test =np.unique(clusters)

# Reconstruct the clustered raster
clustered_raster = np.full_like(raster_flat, 99, dtype=np.int32)
clustered_raster[raster_flat != profile['nodata']] = clusters
clustered_raster = clustered_raster.reshape(raster_data.shape)

# Save the clustered raster
profile.update(dtype='int32', count=1, compress='lzw')

output_raster = out_dir+os.sep+ f"{os.path.basename(tiffile)[:-4]}_cluster.tif"
with rasterio.open(output_raster, 'w', **profile) as dst:
    dst.write(clustered_raster, 1)

print(f"Clustered raster saved at: {output_raster}")

# Plot the clustered raster
plt.figure(figsize=(10, 6))
plt.imshow(clustered_raster, cmap='tab20', interpolation='none')
plt.colorbar(label="Cluster ID")
plt.title("Clustered Raster")
plt.show()             
        
 
    