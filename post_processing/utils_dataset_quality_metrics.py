import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils_pp_standalone import *
from collections import defaultdict
import re
from sklearn.preprocessing import MinMaxScaler

def conv_hull_volume(points,chv_dims, scaler, plot=False, axes=None, plot_params=None):
    points_scaled = scaler.transform(points)

    # Compute convex hull (uses Qhull internally)
    hull = ConvexHull(points_scaled)
    
    
    print("Hull vertices:")
    print(hull.vertices)
    
    print("Hull volume:")
    print(hull.volume)
    
    # if points.shape[1]==3:
    #     # Create 3D plot
    #     #fig = plt.figure()
    #     #ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot original points
    #     ax.scatter(points_scaled[:, 0], points_scaled[:, 1], points_scaled[:, 2], alpha = 0.3)#color='gray')
        
    #     # Plot hull triangles
    #     for simplex in hull.simplices:
    #         triangle = points_scaled[simplex]
    #         ax.plot_trisurf(triangle[:, 0],
    #                         triangle[:, 1],
    #                         triangle[:, 2],
    #                         color='gray',
    #                         alpha=0.3)
        
    #     ax.set_xlabel(chv_dims[0])
    #     ax.set_ylabel(chv_dims[1])
    #     ax.set_zlabel(chv_dims[2])
        
    #     plt.show()
       
    if plot ==True:
        hull_2d = ConvexHull(points_scaled[:,:2])

        # Plot
        axes[0].scatter(points_scaled[:,0], points_scaled[:,1], marker=plot_params['marker'], color =plot_params['color'])
    
        for simplex in hull_2d.simplices:
            axes[0].plot(points_scaled[simplex, 0], points_scaled[simplex, 1], color =darken(plot_params['color'], 0.9), linestyle = plot_params['linestyle'])
            
        hull_2d = ConvexHull(points_scaled[:,[2,1]])

        axes[1].scatter(points_scaled[:,2], points_scaled[:,1], marker=plot_params['marker'], color =plot_params['color'])
    
        for simplex in hull_2d.simplices:
            axes[1].plot(points_scaled[simplex, 2], points_scaled[simplex, 1], color = darken(plot_params['color'], 0.9), linestyle = plot_params['linestyle'])
        
    return hull

import matplotlib.colors as mcolors

def darken(color, factor=0.7):
    rgb = mcolors.to_rgb(color)
    return tuple(c * factor for c in rgb)

dark_silver = darken('silver', 0.6)
