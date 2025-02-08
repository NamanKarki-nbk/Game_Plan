import numpy as np
from development_and_analysis.k_means_custom import KMeansCustom
from collections import Counter

def get_top_half(image):
    """Extract the top half of the given image."""
    return image[0:int(image.shape[0] / 2), :]

def reshape_to_2d(image):
    """Reshape the image to a 2D array of pixels."""
    return image.reshape(-1, 3)

def perform_kmeans(image_2d, n_clusters=2, random_state=0):
    """Perform K-Means clustering using K-Means++ for better initialization."""
    kmeans = KMeansCustom(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(image_2d)
    return kmeans




def analyze_corners(clustered_image):
    """Determine which cluster represents the background based on corner pixels."""
    # Get the labels from the 4 corners
    corner_clusters = [
        clustered_image[0, 0],  # Top-left
        clustered_image[0, -1],  # Top-right
        clustered_image[-1, 0],  # Bottom-left
        clustered_image[-1, -1]  # Bottom-right
    ]
    
    # Find the most common label in corners (assumed to be background)
    background_cluster = Counter(corner_clusters).most_common(1)[0][0]
    
    # The player cluster is the other one
    player_cluster = 1 - background_cluster  # Assuming only 2 clusters (0 and 1)

    return player_cluster, background_cluster

# import numpy as np
# import cv2
# from development_and_analysis.k_meansplus import KMeansLab  # Import the updated KMeansLab class

# def get_top_half(image):
#     """Extract the top half of the given image."""
#     return image[: image.shape[0] // 2, :]

# def reshape_to_2d(image):
#     """Reshape the image to a 2D array of pixels."""
#     return image.reshape(-1, 3)

# def perform_kmeans(image_2d, n_clusters=2, random_state=42):
#     """Perform K-Means++ clustering on the given 2D image data using Lab color space."""
    
#     # Convert to Lab color space
#     image_2d = np.array(image_2d, dtype=np.uint8)

#     # Use KMeans++ with Lab color space
#     kmeans = KMeansLab(n_clusters=n_clusters, random_state=random_state)
#     kmeans.fit(image_2d)

#     return kmeans

# def analyze_corners(clustered_image):
#     """Analyze the corner clusters to identify the player cluster."""
    
#     # Extract corner clusters (top-left, top-right, bottom-left, bottom-right)
#     corner_clusters = [
#         clustered_image[0, 0],
#         clustered_image[0, -1],
#         clustered_image[-1, 0],
#         clustered_image[-1, -1]
#     ]

#     # The background is the most frequent color in the corners
#     non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

#     # The player cluster is the opposite of the non-player cluster
#     player_cluster = 1 - non_player_cluster

#     return player_cluster, non_player_cluster
