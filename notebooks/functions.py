import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage

from sklearn.cluster import KMeans

# ----------  Build TopoTensor: select top-M and cluster remaining into m ----------
def build_topotensor_from_diagram(h1_points, M=2, m=3, image_size=(32,32),sigma=0.05, weight=None):
    """
    h1_points: (N,2) array of (birth, death)
    M: keep top M single-feature channels
    m: number of clusters for the remaining points
    image_size: (H,W) resolution for each persistence image
    sigma: gaussian bandwidth for PersistenceImage
    weight: optional weight function for PersistenceImage (None -> default)
    """
    if h1_points.size == 0:
        return np.zeros((0,image_size[0], image_size[1])),{'M_eff':0, 'n_clusters':0}
    
    births = h1_points[:,0]
    deaths = h1_points[:,1]
    persists = deaths - births
    order = np.argsort(-persists)
    sorted_pts = h1_points[order]

    M_eff = min(M, sorted_pts.shape[0])
    top_pts = sorted_pts[:M_eff]
    rem_pts = sorted_pts[M_eff:]

    # Cluster remaining points in (birth, persistence) coords
    clusters = []
    if rem_pts.shape[0] > 0:
        X_cluster = np.column_stack([rem_pts[:,0],rem_pts[:,1]-rem_pts[:,0]])
        n_clusters = min(m,rem_pts.shape[0])
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(X_cluster)
        labels = km.labels_
        for k in range(n_clusters):
            clusters.append(rem_pts[labels == k])
    # PersistenceImage transformer (giotto-tda), expects (n_diagrams, n_points, 2)
    pim = PersistenceImage(sigma=sigma, weight_function=weight, n_bins=image_size[0])
    
    channels = []
    # top single-point channels
    for pt in top_pts:
        diag_subset = np.hstack([pt[None, :], np.ones((1, 1))])  # (1, 3)
        diag_subset = diag_subset[None, ...]                     # (1, 1, 3)
        img = pim.fit_transform(diag_subset)[0]
        channels.append(img)

    # cluster channels: aggregate all points inside each cluster
    for cluster_pts in clusters:
        pts = np.atleast_2d(cluster_pts )
        diag3 = np.hstack([pts, np.ones((pts.shape[0],1))])
        diag_subset = diag3[None, ...]
        img = pim.fit_transform(diag_subset)[0]
        channels.append(img)
    
    tensor = np.stack(channels, axis = 0) if channels else np.zeros((0,image_size[0],image_size[1]))
    meta = {'M_requested': M, 'M_eff': M_eff, 'n_clusters': len(clusters), 'tensor_shape':tensor.shape}
    return tensor, meta