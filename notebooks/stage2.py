import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage

from sklearn.cluster import KMeans
from functions import build_topotensor_from_diagram

# ---------- 1) Synthetic dataset (two circles + small Gaussian cluster) ----------
np.random.seed(0)

def sample_circle(radius, n_points=300, noise=0.03, center=(0,0)):
    theta=np.random.rand(n_points) * 2 * np.pi
    x = center[0] + radius * np.cos(theta) + noise * np.random.randn(n_points)
    y = center[1] + radius * np.sin(theta) + noise * np.random.randn(n_points)
    return np.vstack([x,y]).T

c1=sample_circle(radius=1.0, n_points=400, noise=0.03, center=(0,0))
c2=sample_circle(radius=0.5, n_points=200, noise=0.03, center=(0.5, 0.2))
cluster = np.random.randn(150, 2) * 0.12 +np.array([3.0, -0.5])

point_cloud = np.vstack([c1, c2, cluster])

plt.figure(figsize=(5,5))
plt.scatter(point_cloud[:,0], point_cloud[:,1], s=6)
plt.title("Synthetic point cloud: two circles + cluster")
plt.axis('equal')
plt.show()

# ---------- 2) Compute persistent homology (H1 loops) ----------
vr = VietorisRipsPersistence(homology_dimensions=[1])
diagrams = vr.fit_transform([point_cloud])
pd = diagrams[0]

if pd.shape[1] >= 3:
    h1_points = pd[pd[:,2] ==1][:,:2]
else:
    hi_points = pd

persistence_vals = h1_points[:,1] - h1_points[:,0]
nonzero_mask = persistence_vals > 1e-9
h1_points = h1_points[nonzero_mask]
persistence_vals = persistence_vals[nonzero_mask]

print("Found H1 features;", h1_points.shape[0])

plt.figure(figsize=(5,5))
plt.scatter(h1_points[:,0], h1_points[:,1], c=persistence_vals, cmap='viridis', s=40)
mx = max(np.max(h1_points[:,0]), np.max(h1_points[:,1])) if h1_points.size else 1.0
plt.plot([0,mx], [0,mx], 'k--', alpha=0.6)
plt.xlabel("birth"); plt.ylabel("death"); plt.title("H1 Persistence Diagram")
plt.colorbar(label='persistence (death - birth)')
plt.show()





# Build Topotensor (adjust M and m as you like)
tensor, meta = build_topotensor_from_diagram(h1_points, M=2, m=3, image_size=(32,32),sigma=0.05, weight=None)
print("TopoTensor meta:", meta)

# ---------- 4) Visualize channels ----------
if tensor.shape[0] == 0:
    print('No Hi features found; Topotensor is empty.')
else:
    n_channels = tensor.shape[0]
    cols = min(6, n_channels)
    rows = int(np.ceil(n_channels /cols))
    plt.figure(figsize=(3*cols, 2.6*rows))
    for i in range(n_channels):
        plt.subplot(rows, cols, i+1)
        plt.imshow(tensor[i][0], origin = 'lower', interpolation='nearest')
        plt.title(f'Channel {i+1}')
        plt.axis('off')

plt.suptitle("TopoTensor channels (each = persistence image for a feature or cluster)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # <-- this adds space for the suptitle
plt.show()


