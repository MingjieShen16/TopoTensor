import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

np.random.seed(42)

theta = np.linspace(0, 2 * np.pi, 150)
circle = np.stack((np.cos(theta), np.sin(theta)), axis=1)
noise = 0.05 * np.random.rand(*circle.shape)
point_cloud =circle + noise

plt.figure(figsize=(4,4))
plt.scatter(point_cloud[:,0], point_cloud[:,1], s=10)
plt.title("Noisy Circle Point Cloud")
plt.axis('equal')
plt.show()

vr = VietorisRipsPersistence(homology_dimensions=[0,1])
diagrams = vr.fit_transform([point_cloud])

plot_diagram(diagrams[0])

print("Shape of persistence diagram:", diagrams[0].shape)
print("First few entries (birth, death, dimension):")
print(diagrams[0][:5])
