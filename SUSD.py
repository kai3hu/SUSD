import numpy as np
import matplotlib.pyplot as plt
import Robot as rb
import Pollution as pl

pollution = pl(0, 0, 1.59, 30)
r = rb(20, 20)
group=r.build_group(5)

size = 500
resolution = 100

x = np.linspace(-1, size, resolution)
y = np.linspace(-1, size, resolution)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 0)  # Set Z to 30 for all points
C = pollution(X, Y, Z)
# Create a figure
fig, ax = plt.subplots(figsize=(14, 12))
# Create a contour plot
contour = ax.contour(X, Y, C, levels=50, colors='black')
# Highlight the pollution source
ax.scatter(pollution.x, pollution.y, color='red', s=100, marker='*', label='Pollution Source')
ax.set_title(f'2D Pollution Concentration at Z = 30m\nWind Speed: {pollution.u:.2f} m/s, Direction: {np.degrees(pollution.v):.2f}°', fontsize=16)
ax.set_xlabel('X coordinate (m)', fontsize=12)
ax.set_ylabel('Y coordinate (m)', fontsize=12)
# Add legend
ax.legend()


plt.scatter([robot.x for robot in group], [robot.y for robot in group], s=50)
plt.show()
plt.close()
