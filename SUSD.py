import numpy as np
import matplotlib.pyplot as plt
import Robot as rb
import Pollution as pl
from matplotlib.animation import FuncAnimation

pollution = pl.pollution(60, 60, 1.59, 30)
r = rb.robot(600, 800)
group = r.group = r.build_group(2)

size = 1000
resolution = 100

x = np.linspace(-1, size, resolution)
y = np.linspace(-1, size, resolution)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 0)  # Set Z to 30 for all points
C = pollution(X, Y, Z)

# Create a figure
fig, ax = plt.subplots(figsize=(8, 6))

route_history = [[] for _ in range(len(group))]

# Create a colormap for concentration
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=C.min(), vmax=C.max())

def update(frame):
    t = frame * 0.2  # Time step
    
    # Calculate velocities for all vehicles
    new_velocities = []
    for auv in group:
        vper = auv.velPerpendicular2SUSD(group, pollution)
        print(f"Perpendicular velocity: {vper}")
        vpar = auv.velParallelSUSD()
        print(f"Parallel velocity: {vpar}")
        auv.speed = vper + vpar
        new_velocities.append(auv.speed)
    
    # Compare velocities and adjust if the difference is less than 1
    for i in range(len(new_velocities)):
        for j in range(i+1, len(new_velocities)):
            vel_diff = np.linalg.norm(new_velocities[i] - new_velocities[j])
            if vel_diff < 1:
                if np.linalg.norm(new_velocities[i]) > np.linalg.norm(new_velocities[j]):
                    new_velocities[i] += new_velocities[i] / np.linalg.norm(new_velocities[i])
                else:
                    new_velocities[j] += new_velocities[j] / np.linalg.norm(new_velocities[j])

    # Update velocities and positions simultaneously
    for i, auv in enumerate(group):
        auv.speed = new_velocities[i]
        new_position = auv.position + auv.speed * 0.2
        auv.position = new_position

    current_time = frame * 0.2 # Current time

    ax.clear()
    
    # Plot concentration contours
    contour = ax.contour(X, Y, C, levels=20, cmap=cmap, norm=norm)
    
    # Plot current positions of vehicles
    for i, auv in enumerate(group):
        ax.plot(auv.position[0], auv.position[1], 'o', label=f'Vehicle {i+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Pollution Concentration\nWind Speed: {pollution.u:.2f} m/s, Direction: {np.degrees(pollution.v):.2f}°', fontsize=16)
    ax.legend()
    
    return ax

# Create the animation
anim = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Show the animation
plt.show()
