import numpy as np
import matplotlib.pyplot as plt
import Robot as rb
import Pollution as pl
from matplotlib.animation import FuncAnimation

pollution = pl.pollution(0, 0, 1.59, 30)
r = rb.robot(800, 800)
group=r.build_group(2)

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

def update(frame):
    t = frame * 0.1  # Time step
    
    # Calculate velocities for all vehicles
    new_velocities = []
    for auv in group:
        vper = auv.velPerpendicularSUSD(pollution)
        vpar = auv.velParallelSUSD()
        auv.speed = vper + vpar
        new_velocities.append(auv.speed)
    
    # Update velocities and positions simultaneously
    for i, auv in enumerate(group):
        auv.speed = new_velocities[i]
        new_position = auv.position + auv.speed * 0.1
        auv.position = new_position
        route_history[i].append(list(auv.position))

    ax.clear()
    
    # Re-create the contour plot
    contour = ax.contour(X, Y, C, levels=30, colors='black', alpha=0.5)
    
    for i, history in enumerate(route_history):
        history_array = np.array(history)
        ax.plot(history_array[:, 0], history_array[:, 1], label=f'Vehicle {i+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Pollution Concentration\nWind Speed: {pollution.u:.2f} m/s, Direction: {np.degrees(pollution.v):.2f}°', fontsize=16)
    ax.legend()
    
    return ax

# Create the animation
anim = FuncAnimation(fig, update, frames=500, interval=20, blit=False)

# Save the animation as a GIF
anim.save('pollution_simulation.gif', writer='pillow', fps=30)

plt.close()
