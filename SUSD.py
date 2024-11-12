import numpy as np
import matplotlib.pyplot as plt
import Robot as rb
import Pollution as pl
from matplotlib.animation import FuncAnimation

pollution = pl.pollution(60, 60, 1.59, 30)
r = rb.robot(800, 800)
gp = r.build_group(10)

gp.update_neighbors()
for i in gp.group:
    print(f"Vehicle {i.id} has neighbors: {[n.id for n in i.neighbors]}")

size =1000
resolution = 100

x = np.linspace(-1, size, resolution)
y = np.linspace(-1, size, resolution)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 0)  # Set Z to 30 for all points
C = pollution(X, Y, Z)


# Create a figure
fig, ax = plt.subplots(figsize=(12, 8))

route_history = [[] for _ in range(len(gp.group))]

# Create a colormap for concentration
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=C.min(), vmax=C.max())

# Add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Concentration')

def update(frame):
    t = frame * 0.1  # Time step
    

    
    # Calculate velocities for all vehicles
    new_velocities = []
    for auv in gp.group:
        vper = auv.velPerpendicularSUSD(gp, pollution)

        vpar = auv.velParallelSUSD(gp)
        auv.speed = vper + vpar
        new_velocities.append(auv.speed)

    # Print distance to neighbors for each vehicle
    print(f"\nFrame {frame}: Distance to neighbors")
    for auv in gp.group:
        print(f"Vehicle {auv.id}:")
        for neighbor in auv.neighbors:
            distance = np.linalg.norm(np.array(auv.position[:2]) - np.array(neighbor.position[:2]))
            print(f"  - Distance to Vehicle {neighbor.id}: {distance:.2f}")
        print (f"Position: {auv.position}")
    print()  # Add a blank line for readability

    # Update velocities and positions simultaneously
    for i, auv in enumerate(gp.group):
        auv.speed = new_velocities[i]
        new_position = auv.position + auv.speed * 0.1
        auv.position = new_position


    
    current_time = frame * 0.1 # Current time
    ax.clear()
    
    # Plot concentration contours
    contour = ax.contour(X, Y, C, levels=20, cmap=cmap, norm=norm)
    
    # Plot current positions of vehicles
    for i, auv in enumerate(gp.group):
        ax.plot(auv.position[0], auv.position[1], '.', markersize=5, label=f'V{i}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Pollution Concentration\nWind Speed: {pollution.u:.2f} m/s, Direction: {np.degrees(pollution.v):.2f}°', fontsize=16)
    ax.legend()
    return ax

# Create the animation
anim = FuncAnimation(fig, update, frames=550, interval=10, blit=False)

# Save the animation as a GIF
anim.save('susd_animation.gif', writer='pillow', fps=25)

# Show the animation
plt.show()
