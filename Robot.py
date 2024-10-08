import numpy as np
import matplotlib.pyplot as plt
import random

class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.direction = 0
        self.destination = None
        self.path = []
        self.neighbors = []

    def build_group(self, group_size):
        self.group = []
        self.group_size = group_size
        distance = 30  # Fixed distance between neighbors

        for i in range(group_size):
            x = self.x + (i % 2) * distance
            y = self.y + (i // 2) * distance
            new_robot = robot(x, y)
            self.group.append(new_robot)

        for i in range(group_size):
            neighbors = []
            for j in range(group_size):
                if i != j and len(neighbors) < 3:
                    neighbors.append(self.group[j])
            self.group[i].neighbors = neighbors

        for i, vehicle in enumerate(self.group):
            available_neighbors = [v for j, v in enumerate(self.group) if j != i]
            num_neighbors = random.randint(1, 3)
            if num_neighbors > 0:
                new_neighbors = random.sample(available_neighbors, num_neighbors)
                vehicle.neighbors = new_neighbors
                for neighbor in new_neighbors:
                    if vehicle not in neighbor.neighbors:
                        neighbor.neighbors.append(vehicle)
    
        # Remove duplicates from neighbor lists
        for vehicle in self.group:
            vehicle.neighbors = list(set(vehicle.neighbors))
        
        return self.group

