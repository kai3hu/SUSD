import numpy as np
import matplotlib.pyplot as plt
import random

class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.position = np.array([x, y])
        self.speed = np.array([0, 0])
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
        if self.group_size <= 3:
            # Align vehicles when there are 3 or fewer
            for i, vehicle in enumerate(self.group):
                vehicle.neighbors = [v for j, v in enumerate(self.group) if j != i]
        else:
            # Assign exactly two neighbors to each vehicle
            for i, vehicle in enumerate(self.group):
                available_neighbors = [v for j, v in enumerate(self.group) if j != i]
                new_neighbors = random.sample(available_neighbors, 2)
                vehicle.neighbors = new_neighbors
                for neighbor in new_neighbors:
                    if vehicle not in neighbor.neighbors and len(neighbor.neighbors) < 2:
                        neighbor.neighbors.append(vehicle)
            
            # Ensure each vehicle has exactly two neighbors
            for vehicle in self.group:
                if len(vehicle.neighbors) < 2:
                    available_neighbors = [v for v in self.group if v != vehicle and len(v.neighbors) < 2]
                    while len(vehicle.neighbors) < 2 and available_neighbors:
                        new_neighbor = random.choice(available_neighbors)
                        vehicle.neighbors.append(new_neighbor)
                        new_neighbor.neighbors.append(vehicle)
                        available_neighbors.remove(new_neighbor)
    
        # Remove duplicates from neighbor lists
        for vehicle in self.group:
            vehicle.neighbors = list(set(vehicle.neighbors))
        
        return self.group

    ###############################################################################

    def velPerpendicularSUSD(self, pollution):
        """
        Calculate the perpendicular velocity component for SUSD algorithm in 2D.

        Parameters
        ----------
        pollution : Pollution
            The pollution object to get scalar field measurements.

        Returns
        -------
        numpy.ndarray
            2D perpendicular velocity vector
        """
        k = 10.0  # Attraction constant
        C = 3.0  # Constant offset

        r_i = self.position  # Current vehicle position

        y_r = pollution.calculate_concentration_gradient(r_i)  # Scalar field measurement
        y_r_mag = np.linalg.norm(y_r)

        if y_r_mag > 0:
            y_r_unit = y_r / y_r_mag
            v_perp_mag = k * y_r_mag + C
            v_perp = v_perp_mag * y_r_unit
        else:
            v_perp = np.zeros(2)  # If gradient is zero, no perpendicular velocity

        return v_perp

    ###############################################################################

    def velParallelSUSD(self):
        """
        Calculate the parallel velocity component for SUSD algorithm in 2D.

        Parameters
        ----------
        Returns
        -------
        numpy.ndarray
            2D parallel velocity vector
        """
        k1 = 2  # Constant for parallel velocity calculation
        a0 = 20.0   # Desired inter-agent distance

        r_i = np.array(self.position[:2])  # Current vehicle position (2D)
        q_i = self.calculate_baseline_vector()  # Baseline vector for current vehicle

        v_para = np.zeros(2)  # Initialize parallel velocity vector

        for neighbor in self.neighbors:
            r_j = np.array(neighbor.position[:2])  # Neighbor position (2D)
            a0_ij = a0  # Can be adjusted if different desired distances for different neighbors

            # Calculate the parallel velocity component (2D)
            v_para += k1 * (np.dot(r_j - r_i, q_i) - a0_ij)

        return v_para * q_i

    def calculate_baseline_vector(self):
        """
        Calculate the baseline vector for the current vehicle.

        Returns
        -------
        numpy.ndarray
            2D unit vector representing the baseline
        """
        if not self.neighbors:
            return np.array([1, 0])  # Default direction if no neighbors

        avg_neighbor_pos = np.mean([np.array(n.position[:2]) for n in self.neighbors], axis=0)
        q = avg_neighbor_pos - np.array(self.position[:2])
        return q / np.linalg.norm(q)

    ###############################################################################




