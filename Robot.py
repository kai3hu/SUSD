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
        if self.group_size < 3:
            # Align vehicles when there are fewer than 3
            for i, vehicle in enumerate(self.group):
                vehicle.neighbors = [v for j, v in enumerate(self.group) if j != i]
        else:
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
        """
        k = 10.0  # Attraction constant
        C = 3.0  # Constant offset

        r_i = self.position  # Current vehicle position
        r_j = self.neighbors[0].position  # Neighbor position

        q = r_j - r_i  # Baseline vector
        q_mag = np.linalg.norm(q)
        q_unit = q / q_mag  # Unit vector of the baseline

        y_r = pollution.calculate_concentration_gradient(self.position)  # Scalar field measurement
        y_r_mag = np.linalg.norm(y_r)

        if y_r_mag > 0:
            y_r_unit = y_r / y_r_mag


        v_perp_mag = k * y_r_mag + C

        # Calculate the perpendicular velocity vector aligned with y_r
        v_perp = v_perp_mag * y_r_unit

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
        k_p = 2  # Constant for parallel velocity calculation
        a = 20.0   # Desired inter-agent distance

        r_i = np.array(self.position[:2])  # Current vehicle position (2D)
        r_j = np.array(self.neighbors[0].position[:2])  # Neighbor position (2D)

        q = r_j - r_i  # Baseline vector (2D)
        q_mag = np.linalg.norm(q)
        q_unit = q / q_mag  # Unit vector of the baseline (2D)

        s = np.dot(r_j - r_i, q_unit)  # Projection of relative position onto baseline

        # Calculate the parallel velocity component (2D)
        v_para = k_p * (s - a) * q_unit

        return v_para

    ###############################################################################





