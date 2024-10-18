import numpy as np
import matplotlib.pyplot as plt
import random

class robot:
    def __init__(self, x, y, id=None):
        self.id = id
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
            new_robot = robot(x, y, id=i)
            self.group.append(new_robot)

        for i in range(group_size):
            neighbors = []
            for j in range(group_size):
                if i != j and len(neighbors) < 3:
                    neighbors.append(self.group[j])
            self.group[i].neighbors = neighbors
        if self.group_size < 3:
            # Align vehicles when there are 3 or fewer
            for i, vehicle in enumerate(self.group):
                vehicle.neighbors = [v for j, v in enumerate(self.group) if j != i]
        else:
            # Choose the two closest vehicles as neighbors for each vehicle
            for i, vehicle in enumerate(self.group):
                other_vehicles = [v for j, v in enumerate(self.group) if j != i]
                distances = [np.linalg.norm(np.array(vehicle.position) - np.array(other.position)) for other in other_vehicles]
                sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
                vehicle.neighbors = [other_vehicles[idx] for idx in sorted_indices[:2]]
            
            # Ensure each vehicle has exactly two neighbors
            for vehicle in self.group:
                if len(vehicle.neighbors) < 2:
                    # If a vehicle has less than two neighbors, find the next closest vehicle
                    other_vehicles = [v for v in self.group if v != vehicle and v not in vehicle.neighbors]
                    if other_vehicles:
                        distances = [np.linalg.norm(np.array(vehicle.position) - np.array(other.position)) for other in other_vehicles]
                        closest_vehicle = other_vehicles[np.argmin(distances)]
                        vehicle.neighbors.append(closest_vehicle)
        # Print each vehicle's neighbors
        print("Vehicle Neighbors:")
        for i, vehicle in enumerate(self.group):
            neighbor_indices = [self.group.index(neighbor) for neighbor in vehicle.neighbors]
            print(f"Vehicle {vehicle.id}: Neighbors {neighbor_indices}")
        print()  # Add a blank line for readability
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
        Calculate the parallel velocity component for SUSD algorithm in 3D.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle object containing necessary attributes for calculation.

        Returns

        """
        k_p = 2  # Constant for parallel velocity calculation
        a = 40.0   # Desired inter-agent distance

        r_i = self.position[0:2]  # Current vehicle position
        r_j = self.neighbors[0].position[0:2]  # Neighbor position

        q = r_j - r_i  # Baseline vector
        q_unit = q / np.linalg.norm(q)  # Unit vector of the baseline

        s = np.dot(r_j - r_i, q_unit)  # Projection of relative position onto baseline

        # Calculate the parallel velocity component
        v_para = k_p * (s - a) * q_unit



        return v_para


    def velPerpendicular2SUSD(self, group, pollution):
        """
        Calculate the perpendicular velocity component for SUSD algorithm in 2D.

        Parameters
        ----------
        pollution : Pollution
            The pollution object to get scalar field measurements.

        Returns
        -------
        numpy.ndarray
            2D perpendicular velocity vector along the SUSD direction
        """
        k = 10.0  # Attraction constant
        C = 0.0  # Constant offset

        ri = np.array(group[0].position[:2])
        rj = np.array(group[1].position[:2])
        y_r_i = pollution.calculate_concentration(self.position)  # Concentration measurement for current vehicle
        print(f"Robot: {self.id} Concentration measurement for current vehicle: {y_r_i}")
        
        # Calculate the unit vector q
        q = ri - rj

        # Calculate the unit vecotr n (n is perpendicular to q)
        n = np.array([-q[1], q[0]])
        n_unit = n / np.linalg.norm(n)


        # Calculate the perpendicular velocity magnitude
        v_perp_mag = k/(y_r_i+1) + C
        
        # Calculate the projection of v_perp onto n
        v_perp_n = np.multiply(v_perp_mag, n_unit)


        
        return v_perp_n