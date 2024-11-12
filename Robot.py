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
        gp=group(group_size)
        distance = 30  # Fixed distance between neighbors
        
        
        for i in range(group_size):
            x = self.x + (i % 2) * distance
            y = self.y + (i // 2) * distance
            new_robot = robot(x, y, id=i)
            gp.group.append(new_robot)

        # Define the sensing range for each agent
        sensing_range = 100  # Adjust this value as needed

        # Create an undirected graph G=(U,E)
        U = gp.group  # Node set U
        E = set()  # Edge set E

        # Determine edges based on sensing range and neighbor limit
        for i, agent_i in enumerate(U):
            distances = [(j, np.linalg.norm(np.array(agent_i.position) - np.array(agent_j.position)))
                         for j, agent_j in enumerate(U) if i != j]
            distances.sort(key=lambda x: x[1])
            
            # Connect to 2-3 nearest neighbors within sensing range
            for j, dist in distances[:3]:
                if dist <= sensing_range:
                    E.add((i, j))
                    E.add((j, i))  # Undirected graph, so add both directions

        # Assign neighbors based on the graph
        for i, agent in enumerate(U):
            agent.neighbors = [U[j] for j in range(len(U)) if (i, j) in E]

        # Ensure the graph is connected
        def dfs(node, visited):
            visited.add(node)
            for neighbor in U[node].neighbors:
                neighbor_index = U.index(neighbor)
                if neighbor_index not in visited:
                    dfs(neighbor_index, visited)

        visited = set()
        dfs(0, visited)

        if len(visited) != len(U):
            print("Warning: The graph is not connected. Adjusting connections...")
            # Connect disconnected components
            unvisited = set(range(len(U))) - visited
            while unvisited:
                unvisited_node = unvisited.pop()
                closest_visited = min(visited, key=lambda x: np.linalg.norm(np.array(U[x].position) - np.array(U[unvisited_node].position)))
                E.add((unvisited_node, closest_visited))
                E.add((closest_visited, unvisited_node))
                U[unvisited_node].neighbors.append(U[closest_visited])
                U[closest_visited].neighbors.append(U[unvisited_node])
                dfs(unvisited_node, visited)


        if group_size > 2:        
        # Ensure each agent has 2-3 neighbors
            for agent in U:
                if len(agent.neighbors) < 2:
                    # Add nearest non-neighbor as a neighbor
                    non_neighbors = [a for a in U if a not in agent.neighbors and a != agent]
                    nearest_non_neighbor = min(non_neighbors, key=lambda x: np.linalg.norm(np.array(agent.position) - np.array(x.position)))
                    agent.neighbors.append(nearest_non_neighbor)
                    nearest_non_neighbor.neighbors.append(agent)
                elif len(agent.neighbors) > 3:
                    # Remove furthest neighbor
                    furthest_neighbor = max(agent.neighbors, key=lambda x: np.linalg.norm(np.array(agent.position) - np.array(x.position)))
                    agent.neighbors.remove(furthest_neighbor)
                    furthest_neighbor.neighbors.remove(agent)

        # Remove duplicates from neighbor lists
        for vehicle in gp.group:
            vehicle.neighbors = list(set(vehicle.neighbors))
        
                # Print each agent's neighbors
        print("Agent Neighbors:")
        for i, agent in enumerate(U):
            neighbor_indices = [U.index(neighbor) for neighbor in agent.neighbors]
            print(f"Agent {agent.id}: Neighbors {neighbor_indices}")
        print()  # Add a blank line for readability
        
        #Plot the graph of vehicle relationships
        # self.plot_vehicle_graph(gp.group)     
        return gp
    
    
    def plot_vehicle_graph(self, group):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot vehicles as nodes
        for vehicle in group:
            ax.plot(vehicle.x, vehicle.y, 'ro')
            ax.annotate(f'V{vehicle.id}', (vehicle.x, vehicle.y), xytext=(5, 5), textcoords='offset points')
        
        # Plot edges between neighbors
        for vehicle in group:
            for neighbor in vehicle.neighbors:
                ax.plot([vehicle.x, neighbor.x], [vehicle.y, neighbor.y], 'b-', alpha=0.5)
        
        ax.set_title('Vehicle Relationship Graph')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.grid(True)
        plt.show()
        

    ###############################################################################
    def velParallelSUSD(self,vehicles):
        """
        Calculate the parallel velocity component for SUSD algorithm in 2D.

        Parameters
        ----------
        vehicles : group
            The group object containing the current baseline and other vehicles.

        Returns
        -------
        numpy.ndarray
            2D parallel velocity vector
        """
        kp = 1  # Constant for parallel velocity calculation
        a0 = 20.0  # Desired inter-agent distance

        r_i = self.position[0:2]  # Current vehicle position
        v_para = np.zeros(2)  # Initialize parallel velocity component

        for neighbor in self.neighbors:
            r_j = neighbor.position[0:2]  # Neighbor position
            r_ij = r_j - r_i  # Vector from current vehicle to neighbor
            r_ij_norm = np.linalg.norm(r_ij)
            if r_ij_norm > 0:
                r_ij_unit = r_ij / r_ij_norm
                v_para += kp * (r_ij_norm - a0) * r_ij_unit

        return v_para
    #######################################################################

    def velPerpendicularSUSD(self, vehicles, pollution):
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
        
        k = 5000  # Attraction constant
        C = 15  # Constant offset

        y_r_i = pollution.calculate_concentration(self.position)  # Concentration measurement for current vehicle
        print(f"Robot: {self.id} Concentration measurement for current vehicle: {y_r_i}")
        

        # Calculate the unit vecotr n (n is perpendicular to q)
        n = self.calculate_perpendicular_vector(pollution)
        n_unit = n / np.linalg.norm(n)

        # Calculate the perpendicular velocity magnitude
        y_min = 0.2e-5
        y_max = 1.1e-5
        y_normalized = (y_r_i - y_min) / (y_max - y_min)
        y_normalized = y_normalized ** 3  # Squaring to increase the effect at higher concentrations
        v_perp_mag = k * y_r_i + C
        
        # Calculate the projection of v_perp onto n
        v_perp_n = np.multiply(v_perp_mag, n_unit)

        return v_perp_n
    
    def calculate_perpendicular_vector(self, pollution):
        n_sum = np.zeros(2)
        for neighbor in self.neighbors:
            # Calculate direction vector from self to neighbor
            n = np.array(neighbor.position[:2]) - np.array(self.position[:2])
            n = n / np.linalg.norm(n)  # Normalize
            
            # Get concentrations
            my_conc = pollution.calculate_concentration(self.position)
            neighbor_conc = pollution.calculate_concentration(neighbor.position)
            
            # Point n towards higher concentration
            if my_conc > neighbor_conc:
                n = -n
                
            n_sum += n
            
            
        # Normalize final direction
        if np.linalg.norm(n_sum) > 0:
            n_sum = n_sum / np.linalg.norm(n_sum)
            
        return n_sum

class group:
    def __init__(self, group_size):
        self.group_size = group_size
        self.group = []
        self.velocity = []
        self.current_baseline = [-1,0]

    def calculate_baseline(self):
        # Calculate the common virtual baseline for the group
        q_prev = np.array(self.current_baseline)
        q_sum = np.zeros(2)
        for i in self.group:
            q_i = np.zeros(2)
            for neighbor in i.neighbors:
                    q0_ij = (np.array(neighbor.position[:2]) - np.array(i.position[:2]))
                    q0_ij = q0_ij / np.linalg.norm(q0_ij)  # Normalize to get unit vector
                    q0_ij = q0_ij.astype(float)  # Ensure q0_ij is float
                    
                    # Determine qi,j based on the dot product with previous baseline
                    if np.dot(q0_ij, q_prev) >= 0:
                        q_ij = q0_ij
                    else:
                        q_ij = -q0_ij
                    
                    q_i += q_ij

            # Normalize q_i
            if np.linalg.norm(q_i) > 0:
                q_i /= np.linalg.norm(q_i)
            
            q_sum += q_i
        
        # Normalize q_sum to get the new baseline unit vector
        if np.linalg.norm(q_sum) > 0:
            self.current_baseline = q_sum / np.linalg.norm(q_sum)
        else:
            # If q_sum is zero, keep the previous baseline
            self.current_baseline = q_prev
            
        self.update_neighbors()       
        return self.current_baseline
    
    def update_neighbors(self):
        # Iterate through all vehicles
        for i, vehicle in enumerate(self.group):
            # Calculate distances to all other vehicles
            distances = []
            for j, other in enumerate(self.group):
                if i != j:  # Don't include self
                    dist = np.linalg.norm(np.array(vehicle.position[:2]) - np.array(other.position[:2]))
                    distances.append((dist, other))
            
            # Sort by distance and take 2 closest
            distances.sort(key=lambda x: x[0])
            vehicle.neighbors = set()
            for k in range(min(2, len(distances))):  # Take up to 2 closest neighbors
                vehicle.neighbors.add(distances[k][1])
        # Print neighbor information for each vehicle
        print("\nNeighbor Information:")
        for vehicle in self.group:
            neighbor_ids = [n.id for n in vehicle.neighbors]
            print(f"Vehicle {vehicle.id} has neighbors: {neighbor_ids}")
        
  
  
  
        
'''
    def update_neighbors(self):
        # Get the current baseline vector
        q = np.array(self.current_baseline)
        
        # Calculate projections of all agents onto q
        projections = []
        for i, agent in enumerate(self.group):
            r_i = np.array(agent.position[:2])
            r_parallel = np.dot(r_i, q) * q
            projections.append((i, r_parallel))
        
        # Sort agents based on their projections
        sorted_agents = sorted(projections, key=lambda x: np.linalg.norm(x[1]))
        
        # Update neighbors for each agent
        for i, agent in enumerate(self.group):
            agent_index = [x[0] for x in sorted_agents].index(i)
            
            # Initialize neighbors set
            agent.neighbors = set()
            
            # Add left neighbor if not leftmost
            if agent_index > 0:
                agent.neighbors.add(self.group[sorted_agents[agent_index - 1][0]])
            
            # Add right neighbor if not rightmost
            if agent_index < len(self.group) - 1:
                agent.neighbors.add(self.group[sorted_agents[agent_index + 1][0]])
'''