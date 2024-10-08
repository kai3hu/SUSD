import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


NPFltArr = NDArray[np.float64]

class pollution:
    """
    A class to represent the polluted source using a Gaussian plume model.

    Attributes
    ----------
    x : float
        x-coordinate of the pollution source
    y : float
        y-coordinate of the pollution source
    z : float
        z-coordinate of the pollution source, equal to source height
    Q : float
        Source strength (emission rate) in g/s
    u : float
        Wind speed in m/s
    v : float
        Wind direction in radians
    hs : float
        Source height in meters

    Methods
    -------
    __call__(x, y, z)
        Calculate the concentration at given (x, y, z) coordinates
    display2D(size, resolution)
        Display 2D image of pollution concentration.
    display3D(size, resolution)
        Display 3D representation of pollution concentration.

    Notes
    -----
    - Q is the source strength (emission rate)
    - u is the wind speed
    - σy and σz are the dispersion coefficients
    - He is the effective release height
    - y' is the perpendicular distance from the plume centerline

    The pollution plume is simulated using a basic Gaussian fluid mechanics model.
    Assuming wind along the x-axis, the pollutant concentration C at any point (x, y, z) is given by:

    C(x, y, z) = (Q / (2πuσyσz)) * exp(-y²/(2σy²)) * [exp(-(z-He)²/(2σz²)) + exp(-(z+He)²/(2σz²))]

    Where:
    - He = hs + Δh(x) is the effective release height
    - Δh(x) = 2.126 × 10⁻⁴ · x^(2/3) is the plume elevation
    - σy = 1.36 · |x|^0.82 is the horizontal dispersion coefficient
    - σz = 0.275 · |x|^0.69 is the vertical dispersion coefficient

    //GQ 09/2024
    """

    ## Constructor ===========================================================#
    def __init__(self, x=0.0, y=0.0, Q=1.59, hs=30.0):
        """
        Initialize the Pollution source with given parameters.

        Parameters
        ----------
        x : float, optional
            x-coordinate of the pollution source (default is 0.0)
        y : float, optional
            y-coordinate of the pollution source (default is 0.0)
        Q : float, optional
            Source strength (emission rate) in g/s (default is 1.59)
        hs : float, optional
            Source height in meters (default is 30.0)
        """
        self.x = x
        self.y = y
        self.z = hs
        self.Q = Q
        self.hs = 30.0
        self.u = np.random.uniform(1, 10.0)  # Random wind speed between 1 and 10 m/s
        self.v = np.random.uniform(-np.pi/4, np.pi/4)  # Random wind direction in radians

    ## Methods ================================================================#
    def __call__(self, x: NPFltArr, y: NPFltArr, z: NPFltArr) -> NPFltArr:
        """
        Calculate the concentration at given (x, y, z) coordinates.

        Parameters
        ----------
        x : NPFltArr
            Array of x-coordinates
        y : NPFltArr
            Array of y-coordinates
        z : NPFltArr
            Array of z-coordinates

        Returns
        -------
        NPFltArr
            Array of concentration values
        """
        
        # Rotate coordinates based on wind direction
        x_rot = x * np.cos(self.v) + y * np.sin(self.v)
        y_rot = -x * np.sin(self.v) + y * np.cos(self.v)

        # Calculate effective release height
        delta_h = 2.126e-4 * np.abs(x_rot)**(2/3)
        He = self.hs + delta_h

        # Calculate dispersion coefficients
        sigma_y = 1.36 * np.abs(x_rot)**0.82
        sigma_z = 0.275 * np.abs(x_rot)**0.69

        # Calculate concentration
        coeff = self.Q / (2 * np.pi * self.u * sigma_y * sigma_z)
        exp_y = np.exp(-y_rot**2 / (2 * sigma_y**2))
        exp_z = np.exp(-(z - He)**2 / (2 * sigma_z**2)) + np.exp(-(z + He)**2 / (2 * sigma_z**2))
        


        # Set concentration to 0 for opposite direction of wind
        concentration = np.where(x_rot >= 0, coeff * exp_y * exp_z, 0)
        
        return concentration

    def display2D(self, size=500, resolution=100):
        """
        2D representation of pollution concentration with continuous color gradient at z = 0.
    
        Parameters
        ----------
        size : int, optional
            Size of the area to display (default is 500)
        resolution : int, optional
            Number of points to sample in each dimension (default is 80)
        """
        x = np.linspace(-1, size, resolution)
        y = np.linspace(-1, size, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, 0)  # Set Z to 30 for all points
        C = self(X, Y, Z)
    
        # Normalize concentration for color mapping
        C_normalized = (C - np.min(C)) / (np.max(C) - np.min(C))
    
        # Create a figure
        fig, ax = plt.subplots(figsize=(14, 12))
    
        # Set up the colormap
        scalarMap = plt.cm.ScalarMappable(cmap='viridis_r')
        scalarMap.set_array(C)
    
        # Create a contour plot
        contour = ax.contourf(X, Y, C, levels=50, cmap='viridis_r')
    
        # Highlight the pollution source
        ax.scatter(self.x, self.y, color='red', s=100, marker='*', label='Pollution Source')
    
        ax.set_title(f'2D Pollution Concentration at Z = 30m\nWind Speed: {self.u:.2f} m/s, Direction: {np.degrees(self.v):.2f}°', fontsize=16)
        ax.set_xlabel('X coordinate (m)', fontsize=12)
        ax.set_ylabel('Y coordinate (m)', fontsize=12)
    
        # Add a colorbar
        cbar = plt.colorbar(scalarMap, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Concentration', fontsize=12)
    
        # Add legend
        ax.legend()
    
        plt.show()
    def display3D(self, size=1000, resolution=80):
        """
        Optimized 3D representation of pollution concentration with continuous color gradient.

        Parameters
        ----------
        size : int, optional
            Size of the area to display (default is 1000)
        resolution : int, optional
            Number of points to sample in each dimension (default is 50)
        """
        x = np.linspace(0, size, resolution)
        y = np.linspace(-size/2, size/2, resolution)
        z = np.linspace(0, 100, resolution)  # Increased z range
        X, Y, Z = np.meshgrid(x, y, z)
        C = self(X, Y, Z)

        # Normalize concentration for color mapping
        C_normalized = (C - np.min(C)) / (np.max(C) - np.min(C))

        # Create a figure with 3D axes
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Set up the colormap once
        scalarMap = plt.cm.ScalarMappable(cmap='viridis_r')
        scalarMap.set_array(C)

        # Loop through layers once, and avoid redundant operations
        for i in range(resolution):
            # Precompute the color map for the current slice
            color = scalarMap.to_rgba(C[:,:,i])
            # Calculate alpha (transparency) once
            alpha = np.where(C[:,:,i] == 0, 0, 0.07 + 0.93 * (C_normalized[:,:,i]))
            color[:,:,3] = alpha

            # Create a surface plot for each layer
            ax.plot_surface(X[:,:,i], Y[:,:,i], Z[:,:,i], facecolors=color,
                            rstride=1, cstride=1, shade=False, antialiased=False)
        

        
        # Highlight the pollution source
        ax.scatter(self.x, self.y, self.hs, color='red', s=100, marker='*', label='Pollution Source')

        ax.set_title(f'3D Pollution Concentration\nWind Speed: {self.u:.2f} m/s, Direction: {np.degrees(self.v):.2f}°', fontsize=16)
        ax.set_xlabel('X coordinate (m)', fontsize=12)
        ax.set_ylabel('Y coordinate (m)', fontsize=12)
        ax.set_zlabel('Z coordinate (m)', fontsize=12)

        # Add a colorbar
        cbar = plt.colorbar(scalarMap, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Concentration', fontsize=12)

        # Add legend
        ax.legend()

    def calculate_concentration_gradient(self, eta: NPFltArr[float]) -> NPFltArr[float]:
        """
        Calculate the concentration gradient at a given point in 2D.

        Parameters
        ----------
        eta : NPFltArr
            Array of [x, y] coordinates

        Returns
        -------
        NPFltArr
            Array of concentration gradient values [dx, dy] pointing towards highest concentration (pollution source)
        """
        x, y = eta

        # Rotate coordinates based on wind direction
        x_rot = x * np.cos(self.v) + y * np.sin(self.v)
        y_rot = -x * np.sin(self.v) + y * np.cos(self.v)
        # Calculate effective release height
        delta_h = 2.126e-4 * np.abs(x_rot)**(2/3)
        He = self.hs + delta_h

        # Calculate dispersion coefficients
        sigma_y = 1.36 * np.abs(x_rot)**0.82

        # Calculate concentration
        coeff = self.Q / (2 * np.pi * self.u * sigma_y)
        exp_y = np.exp(-y_rot**2 / (2 * sigma_y**2))
        
        concentration = coeff * exp_y
    
        # Calculate gradients
        dC_dx_rot = concentration * (
            -1 / x_rot + 
            y_rot**2 * 0.82 / (x_rot * sigma_y**2)
        )
        dC_dy_rot = -concentration * y_rot / sigma_y**2

        # Rotate gradients back to original coordinate system
        dC_dx = dC_dx_rot * np.cos(self.v) - dC_dy_rot * np.sin(self.v)
        dC_dy = dC_dx_rot * np.sin(self.v) + dC_dy_rot * np.cos(self.v)

        # Calculate gradient pointing towards the pollution source (local maximum)
        gradient = np.array([dC_dx, dC_dy])
        
        # Normalize the gradient
        gradient_magnitude = np.linalg.norm(gradient)
        if gradient_magnitude > 0:
            gradient = gradient / gradient_magnitude
        
        print(f"Gradient at point ({x}, {y}): {gradient}")
        return gradient

