import sys
sys.path.append('/Users/zhangyichi/opt/anaconda3/lib/python3.9/site-packages')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import convolve


def compute_depth_gradient(depth_data):
    """Compute the gradient of the depth data using a Sobel operator."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve(depth_data, sobel_x, mode='constant')
    grad_y = convolve(depth_data, sobel_y, mode='constant')

    return grad_x, grad_y


def generate_direction_vectors():
    """Generate 24 evenly spaced direction vectors."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    direction_vectors = np.vstack([np.cos(angles), np.sin(angles)]).T
    return direction_vectors


def get_new_position_and_direction(current_pos, current_dir, k, move_length, grad_x, grad_y, direction_vectors):
    """Calculate the new position and direction based on the current position and direction, k value, and depth data gradients."""

    # Get the gradient at the current position
    grad_at_pos = np.array(
        [grad_x[int(current_pos[1]), int(current_pos[0])], grad_y[int(current_pos[1]), int(current_pos[0])]])

    # Get the perpendicular direction to the gradient (i.e., the isobath direction)
    isobath_direction = np.array([-grad_at_pos[1], grad_at_pos[0]])
    isobath_direction /= np.linalg.norm(isobath_direction)

    # Get the new direction as a weighted sum of the current direction and the isobath direction
    new_direction = k * current_dir + (1 - k) * isobath_direction
    new_direction /= np.linalg.norm(new_direction)

    # Get the new position
    new_pos = current_pos + move_length * new_direction

    # Check if the new position is near the boundaries, and if so, adjust the direction towards the center
    buffer_zone = 100
    center_pos = np.array([depth_data.shape[1] / 2, depth_data.shape[0] / 2])
    if np.any(new_pos < buffer_zone) or np.any(new_pos > (np.array(grad_x.shape[::-1]) - buffer_zone)):
        direction_to_center = center_pos - current_pos
        direction_to_center /= np.linalg.norm(direction_to_center)
        new_direction = 0.7 * new_direction + 0.3 * direction_to_center
        new_pos = current_pos + move_length * new_direction

    # If the new position is outside the bounds, choose a random direction that stays within the bounds
    if new_pos[0] < 0 or new_pos[0] >= grad_x.shape[1] or new_pos[1] < 0 or new_pos[1] >= grad_x.shape[0]:
        # Generate random directions until a valid one is found
        while new_pos[0] < 0 or new_pos[0] >= grad_x.shape[1] or new_pos[1] < 0 or new_pos[1] >= grad_x.shape[0]:
            random_direction = direction_vectors[np.random.randint(0, len(direction_vectors))]
            new_pos = current_pos + move_length * random_direction

    # Get the closest direction vector to the new direction
    closest_dir_idx = np.argmin(np.linalg.norm(direction_vectors - new_direction, axis=1))
    new_dir = direction_vectors[closest_dir_idx]

    return new_pos, new_dir


def generate_path(depth_data, grad_x, grad_y, direction_vectors, k_range=(0.0, 1.0), N_range=(10, 200),
                  move_length=100 / 1852 / 0.004):
    """Generate a path based on the depth data and specified parameters."""

    # Randomly select the initial values for k, N, position and direction
    k = np.random.uniform(k_range[0], k_range[1])
    N = np.random.randint(N_range[0], N_range[1] + 1)
    init_pos = np.array([400,500])
    init_dir_idx = np.random.randint(0, len(direction_vectors))

    # Initialize the list to store the path
    path = [init_pos]

    # Get the initial direction vector
    current_dir = direction_vectors[init_dir_idx]

    # Generate the path
    for i in range(N):
        # Get the new position and direction
        new_pos, new_dir = get_new_position_and_direction(path[-1], current_dir, k, move_length, grad_x, grad_y,
                                                          direction_vectors)

        # Add the new position to the path
        path.append(new_pos)

        # Update the current direction
        current_dir = new_dir

        # Slightly increase k in each iteration (but not beyond the maximum value in k_range)
        k = min(k + np.random.rand() * 0.01, k_range[1])

    return np.array(path)


def plot_paths(depth_data, paths):
    """Plot the generated paths on the depth data."""

    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    # Plot the depth data and paths
    for i, path in enumerate(paths):
        ax = axes[i]

        # Plot the depth data as a heatmap
        c = ax.imshow(depth_data, cmap='viridis', extent=[0, depth_data.shape[1], 0, depth_data.shape[0]])

        # Plot the path
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)

        # Plot the start and end points
        ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'ro', label='End')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {10*(i+1)}')

        # Add a legend
        ax.legend()

        # Add a colorbar
        fig.colorbar(c, ax=ax)

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# Load the depth data
depth_data = pd.read_excel('/Users/zhangyichi/Downloads/refined_depth_data.xlsx').values

# Compute the gradient of the depth data
grad_x, grad_y = compute_depth_gradient(depth_data)

# Generate the direction vectors
direction_vectors = generate_direction_vectors()

# Define the parameter ranges for each plot
params = [
    {"k_range": (0.0, 0.25), "N_range": (1000, 1500)},
    {"k_range": (0.2, 0.5), "N_range": (1000, 1500)},
    {"k_range": (0.3, 0.7), "N_range": (1000, 1500)},
    {"k_range": (0.5, 0.9), "N_range": (1000, 1500)},
]

# Generate four paths with the specified parameters
paths = [generate_path(depth_data, grad_x, grad_y, direction_vectors, **param, move_length=70 * 5 / 1852 / 0.004) for
         param in params]

# Plot the paths on the depth data
plot_paths(depth_data, paths)
