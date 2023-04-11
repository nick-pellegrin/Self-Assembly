import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


num_agents = 20 # number of agents
num_iterations = 2000 # number of iterations
target_shape = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) # the target shape the agents should form
target_shape *= 10 # scale up the target shape for better visualization
noise = 0.1 # amount of noise added to the agent movements

def distance(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)


network = Sequential()
network.add(Dense(64, input_dim=num_agents * 4, activation='relu'))
network.add(Dense(32, activation='relu'))
network.add(Dense(num_agents * 2, activation=None))

positions = np.random.rand(num_agents, 2) * 10
velocities = np.random.randn(num_agents, 2) * 0.1

network.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))

for i in range(num_iterations):
    # Concatenate the positions and velocities into a single input array
    input_data = np.concatenate((positions.flatten(), velocities.flatten()))
    
    # Reshape the input array to have the correct dimensions
    input_data = input_data.reshape((1, num_agents * 4))
    
    # Compute the output of the neural network
    output_data = network.predict(input_data)[0]
    
    # Reshape the output array to have the correct dimensions
    output_data = output_data.reshape((num_agents, 2))
    
    # Update the velocities of the agents based on the output of the neural network
    velocities += output_data
    
    # Normalize the velocities to a maximum speed of 0.1
    velocities = velocities / np.linalg.norm(velocities, axis=1, keepdims=True) * 0.1
    
    # Add some noise to the agent movements
    positions += velocities + np.random.randn(num_agents, 2) * noise
    
    # Enforce the boundary conditions (wrap around the edges)
    positions = np.fmod(positions + 10, 10)
    
    # Compute the distances between all pairs of agents
    distances = distance(positions[:, np.newaxis], positions[np.newaxis, :])
    
    # Compute the matrix of attractive forces
    attractive_forces = np.zeros((num_agents, 2))
    for j in range(4):
        attractive_forces += (positions - target_shape[j]) * (distances[:, j] ** 2)[:, np.newaxis]
        
    # Compute the matrix of repulsive forces
    repulsive_forces = np.zeros((num_agents, 2))
    for j in range(num_agents):
        repulsive_forces[j] = np.sum((positions[j] - positions) / (distances[j] ** 2))[:, np.newaxis]