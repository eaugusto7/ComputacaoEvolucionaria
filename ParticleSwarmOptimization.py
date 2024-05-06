import random
import math
import time

# Function Langermann
def langermann(xx):
    d = len(xx)
    m = 5
    c = [1, 2, 5, 2, 3]
    A = [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]

    outer = 0.0
    for i in range(m):
        inner = 0.0
        for j in range(d):
            inner += (xx[j] - A[i][j]) ** 2
        newTerm = c[i] * math.exp(-inner / math.pi) * math.cos(math.pi * inner)
        outer += newTerm
    return outer

# Function DeJong N5
def dejong5(xx):
    outer = 0.0

    a1 = [-32.0, -16.0, 0.0, 16.0, 32.0,
          -32.0, -16.0, 0.0, 16.0, 32.0,
          -32.0, -16.0, 0.0, 16.0, 32.0,
          -32.0, -16.0, 0.0, 16.0, 32.0,
          -32.0, -16.0, 0.0, 16.0, 32.0]

    a2 = [-32.0, -32.0, -32.0, -32.0, -32.0,
          -16.0, -16.0, -16.0, -16.0, -16.0,
          0.0, 0.0, 0.0, 0.0, 0.0,
          16.0, 16.0, 16.0, 16.0, 16.0,
          32.0, 32.0, 32.0, 32.0, 32.0]

    sum_val = 0.002
    for i in range(1, 26):
        sum_val += 1 / (i + math.pow((xx[0] - a1[i - 1]), 6) + math.pow((xx[1] - a2[i - 1]), 6))

    outer = 1.0 / sum_val
    return outer

# Filter Settings
num_particles = 100
num_dimensions = 2
max_iterations = 10000
w = 0.7
c1 = 2.05
c2 = 2.05
tolerance = 1e-4

random.seed(int(time.time() * 1e9))

# Using Langermann Function
#particles = [[random.uniform(0, 10) 
#              for _ in range(num_dimensions)] 
#              for _ in range(num_particles)]

# Using DeJong N5 Function
particles = [[random.uniform(-65536.0, 65536.0) 
              for _ in range(num_dimensions)] 
              for _ in range(num_particles)]

velocities = [[0 
               for _ in range(num_dimensions)] 
               for _ in range(num_particles)]

best_local_positions = particles[:]
best_global_position = particles[0]

for iteration in range(max_iterations):
    for i in range(num_particles):
        particle = particles[i]
        velocity = velocities[i]
        fitness_current = dejong5(particle)

        # Update Local Best
        if fitness_current < dejong5(best_local_positions[i]):
            best_local_positions[i] = particle[:]

        # Update Global Best
        if fitness_current < dejong5(best_global_position):
            best_global_position = particle[:]

        for j in range(num_dimensions):
            random.seed(int(time.time() * 1e9))
            r1 = random.random()

            random.seed(int(time.time() * 1e9))
            r2 = random.random()
            velocities[i][j] = w * velocity[j] + c1 * r1 * (best_local_positions[i][j] - particle[j]) + c2 * r2 * (best_global_position[j] - particle[j])

        for j in range(num_dimensions):
            particles[i][j] += velocities[i][j]

#print("Best Position:", best_global_position)
#print("Value of Best Position:", langermann(best_global_position))
print(dejong5(best_global_position))