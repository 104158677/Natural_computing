import numpy as np
import itertools
import math
import matplotlib.pyplot as plt

def rastrigin(pos,dim):
    return 10 * dim + sum([xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in pos])


class Particle:  # all the material that is relavant at the level of the individual particles

    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim

        self.fitness = rastrigin(self.position, dim)
        self.best_particle_fitness = self.fitness  # we could start with very large number here,
        # but the actual value is better in case we are lucky

    def setPos(self, pos):
        self.position = pos
        self.fitness = rastrigin(self.position, self.dim)
        if self.fitness < self.best_particle_fitness:  # to update the personal best both
            # position (for velocity update) and
            # fitness (the new standard) are needed
            # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
        # Here we use the canonical version
        # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size=self.dim)
        r2 = np.random.uniform(low=0, high=1, size=self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
        # the next line is the main equation, namely the velocity update,
        # the velocities are added to the positions at swarm level
        new_vel = inertia * cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)
        self.velocity = new_vel
        return new_vel


class PSO:  # all the material that is relavant at swarm level

    def __init__(self, w, a1, a2, dim, population_size, time_steps, search_range):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.w = w  # Inertia
        self.a1 = a1  # Attraction to personal best
        self.a2 = a2  # Attraction to global best
        self.dim = dim

        self.swarm = [Particle(dim, -search_range, search_range) for i in range(population_size)]
        self.time_steps = time_steps
        #print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                new_position = particle.position + particle.updateVel(self.w, self.a1, self.a2,
                                                                      particle.best_particle_pos, self.best_swarm_pos)

                if new_position @ new_position > 1.0e+18:  # The search will be terminated if the distance
                    # of any particle from center is too large
                    print('Time:', t, 'Best Pos:', self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')

                self.swarm[p].setPos(new_position)

                new_fitness = rastrigin(new_position, self.dim)

                if new_fitness < self.best_swarm_fitness:  # to update the global best both
                    # position (for velocity update) and
                    # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            if self.best_swarm_fitness == 0:  # we print only two components even it search space is high-dimensional
                # print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                # t, self.best_swarm_fitness, self.best_swarm_pos[0], self.best_swarm_pos[1]), end=" ")
                # if self.dim > 2:
                #     print('...')
                # else:
                #     print('')
                return t
        return 10000


# avgs = []
# populations = range(1,100,1)
avgs = []
#populations = [3,5,10,20,30,40,50,70,100,200,300,400,500,700,1000]
populations = range(5,105,5)
for i in populations:
    times = []
    for _ in range(10):
        pso = PSO(dim=4, w=0.7, a1=2.02, a2=2.02, population_size=i, time_steps=1001, search_range=5.12)
        times.append(i * pso.run())
        avg = np.average(times)
    avgs.append(avg)
    print(i,avg)
fig_size = (8, 4) # Set figure size in inches (width, height)
fig = plt.figure(figsize=fig_size) # Create a new figure object
ax = fig.add_subplot(1, 1, 1) # Add a single axes to the figure
# Plot lines giving each a label for the legend and setting line width to 2
ax.plot(populations, avgs,'ro-', linewidth=2)
ax.set_xlabel('Population Size', fontsize=12)
ax.set_ylabel('Runtime', fontsize=12)
# for i, j in zip(populations,avgs):
#     ax.annotate(str(j),xy = (i, j))
plt.title('PSO_Rastrigin_Dim_4')
ax.grid('on') # Turn axes grid on
fig.tight_layout() # This minimises whitespace around the axes.
fig.savefig('pop_size_rastrigin_dim_4.png') # Save figure to current directory in png format
plt.show()
# times = []
# for _ in range(10):
#     pso = PSO(dim=2, w=0.7, a1=2.02, a2=2.02, population_size=25, time_steps=1001, search_range=5.12)
#     times.append(pso.run())
# print(np.average(times))