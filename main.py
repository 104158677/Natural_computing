import numpy as np
import random
import operator
import matplotlib.pyplot as plt

def generate_problem(size: int, max_num: int = 10, deletion_rate: float = 1/3):
    matrix = np.random.randint(1, max_num, size=(size, size))
    solution = np.zeros((size,size))
    for i in range(size):
        solution[i] = (random.choices([0, 1], weights=[deletion_rate, 1 - deletion_rate], k = size))
    row_sum = np.zeros(size)
    col_sum = np.zeros(size)
    for i in range(size):
        row_sum[i] = (np.matmul(matrix[i], np.transpose(solution[i])))
        col_sum[i] = (np.matmul(matrix[:,i], np.transpose(solution[:,i])))
    return solution.reshape(1,size ** 2)[0]

def generate_random_solution(size: int, pop):
    population = []
    for i in range(pop):
        population.append(random.choices([0, 1], weights=[0.5, 0.5], k=size**2))
    return population

def fitness_binary(candidate: list, solution: list):
    for i in range(len(candidate)):
        if candidate[i] != solution[i]:
            return 0
    return 1

def fitness_prop(candidate: list, solution: list):
    fittotal = 0
    for i in range(len(candidate)):
        if candidate[i] == solution[i]:
            fittotal += 1
    return fittotal/len(candidate)

def ranking_pop(population, fitness_type, solution):
    if fitness_type != "fitness_prop" or "fitness_binary":
        assert "fitness type must be either fitness_prop or fitness_binary!"
    fitnesses = {}
    sum_fit = 0.0
    if fitness_type == "fitness_prop":
        for i in range(len(population)):
            fitnesses[i] = fitness_prop(population[i],solution)
            sum_fit += fitnesses[i]
    if fitness_type == "fitness_binary":
        for i in range(len(population)):
            fitnesses[i] = fitness_binary(population[i],solution)
            sum_fit += fitnesses[i]
    return sorted(fitnesses.items(), key=operator.itemgetter(1),reverse=True), sum_fit

def selection(ranked, tot, elitism):
    probabilities = np.zeros(len(ranked))
    members = np.arange(len(ranked))
    size = len(ranked)
    elite_members = []
    for i in range(len(ranked)):
        x,y = ranked[i][0], ranked[i][1]
        if i < elitism:
            elite_members.append(x)
        probabilities[x] = y
    probabilities=probabilities/np.sum(probabilities)
    selected_members = np.random.choice(members, size-elitism, p=probabilities)
    return selected_members, elite_members


def mating_pool(population, selected_members, elite_members):
    mating_pool = []
    elite = []
    for i in range(len(selected_members)):
        index = selected_members[i]
        mating_pool.append(population[index])

    for i in range(len(elite_members)):
        index = elite_members[i]
        elite.append(population[index])
    return mating_pool, elite


def breed_from_parents(parent1, parent2):
    geneA = int(np.random.rand() * len(parent1))
    geneB = int(np.random.rand() * len(parent1))

    while geneB == geneA:
        geneB = int(np.random.rand() * len(parent1))

    start = min(geneA, geneB)
    end = max(geneA, geneB)

    child = []
    for i in range(len(parent1)):
        if i >= start and i < end:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child


def new_population(mating_pool, elite_pool):
    children = []
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(len(elite_pool)):
        children.append(elite_pool[i])
    for i in range(len(mating_pool)):
        child = breed_from_parents(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)
    return children

def mutate(chromosome, prob_mut):
    for mutable in range(len(chromosome)):
        if (np.random.rand()<prob_mut):
            chromosome[mutable]=1-chromosome[mutable]
    return chromosome

def mutation_over_pop(population, prob_mut):
    mutated_pop = []
    for i in range(len(population)):
        mutated_chromo = mutate(population[i], prob_mut)
        mutated_pop.append(mutated_chromo)
    return mutated_pop

def new_generation(current_gen, elitism, prob_mut, solution):
    rank, tot = ranking_pop(current_gen, "fitness_prop", solution)
    selected_members, elite_members = selection(rank, tot, elitism)
    mates, elites = mating_pool(current_gen, selected_members, elite_members)
    children = new_population(mates, elites)
    next_gen = mutation_over_pop(children, prob_mut)
    return next_gen

def genetic_algorithm(size, population, elitism, prob_mut, generations):
    solution= generate_problem(size)
    pop = generate_random_solution(size,population)
    #fitness_type = "fitness_binary"
    fitness_type = "fitness_prop"
    ranked, tot = ranking_pop(pop, fitness_type, solution)
    print("Initial best fitness: {}".format(ranked[0][1]))

    for i in range(generations):
        pop = new_generation(pop, elitism, prob_mut,solution)
        ranked, tot = ranking_pop(pop, fitness_type, solution)
        if ranked[0][1] == 1:
            break;

    ranked, tot = ranking_pop(pop, fitness_type, solution)
    print("Final best fitness: {}".format(ranked[0][1]))

    return i

zz=genetic_algorithm(4, 50, 1, 0.01, 1000)
print("run time: ",zz)
