from __future__ import print_function
import multiprocessing
import os
import pickle
import neat
import pandas
import numpy as np
import sklearn.preprocessing as preprocessing
import SimMarket

dataframe = pandas.read_excel("prunedxrp.xlsx")
arr = np.delete(dataframe.to_numpy(),[0,1,2,3],1)
arr = np.delete(arr, [range(0,1000)], 0)
maxabs = preprocessing.MaxAbsScaler()
scaledinput = maxabs.fit_transform(arr)
invertedinput = maxabs.inverse_transform(scaledinput)
pricesarr = invertedinput[:,5]

inputs = scaledinput

runs_per_net = 1

time_const = 1

def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)
    fitnesses = []
    
    for runs in range(runs_per_net):
        sim = SimMarket.SimMarket(500, 0, inputs, pricesarr)
        net.reset()
        fitness = 0
        while sim.row < len(arr):
            input = sim.getInputs()
            output = net.advance(input, time_const, time_const)[0]
            
            sim.step(output)
            fitness = sim.acctvalue
            if sim.failstate == True:
                break
        fitnesses.append(fitness)
    return max(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome,config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    # Create the population, which is the top-level object for a NEAT run.
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pe = neat.ParallelEvaluator(6, eval_genome)
    #pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)


    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)
    
    print(winner)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    run()