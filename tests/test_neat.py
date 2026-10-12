import random

import numpy as np

from marlio.agents.neat import N_INPUTS, N_OUTPUTS, Gene, Genome, NEATConfig, Population, run_genome
from marlio.agents.neat.genome import (
    MAX_NODES,
    Innovations,
    crossover,
    disjoint,
    initial_mutation_rates,
    link_mutate,
    mutate,
    node_mutate,
    same_species,
)
from marlio.agents.neat.network import Network
from marlio.env import make_marios_env

CONFIG = NEATConfig()
RIGHT = MAX_NODES + 5  # output neuron for the "right" button
A = MAX_NODES + 0  # output neuron for the "A" (jump) button
BIAS = N_INPUTS - 1


def genome_with(*genes):
    return Genome(genes=list(genes), max_neuron=N_INPUTS - 1, mutation_rates=initial_mutation_rates(CONFIG))


def test_bias_to_right_presses_right():
    network = Network(genome_with(Gene(BIAS, RIGHT, weight=1.0, innovation=1)), N_INPUTS, N_OUTPUTS)
    buttons = network.evaluate(np.zeros(N_INPUTS - 1))
    assert buttons == [0, 0, 0, 0, 0, 1]


def test_disabled_genes_are_ignored():
    network = Network(genome_with(Gene(BIAS, RIGHT, weight=1.0, enabled=False, innovation=1)), N_INPUTS, N_OUTPUTS)
    assert network.evaluate(np.zeros(N_INPUTS - 1)) == [0] * N_OUTPUTS


def test_input_cell_drives_jump():
    cell = 7
    network = Network(genome_with(Gene(cell, A, weight=-1.0, innovation=1)), N_INPUTS, N_OUTPUTS)
    inputs = np.zeros(N_INPUTS - 1)
    assert network.evaluate(inputs)[0] == 0
    inputs[cell] = -1  # an enemy in that cell
    assert network.evaluate(inputs)[0] == 1


def test_node_mutation_splits_a_connection():
    rng = random.Random(0)
    genome = genome_with(Gene(BIAS, RIGHT, weight=0.7, innovation=1))
    node_mutate(genome, Innovations(10), rng)
    assert not genome.genes[0].enabled
    first, second = genome.genes[1:]
    assert (first.into, first.out, first.weight) == (BIAS, N_INPUTS, 1.0)
    assert (second.into, second.out, second.weight) == (N_INPUTS, RIGHT, 0.7)
    # The split network still behaves like the original connection did.
    assert Network(genome, N_INPUTS, N_OUTPUTS).evaluate(np.zeros(N_INPUTS - 1))[5] == 1


def test_link_mutation_never_points_into_an_input():
    rng = random.Random(1)
    genome = genome_with()
    innovations = Innovations(N_OUTPUTS)
    for _ in range(200):
        link_mutate(genome, False, N_INPUTS, N_OUTPUTS, innovations, rng)
    assert genome.genes
    assert all(gene.out >= N_INPUTS for gene in genome.genes)
    assert len({(g.into, g.out) for g in genome.genes}) == len(genome.genes)


def test_crossover_keeps_fitter_parents_extra_genes():
    shared = Gene(BIAS, RIGHT, weight=1.0, innovation=1)
    fit = genome_with(shared, Gene(3, A, weight=0.5, innovation=2))
    fit.fitness = 100
    weak = genome_with(Gene(BIAS, RIGHT, weight=-1.0, innovation=1), Gene(4, A, innovation=3))
    weak.fitness = 10
    child = crossover(weak, fit, random.Random(0))
    assert sorted(g.innovation for g in child.genes) == [1, 2]


def test_speciation():
    a = genome_with(Gene(BIAS, RIGHT, weight=1.0, innovation=1))
    b = genome_with(Gene(BIAS, RIGHT, weight=1.1, innovation=1))
    c = genome_with(Gene(3, A, weight=1.0, innovation=2))
    assert disjoint(a.genes, c.genes) == 2.0
    assert same_species(a, b, CONFIG)
    assert not same_species(a, c, CONFIG)


def test_mutation_keeps_rates_positive_and_genes_valid():
    rng = random.Random(2)
    genome = genome_with()
    innovations = Innovations(N_OUTPUTS)
    for _ in range(50):
        mutate(genome, N_INPUTS, N_OUTPUTS, CONFIG, innovations, rng)
    assert all(rate > 0 for rate in genome.mutation_rates.values())
    neurons = {g.into for g in genome.genes} | {g.out for g in genome.genes}
    assert all(n < N_INPUTS or n <= genome.max_neuron or n >= MAX_NODES for n in neurons)


def test_generation_keeps_population_size_and_saves(tmp_path):
    population = Population(N_INPUTS, N_OUTPUTS, NEATConfig(population=30), seed=0)
    rng = random.Random(0)
    for _ in range(3):
        for genome in population.genomes():
            if genome.fitness == 0:
                genome.fitness = rng.uniform(-10, 300)
        population.new_generation()
        assert len(population.genomes()) == 30
    path = tmp_path / "population.json"
    population.save(path)
    loaded = Population.load(path)
    assert loaded.generation == 3
    assert len(loaded.genomes()) == 30
    assert loaded.innovations.last == population.innovations.last
    assert loaded.rng.random() == population.rng.random()


def test_run_genome_scores_like_marios():
    env = make_marios_env()
    # Standing still: no progress, the run times out quickly.
    still = run_genome(env, genome_with())
    # Holding right (bias -> right) gets further.
    runner = run_genome(env, genome_with(Gene(BIAS, RIGHT, weight=1.0, innovation=1)))
    env.close()
    assert still.frames < 40
    assert runner.rightmost > still.rightmost
    assert runner.fitness == runner.rightmost - runner.frames / 2
