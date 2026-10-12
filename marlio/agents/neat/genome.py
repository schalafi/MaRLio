"""NEAT genomes: the "DNA" of a network, and how it mutates and mates.

A genome is a list of connection genes. Each gene says "neuron ``into``
feeds neuron ``out`` with this ``weight``" and carries an *innovation
number*: a global id given the first time that kind of change appears, so
genes from different genomes can be lined up for crossover and for measuring
how different two genomes are (Stanley & Miikkulainen 2002, section 3.2).

This is a port of the NEAT part of SethBling's MarI/O script, keeping its
constants, mutation operators and their probabilities. Neuron ids:

* ``0 .. n_inputs - 1``: inputs (the last one is the bias, always 1)
* ``MAX_NODES + o``: output ``o`` (one per button)
* anything in between: hidden neurons, numbered as they are created
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

# Outputs live far above every other neuron id, as in MarI/O.
MAX_NODES = 1_000_000


@dataclass
class NEATConfig:
    """MarI/O's NEAT constants (same names, same values)."""

    population: int = 300
    # Speciation: two genomes are the same species when
    # delta_disjoint * (share of unmatched genes)
    # + delta_weights * (mean weight difference of matched genes) < delta_threshold
    delta_disjoint: float = 2.0
    delta_weights: float = 0.4
    delta_threshold: float = 1.0
    # A species that hasn't improved for this many generations is removed.
    stale_species: int = 15
    # Starting mutation rates (each genome then evolves its own).
    mutate_connections_chance: float = 0.25
    perturb_chance: float = 0.90
    crossover_chance: float = 0.75
    link_mutation_chance: float = 2.0
    node_mutation_chance: float = 0.50
    bias_mutation_chance: float = 0.40
    step_size: float = 0.1
    disable_mutation_chance: float = 0.4
    enable_mutation_chance: float = 0.2


@dataclass
class Gene:
    into: int
    out: int
    weight: float = 0.0
    enabled: bool = True
    innovation: int = 0


@dataclass
class Genome:
    genes: list[Gene] = field(default_factory=list)
    # Highest neuron id in use; new hidden neurons get max_neuron + 1.
    max_neuron: int = 0
    mutation_rates: dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    global_rank: int = 0

    def copy(self) -> Genome:
        """A fresh child with the same genes and rates (fitness not copied)."""
        return Genome(
            genes=copy.deepcopy(self.genes),
            max_neuron=self.max_neuron,
            mutation_rates=dict(self.mutation_rates),
        )

    def to_dict(self) -> dict:
        return {
            "genes": [vars(gene) for gene in self.genes],
            "max_neuron": self.max_neuron,
            "mutation_rates": self.mutation_rates,
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Genome:
        return cls(
            genes=[Gene(**gene) for gene in data["genes"]],
            max_neuron=data["max_neuron"],
            mutation_rates=dict(data["mutation_rates"]),
            fitness=data.get("fitness", 0.0),
        )


class Innovations:
    """Hands out global innovation numbers."""

    def __init__(self, start: int = 0):
        self.last = start

    def next(self) -> int:
        self.last += 1
        return self.last


def initial_mutation_rates(config: NEATConfig) -> dict[str, float]:
    return {
        "connections": config.mutate_connections_chance,
        "link": config.link_mutation_chance,
        "bias": config.bias_mutation_chance,
        "node": config.node_mutation_chance,
        "enable": config.enable_mutation_chance,
        "disable": config.disable_mutation_chance,
        "step": config.step_size,
    }


def basic_genome(n_inputs: int, n_outputs: int, config: NEATConfig,
                 innovations: Innovations, rng: random.Random) -> Genome:
    """A genome with no genes, mutated once: NEAT starts minimal."""
    genome = Genome(max_neuron=n_inputs - 1, mutation_rates=initial_mutation_rates(config))
    mutate(genome, n_inputs, n_outputs, config, innovations, rng)
    return genome


# --- Mutations ---------------------------------------------------------------

def _random_neuron(genes: list[Gene], non_input: bool, n_inputs: int, n_outputs: int,
                   rng: random.Random) -> int:
    neurons = set()
    if not non_input:
        neurons.update(range(n_inputs))
    neurons.update(MAX_NODES + o for o in range(n_outputs))
    for gene in genes:
        if not non_input or gene.into >= n_inputs:
            neurons.add(gene.into)
        if not non_input or gene.out >= n_inputs:
            neurons.add(gene.out)
    return rng.choice(sorted(neurons))


def _contains_link(genes: list[Gene], link: Gene) -> bool:
    return any(gene.into == link.into and gene.out == link.out for gene in genes)


def point_mutate(genome: Genome, config: NEATConfig, rng: random.Random) -> None:
    """Nudge every weight a little, or (rarely) replace it with a random one."""
    step = genome.mutation_rates["step"]
    for gene in genome.genes:
        if rng.random() < config.perturb_chance:
            gene.weight += rng.random() * step * 2 - step
        else:
            gene.weight = rng.random() * 4 - 2


def link_mutate(genome: Genome, force_bias: bool, n_inputs: int, n_outputs: int,
                innovations: Innovations, rng: random.Random) -> None:
    """Add a new connection between two neurons that aren't linked yet."""
    neuron1 = _random_neuron(genome.genes, False, n_inputs, n_outputs, rng)
    neuron2 = _random_neuron(genome.genes, True, n_inputs, n_outputs, rng)
    if neuron1 < n_inputs and neuron2 < n_inputs:
        return  # both inputs
    if neuron2 < n_inputs:
        neuron1, neuron2 = neuron2, neuron1
    link = Gene(into=neuron1, out=neuron2)
    if force_bias:
        link.into = n_inputs - 1
    if _contains_link(genome.genes, link):
        return
    link.innovation = innovations.next()
    link.weight = rng.random() * 4 - 2
    genome.genes.append(link)


def node_mutate(genome: Genome, innovations: Innovations, rng: random.Random) -> None:
    """Split a connection in two by putting a new neuron in the middle."""
    if not genome.genes:
        return
    genome.max_neuron += 1
    gene = rng.choice(genome.genes)
    if not gene.enabled:
        return
    gene.enabled = False
    genome.genes.append(Gene(into=gene.into, out=genome.max_neuron, weight=1.0,
                             enabled=True, innovation=innovations.next()))
    genome.genes.append(Gene(into=genome.max_neuron, out=gene.out, weight=gene.weight,
                             enabled=True, innovation=innovations.next()))


def enable_disable_mutate(genome: Genome, enable: bool, rng: random.Random) -> None:
    """Switch one connection on (or off)."""
    candidates = [gene for gene in genome.genes if gene.enabled != enable]
    if candidates:
        gene = rng.choice(candidates)
        gene.enabled = not gene.enabled


def _repeat(chance: float, rng: random.Random, action) -> None:
    """Rates above 1 mean "possibly several times": try once per whole unit."""
    while chance > 0:
        if rng.random() < chance:
            action()
        chance -= 1


def mutate(genome: Genome, n_inputs: int, n_outputs: int, config: NEATConfig,
           innovations: Innovations, rng: random.Random) -> None:
    # The mutation rates themselves drift, so evolution tunes how it mutates.
    for name, rate in genome.mutation_rates.items():
        genome.mutation_rates[name] = rate * (0.95 if rng.randint(1, 2) == 1 else 1.05263)
    rates = genome.mutation_rates

    if rng.random() < rates["connections"]:
        point_mutate(genome, config, rng)
    _repeat(rates["link"], rng,
            lambda: link_mutate(genome, False, n_inputs, n_outputs, innovations, rng))
    _repeat(rates["bias"], rng,
            lambda: link_mutate(genome, True, n_inputs, n_outputs, innovations, rng))
    _repeat(rates["node"], rng, lambda: node_mutate(genome, innovations, rng))
    _repeat(rates["enable"], rng, lambda: enable_disable_mutate(genome, True, rng))
    _repeat(rates["disable"], rng, lambda: enable_disable_mutate(genome, False, rng))


# --- Crossover and speciation ------------------------------------------------

def crossover(genome1: Genome, genome2: Genome, rng: random.Random) -> Genome:
    """Child of two parents: genes line up by innovation number.

    Matching genes come from either parent at random; genes only the fitter
    parent has are kept, genes only the weaker parent has are dropped.
    """
    if genome2.fitness > genome1.fitness:
        genome1, genome2 = genome2, genome1
    by_innovation = {gene.innovation: gene for gene in genome2.genes}
    child = Genome(max_neuron=max(genome1.max_neuron, genome2.max_neuron),
                   mutation_rates=dict(genome1.mutation_rates))
    for gene1 in genome1.genes:
        gene2 = by_innovation.get(gene1.innovation)
        if gene2 is not None and rng.randint(1, 2) == 1 and gene2.enabled:
            child.genes.append(copy.copy(gene2))
        else:
            child.genes.append(copy.copy(gene1))
    return child


def disjoint(genes1: list[Gene], genes2: list[Gene]) -> float:
    """Share of genes that have no partner (same innovation) in the other genome."""
    innovations1 = {gene.innovation for gene in genes1}
    innovations2 = {gene.innovation for gene in genes2}
    unmatched = len(innovations1 - innovations2) + len(innovations2 - innovations1)
    n = max(len(genes1), len(genes2))
    return unmatched / n if n else 0.0


def weights(genes1: list[Gene], genes2: list[Gene]) -> float:
    """Mean weight difference of the genes both genomes share."""
    by_innovation = {gene.innovation: gene for gene in genes2}
    total, coincident = 0.0, 0
    for gene in genes1:
        partner = by_innovation.get(gene.innovation)
        if partner is not None:
            total += abs(gene.weight - partner.weight)
            coincident += 1
    # MarI/O divides by zero here (giving NaN, so "different species");
    # treat "nothing in common" as infinitely different instead.
    return total / coincident if coincident else float("inf")


def same_species(genome1: Genome, genome2: Genome, config: NEATConfig) -> bool:
    dd = config.delta_disjoint * disjoint(genome1.genes, genome2.genes)
    dw = config.delta_weights * weights(genome1.genes, genome2.genes)
    return dd + dw < config.delta_threshold
