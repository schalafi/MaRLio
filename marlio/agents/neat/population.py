"""A NEAT population: species, selection and breeding (MarI/O's pool)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from marlio.agents.neat.genome import (
    Genome,
    Innovations,
    NEATConfig,
    basic_genome,
    crossover,
    mutate,
    same_species,
)


def _rng_state_to_json(state: tuple) -> list:
    version, internal, gauss_next = state
    return [version, list(internal), gauss_next]


def _rng_state_from_json(state: list) -> tuple:
    version, internal, gauss_next = state
    return (version, tuple(internal), gauss_next)


@dataclass
class Species:
    genomes: list[Genome] = field(default_factory=list)
    top_fitness: float = 0.0
    staleness: int = 0
    average_fitness: float = 0.0


class Population:
    """All genomes, grouped into species of similar networks.

    One generation: evaluate every genome (done outside this class, it only
    needs ``genome.fitness`` set), then ``new_generation()`` keeps the best,
    drops stale and weak species, and breeds children to refill.
    Speciation protects new ideas: a genome with a fresh mutation competes
    mostly with its own species, so it has time to improve before it has to
    beat the whole population.
    """

    def __init__(self, n_inputs: int, n_outputs: int, config: NEATConfig | None = None,
                 seed: int | None = None, initialize: bool = True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.config = config or NEATConfig()
        self.rng = random.Random(seed)
        self.innovations = Innovations(start=n_outputs)
        self.species: list[Species] = []
        self.generation = 0
        self.max_fitness = 0.0
        if initialize:
            for _ in range(self.config.population):
                self._add_to_species(self._basic_genome())

    # --- helpers --------------------------------------------------------------

    def _basic_genome(self) -> Genome:
        return basic_genome(self.n_inputs, self.n_outputs, self.config, self.innovations, self.rng)

    def _mutate(self, genome: Genome) -> None:
        mutate(genome, self.n_inputs, self.n_outputs, self.config, self.innovations, self.rng)

    def genomes(self) -> list[Genome]:
        return [genome for species in self.species for genome in species.genomes]

    def best(self) -> Genome:
        return max(self.genomes(), key=lambda genome: genome.fitness)

    def _add_to_species(self, child: Genome) -> None:
        for species in self.species:
            if same_species(child, species.genomes[0], self.config):
                species.genomes.append(child)
                return
        self.species.append(Species(genomes=[child]))

    # --- one generation ----------------------------------------------------

    def record_fitness(self) -> None:
        """Update the all-time best after an evaluation round."""
        self.max_fitness = max(self.max_fitness, max(g.fitness for g in self.genomes()))

    def _rank_globally(self) -> None:
        for rank, genome in enumerate(sorted(self.genomes(), key=lambda g: g.fitness), start=1):
            genome.global_rank = rank

    def _cull_species(self, cut_to_one: bool) -> None:
        """Keep the top half of each species (or only its best)."""
        for species in self.species:
            species.genomes.sort(key=lambda g: g.fitness, reverse=True)
            remaining = 1 if cut_to_one else math.ceil(len(species.genomes) / 2)
            del species.genomes[remaining:]

    def _remove_stale_species(self) -> None:
        survivors = []
        for species in self.species:
            species.genomes.sort(key=lambda g: g.fitness, reverse=True)
            if species.genomes[0].fitness > species.top_fitness:
                species.top_fitness = species.genomes[0].fitness
                species.staleness = 0
            else:
                species.staleness += 1
            if species.staleness < self.config.stale_species or species.top_fitness >= self.max_fitness:
                survivors.append(species)
        self.species = survivors

    def _calculate_average_fitness(self) -> None:
        # Uses rank, not raw fitness, so one lucky outlier can't take over.
        for species in self.species:
            species.average_fitness = sum(g.global_rank for g in species.genomes) / len(species.genomes)

    def _total_average_fitness(self) -> float:
        return sum(species.average_fitness for species in self.species)

    def _remove_weak_species(self) -> None:
        total = self._total_average_fitness()
        self.species = [
            species for species in self.species
            if math.floor(species.average_fitness / total * self.config.population) >= 1
        ]

    def _breed_child(self, species: Species) -> Genome:
        if self.rng.random() < self.config.crossover_chance:
            parent1 = self.rng.choice(species.genomes)
            parent2 = self.rng.choice(species.genomes)
            child = crossover(parent1, parent2, self.rng)
        else:
            child = self.rng.choice(species.genomes).copy()
        self._mutate(child)
        return child

    def new_generation(self) -> None:
        """Select and breed the next generation (MarI/O's newGeneration)."""
        self.record_fitness()
        self._cull_species(cut_to_one=False)
        self._rank_globally()
        self._remove_stale_species()
        self._rank_globally()
        self._calculate_average_fitness()
        self._remove_weak_species()

        total = self._total_average_fitness()
        children = []
        for species in self.species:
            # Better species (by average rank) get more children.
            breed = math.floor(species.average_fitness / total * self.config.population) - 1
            children.extend(self._breed_child(species) for _ in range(breed))
        self._cull_species(cut_to_one=True)  # each species keeps its champion
        while len(children) + len(self.species) < self.config.population:
            children.append(self._breed_child(self.rng.choice(self.species)))
        for child in children:
            self._add_to_species(child)
        self.generation += 1

    # --- saving --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "config": vars(self.config),
            "generation": self.generation,
            "max_fitness": self.max_fitness,
            "innovation": self.innovations.last,
            "rng_state": _rng_state_to_json(self.rng.getstate()),
            "species": [
                {
                    "top_fitness": s.top_fitness,
                    "staleness": s.staleness,
                    "genomes": [g.to_dict() for g in s.genomes],
                }
                for s in self.species
            ],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> Population:
        data = json.loads(Path(path).read_text())
        population = cls(data["n_inputs"], data["n_outputs"], NEATConfig(**data["config"]),
                         initialize=False)
        population.generation = data["generation"]
        population.max_fitness = data["max_fitness"]
        population.innovations.last = data["innovation"]
        population.rng.setstate(_rng_state_from_json(data["rng_state"]))
        population.species = [
            Species(genomes=[Genome.from_dict(g) for g in s["genomes"]],
                    top_fitness=s["top_fitness"], staleness=s["staleness"])
            for s in data["species"]
        ]
        return population
