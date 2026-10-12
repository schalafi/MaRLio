"""NEAT (NeuroEvolution of Augmenting Topologies), as implemented in MarI/O."""

from marlio.agents.neat.agent import N_INPUTS, N_OUTPUTS, Evaluator, NEATAgent, RunResult, run_genome
from marlio.agents.neat.genome import Gene, Genome, NEATConfig
from marlio.agents.neat.population import Population

__all__ = [
    "N_INPUTS",
    "N_OUTPUTS",
    "Evaluator",
    "Gene",
    "Genome",
    "NEATAgent",
    "NEATConfig",
    "Population",
    "RunResult",
    "run_genome",
]
