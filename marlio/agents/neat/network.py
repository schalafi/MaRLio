"""Turning a genome into a working network (MarI/O's generateNetwork/evaluateNetwork)."""

from __future__ import annotations

import math

import numpy as np

from marlio.agents.neat.genome import MAX_NODES, Genome


def sigmoid(x: float) -> float:
    """MarI/O's squashing function: an S-curve from -1 to 1, steep around 0."""
    return 2 / (1 + math.exp(-4.9 * max(-60.0, min(60.0, x)))) - 1


class Network:
    """The neurons and enabled connections of one genome.

    Like MarI/O, neuron values persist between evaluations (within one run),
    so a connection that loops back gives the network a little memory.
    """

    def __init__(self, genome: Genome, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.values: dict[int, float] = {i: 0.0 for i in range(n_inputs)}
        for o in range(n_outputs):
            self.values[MAX_NODES + o] = 0.0
        self.incoming: dict[int, list[tuple[int, float]]] = {}
        for gene in sorted(genome.genes, key=lambda g: g.out):
            if not gene.enabled:
                continue
            self.incoming.setdefault(gene.out, []).append((gene.into, gene.weight))
            self.values.setdefault(gene.out, 0.0)
            self.values.setdefault(gene.into, 0.0)
        # One pass in id order: inputs, then hidden neurons in the order they
        # were created, then outputs. (MarI/O walks its neuron table with Lua's
        # `pairs`, which in practice gives the same order.)
        self.order = sorted(n for n in self.values if n >= n_inputs)

    def reset(self) -> None:
        for neuron in self.values:
            self.values[neuron] = 0.0

    def evaluate(self, inputs: np.ndarray) -> list[int]:
        """Feed the inputs (without bias) and return 6 button presses (0/1)."""
        values = self.values
        for i, value in enumerate(inputs):
            values[i] = float(value)
        values[self.n_inputs - 1] = 1.0  # bias
        for neuron in self.order:
            incoming = self.incoming.get(neuron)
            if incoming:
                total = sum(weight * values[source] for source, weight in incoming)
                values[neuron] = sigmoid(total)
        return [int(values[MAX_NODES + o] > 0) for o in range(self.n_outputs)]
