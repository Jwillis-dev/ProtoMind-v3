# protomind/task/tool_evolver.py

import copy
import random
from protomind.task.tool import Tool

class ToolEvolver:
    """
    Generates variant tools from an existing tool.
    Variants slightly mutate the affordances to explore new capabilities.
    """

    def __init__(self, mutation_rate: float = 0.1, num_variants: int = 3):
        """
        mutation_rate: max fractional change per numeric affordance
        num_variants: how many variants to generate per evolution
        """
        self.mutation_rate = mutation_rate
        self.num_variants = num_variants

    def generate(self, tool: Tool):
        """
        Returns a list of mutated tool variants.
        """
        variants = []
        for i in range(self.num_variants):
            new_tool = copy.deepcopy(tool)
            new_tool.name = f"{tool.name}_var{i+1}"

            # Mutate numeric affordances
            for key, value in new_tool.affordances.items():
                if isinstance(value, (int, float)):
                    change = value * random.uniform(-self.mutation_rate, self.mutation_rate)
                    new_tool.affordances[key] = value + change

                # For boolean affordances, flip with small probability
                elif isinstance(value, bool):
                    if random.random() < self.mutation_rate:
                        new_tool.affordances[key] = not value

            variants.append(new_tool)

        return variants
