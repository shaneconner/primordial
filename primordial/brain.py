"""Feedforward neural network for organism brains."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import BrainConfig


# Activation functions
def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


_ACTIVATIONS = {
    "tanh": _tanh,
    "relu": _relu,
    "sigmoid": _sigmoid,
}


class Brain:
    """Simple feedforward neural network.

    Architecture: inputs -> hidden (with activation) -> outputs (sigmoid)

    Input/output sizes are determined by the organism's body morphology
    (number of sensors, muscles, etc.) but use fixed-size buffers so the
    network can handle body mutations without complete reconstruction.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_size: int,
        activation: str,
        n_memory: int,
        config: BrainConfig,
        weights_ih: np.ndarray | None = None,
        weights_ho: np.ndarray | None = None,
        bias_h: np.ndarray | None = None,
        bias_o: np.ndarray | None = None,
    ):
        self.config = config
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.activation_fn = _ACTIVATIONS[activation]
        self.n_memory = n_memory

        # Total I/O including memory registers
        self.total_inputs = n_inputs + n_memory  # regular inputs + memory read
        self.total_outputs = n_outputs + n_memory  # regular outputs + memory write

        # Weight matrices (Xavier initialization if not provided)
        if weights_ih is not None:
            self.weights_ih = weights_ih
        else:
            scale = np.sqrt(2.0 / (self.total_inputs + hidden_size))
            self.weights_ih = np.random.randn(self.total_inputs, hidden_size) * scale

        if weights_ho is not None:
            self.weights_ho = weights_ho
        else:
            scale = np.sqrt(2.0 / (hidden_size + self.total_outputs))
            self.weights_ho = np.random.randn(hidden_size, self.total_outputs) * scale

        if bias_h is not None:
            self.bias_h = bias_h
        else:
            self.bias_h = np.zeros(hidden_size)

        if bias_o is not None:
            self.bias_o = bias_o
        else:
            self.bias_o = np.zeros(self.total_outputs)

        # Memory registers (persist across ticks)
        self.memory = np.zeros(n_memory)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run forward pass.

        Args:
            inputs: Array of sensor/proprioception inputs. Length should match
                    n_inputs. If shorter, remaining inputs are 0. If longer, truncated.

        Returns:
            Array of outputs (muscle targets + eat + reproduce signals).
            Does NOT include memory write values - those are handled internally.
        """
        # Build full input vector with memory
        full_input = np.zeros(self.total_inputs)
        n = min(len(inputs), self.n_inputs)
        full_input[:n] = inputs[:n]
        full_input[self.n_inputs : self.n_inputs + self.n_memory] = self.memory

        # Hidden layer
        hidden = full_input @ self.weights_ih + self.bias_h
        hidden = self.activation_fn(hidden)

        # Output layer (sigmoid for bounded outputs)
        output = hidden @ self.weights_ho + self.bias_o
        output = _sigmoid(output)

        # Update memory from output
        if self.n_memory > 0:
            self.memory = output[self.n_outputs : self.n_outputs + self.n_memory]

        # Return only the action outputs (not memory writes)
        return output[: self.n_outputs]

    def get_weight_count(self) -> int:
        """Total number of trainable parameters."""
        return (
            self.weights_ih.size
            + self.weights_ho.size
            + self.bias_h.size
            + self.bias_o.size
        )

    def to_dict(self) -> dict:
        """Serialize brain for recording/genome storage."""
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "hidden_size": self.hidden_size,
            "activation": self.activation_name,
            "n_memory": self.n_memory,
            "weights_ih": self.weights_ih.tolist(),
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": self.bias_h.tolist(),
            "bias_o": self.bias_o.tolist(),
            "memory": self.memory.tolist(),
        }


def create_brain_for_body(
    n_sensors: int,
    n_muscles: int,
    hidden_size: int,
    activation: str,
    n_memory: int,
    config: BrainConfig,
    rng: np.random.Generator,
) -> Brain:
    """Create a brain sized for a specific body morphology.

    Inputs per sensor: 3 (distance to food, distance to organism, organism type)
    Additional inputs: energy, age, proprioception (1 per muscle)
    Outputs: 1 per muscle (contraction), 1 eat signal, 1 reproduce signal
    """
    n_inputs = (n_sensors * 3) + 2 + n_muscles  # sensors + energy + age + proprioception
    n_outputs = n_muscles + 2  # muscle targets + eat + reproduce

    # Clamp to buffer limits
    n_inputs = min(n_inputs, config.max_inputs)
    n_outputs = min(n_outputs, config.max_outputs)

    # Xavier init
    total_in = n_inputs + n_memory
    total_out = n_outputs + n_memory

    scale_ih = np.sqrt(2.0 / (total_in + hidden_size))
    weights_ih = rng.normal(0, scale_ih, (total_in, hidden_size))

    scale_ho = np.sqrt(2.0 / (hidden_size + total_out))
    weights_ho = rng.normal(0, scale_ho, (hidden_size, total_out))

    return Brain(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_size=hidden_size,
        activation=activation,
        n_memory=n_memory,
        config=config,
        weights_ih=weights_ih,
        weights_ho=weights_ho,
        bias_h=np.zeros(hidden_size),
        bias_o=np.zeros(total_out),
    )
