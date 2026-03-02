"""Neural network for organism brains.

Part 1-3: Feedforward with memory registers.
Part 4: Adds recurrent hidden state, memory decay, and neuromodulation.
"""

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
    """Neural network with optional recurrence and neuromodulation.

    Architecture: inputs (+memory +prev_hidden) -> hidden (with activation) -> outputs (sigmoid)

    Part 4 additions:
    - Recurrent: previous hidden state concatenated with inputs
    - Memory decay: memory registers decay each tick (configurable rate)
    - Neuromodulation: dedicated outputs that scale all other outputs globally
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

        # Part 4: Recurrence and modulation
        self.recurrent = getattr(config, 'enable_recurrent', False)
        self.memory_decay = getattr(config, 'memory_decay', 1.0)
        self.n_modulators = getattr(config, 'n_modulators', 0)

        # Total I/O including memory registers
        # If recurrent: inputs also include prev_hidden
        self.recurrent_size = hidden_size if self.recurrent else 0
        self.total_inputs = n_inputs + n_memory + self.recurrent_size
        self.total_outputs = n_outputs + n_memory

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

        # Part 4: Previous hidden state for recurrence
        self.prev_hidden = np.zeros(hidden_size) if self.recurrent else None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run forward pass.

        Args:
            inputs: Array of sensor/proprioception inputs.

        Returns:
            Array of outputs (muscle targets + action signals).
        """
        # Apply memory decay before reading
        if self.memory_decay < 1.0 and self.n_memory > 0:
            self.memory *= self.memory_decay

        # Build full input vector: [inputs | memory | prev_hidden (if recurrent)]
        full_input = np.zeros(self.total_inputs)
        n = min(len(inputs), self.n_inputs)
        full_input[:n] = inputs[:n]
        full_input[self.n_inputs : self.n_inputs + self.n_memory] = self.memory
        if self.recurrent and self.prev_hidden is not None:
            offset = self.n_inputs + self.n_memory
            full_input[offset : offset + self.recurrent_size] = self.prev_hidden

        # Hidden layer
        hidden = full_input @ self.weights_ih + self.bias_h
        hidden = self.activation_fn(hidden)

        # Save hidden state for next tick (recurrence)
        if self.recurrent:
            self.prev_hidden = hidden.copy()

        # Output layer (sigmoid for bounded outputs)
        output = hidden @ self.weights_ho + self.bias_o
        output = _sigmoid(output)

        # Update memory from output
        if self.n_memory > 0:
            self.memory = output[self.n_outputs : self.n_outputs + self.n_memory]

        # Extract action outputs
        action_outputs = output[: self.n_outputs].copy()

        # Part 4: Neuromodulation — last n_modulators of action outputs
        # scale all non-modulator outputs by modulator values mapped to [0.5, 2.0]
        if self.n_modulators > 0 and self.n_outputs > self.n_modulators:
            mod_start = self.n_outputs - self.n_modulators
            modulators = action_outputs[mod_start:]
            # Map [0, 1] sigmoid output to [0.5, 2.0] gain range
            gains = 0.5 + modulators * 1.5
            # Average gain across modulators
            avg_gain = float(np.mean(gains))
            # Scale all non-modulator outputs
            action_outputs[:mod_start] *= avg_gain

        return action_outputs

    def resize_hidden(self, new_size: int, rng: np.random.Generator) -> None:
        """Grow or shrink the hidden layer, preserving existing weights."""
        old_size = self.hidden_size
        if new_size == old_size:
            return

        # Resize weights_ih: (total_inputs, hidden_size)
        new_wih = np.zeros((self.total_inputs, new_size))
        copy_h = min(old_size, new_size)
        new_wih[:, :copy_h] = self.weights_ih[:, :copy_h]
        if new_size > old_size:
            new_wih[:, old_size:] = rng.normal(0, 0.01, (self.total_inputs, new_size - old_size))
        self.weights_ih = new_wih

        # Resize weights_ho: (hidden_size, total_outputs)
        new_who = np.zeros((new_size, self.total_outputs))
        new_who[:copy_h, :] = self.weights_ho[:copy_h, :]
        if new_size > old_size:
            new_who[old_size:, :] = rng.normal(0, 0.01, (new_size - old_size, self.total_outputs))
        self.weights_ho = new_who

        # Resize bias_h
        new_bh = np.zeros(new_size)
        new_bh[:copy_h] = self.bias_h[:copy_h]
        self.bias_h = new_bh

        self.hidden_size = new_size

        # Part 4: Resize recurrent state and update total_inputs
        if self.recurrent:
            new_prev = np.zeros(new_size)
            new_prev[:copy_h] = self.prev_hidden[:copy_h] if self.prev_hidden is not None else 0
            self.prev_hidden = new_prev
            self.recurrent_size = new_size
            self.total_inputs = self.n_inputs + self.n_memory + self.recurrent_size

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
    """Create a brain sized for a specific body morphology."""
    ips = config.inputs_per_sensor
    n_globals = getattr(config, 'n_global_inputs', 2)
    n_inputs = (n_sensors * ips) + n_globals + n_muscles
    n_outputs = n_muscles + config.n_action_outputs

    # Clamp to buffer limits
    n_inputs = min(n_inputs, config.max_inputs)
    n_outputs = min(n_outputs, config.max_outputs)

    # Recurrent size increases total input dimension
    recurrent = getattr(config, 'enable_recurrent', False)
    recurrent_size = hidden_size if recurrent else 0

    # Xavier init
    total_in = n_inputs + n_memory + recurrent_size
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
