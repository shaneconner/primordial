"""Simulation configuration and default parameters."""

from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    width: float = 500.0
    height: float = 500.0
    wrap_around: bool = True
    spatial_cell_size: float = 50.0

    # Resource spawning
    plant_spawn_rate: float = 1.5  # plants per tick
    plant_energy: float = 25.0
    plant_min_spawn_rate: float = 0.5  # floor to prevent total starvation
    initial_food_count: int = 500  # starting food resources
    max_resources: int = 2000  # cap to prevent unbounded growth
    meat_decay_rate: float = 0.005  # fraction of energy lost per tick
    nutrient_plant_boost: float = 2.0  # multiplier on local plant growth

    # Anti-extinction
    min_population: int = 10
    food_boost_threshold: int = 20  # boost food if pop drops below this


@dataclass
class BodyConfig:
    # Physics
    spring_stiffness_bone: float = 50.0
    spring_stiffness_muscle: float = 20.0
    spring_stiffness_tendon: float = 35.0
    spring_damping: float = 0.8
    drag_coefficient: float = 0.05
    dt: float = 0.1  # physics timestep
    substeps: int = 4  # physics substeps per simulation tick

    # Stability constraints
    max_stretch: float = 3.0  # max multiple of rest length before clamping
    max_velocity: float = 50.0  # velocity cap per node

    # Muscle control
    muscle_min_contraction: float = 0.5  # min fraction of rest length
    muscle_max_contraction: float = 1.5  # max fraction of rest length
    muscle_force_scale: float = 1.0

    # Node masses
    mass_core: float = 2.0
    mass_bone: float = 1.5
    mass_muscle_anchor: float = 1.0
    mass_sensor: float = 0.5
    mass_mouth: float = 1.0
    mass_fat: float = 1.5
    mass_armor: float = 2.5

    # Energy costs per tick (per node)
    cost_core: float = 0.01
    cost_bone: float = 0.005
    cost_muscle_anchor: float = 0.015
    cost_sensor: float = 0.01
    cost_mouth: float = 0.01
    cost_fat: float = 0.002
    cost_armor: float = 0.008

    # Eating
    eat_radius: float = 8.0  # contact distance for mouth to eat

    # Brownian motion (base wandering for random-brained organisms)
    brownian_force: float = 2.0  # random force magnitude per tick

    # Sensor range
    sensor_range: float = 100.0
    sensor_fov: float = 0.8  # radians, half-angle of sensor cone


@dataclass
class BrainConfig:
    max_hidden_size: int = 32
    max_inputs: int = 32  # buffer size for variable input count
    max_outputs: int = 16  # buffer size for variable output count
    n_memory: int = 2  # number of memory registers
    default_hidden_size: int = 8
    default_activation: str = "tanh"  # tanh, relu, sigmoid


@dataclass
class EvolutionConfig:
    # Reproduction
    asexual_energy_threshold: float = 0.6  # fraction of max energy
    sexual_energy_threshold: float = 0.5
    asexual_energy_cost: float = 0.5  # fraction given to offspring
    sexual_energy_cost: float = 0.3  # each parent contributes
    min_reproduce_age: int = 100  # ticks
    reproduce_cooldown: int = 100  # ticks between reproductions

    # Mutation rates (defaults, themselves evolvable)
    body_mutation_rate: float = 0.05
    brain_mutation_rate: float = 0.03
    weight_perturb_scale: float = 0.3
    meta_mutation_rate: float = 0.05  # fixed rate for mutating meta-params

    # Body mutations
    add_node_prob: float = 0.6
    remove_node_prob: float = 0.15
    change_type_prob: float = 0.1
    add_remove_edge_prob: float = 0.1
    perturb_pos_prob: float = 0.05
    max_body_nodes: int = 20

    # Speciation
    speciation_threshold: float = 3.0
    body_distance_weight: float = 1.0
    brain_distance_weight: float = 0.5
    topology_distance_weight: float = 2.0

    # Interaction
    attack_damage_per_mouth: float = 5.0
    eat_efficiency: float = 0.8  # fraction of food energy absorbed


@dataclass
class RecorderConfig:
    snapshot_interval: int = 10  # record every N ticks
    max_snapshots: int = 10000  # per recording session
    output_dir: str = "clips"


@dataclass
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)

    initial_population: int = 50
    max_population: int = 500
    max_ticks: int = 100_000
    seed: int = 42
