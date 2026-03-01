"""Simulation configuration and default parameters."""

from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    width: float = 1600.0
    height: float = 900.0
    wrap_around: bool = True
    spatial_cell_size: float = 80.0

    # Resource spawning
    plant_spawn_rate: float = 2.5  # plants per tick
    plant_energy: float = 25.0
    plant_min_spawn_rate: float = 2.0  # floor to prevent total starvation
    initial_food_count: int = 600  # starting food resources
    max_resources: int = 3000  # cap to prevent unbounded growth
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
    substeps: int = 2  # physics substeps per simulation tick

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
    cost_core: float = 0.04
    cost_bone: float = 0.02
    cost_muscle_anchor: float = 0.06
    cost_sensor: float = 0.04
    cost_mouth: float = 0.04
    cost_fat: float = 0.008
    cost_armor: float = 0.035

    # Eating
    eat_radius: float = 8.0  # contact distance for mouth to eat

    # Brownian motion (base wandering for random-brained organisms)
    brownian_force: float = 2.0  # random force magnitude per tick

    # Sensor range
    sensor_range: float = 100.0
    sensor_fov: float = 0.8  # radians, half-angle of sensor cone

    # Body geometry (initial spread and mutation offsets)
    initial_spread: float = 1.5  # default body spread from core
    new_node_offset_sigma: float = 1.5  # sigma for new node placement
    new_node_outward_bias: float = 0.0  # bias new nodes away from core (0=none)
    position_perturb_sigma: float = 0.5  # sigma for position perturbation mutations

    # Anti-minimalism mechanics
    size_force_scaling: bool = False  # larger bodies move faster
    bone_drag_reduction: float = 0.0  # drag reduction per bone node
    fat_repro_bonus: float = 0.0  # reproduction threshold reduction per fat node

    # Part 3: Muscle -> Speed
    muscle_speed_scaling: bool = False  # top speed scales with muscle ratio
    base_max_velocity: float = 40.0  # base speed when muscle_speed_scaling on
    muscle_velocity_bonus: float = 100.0  # multiplied by muscle_ratio and added to base

    # Part 3: Bone -> Reach
    bone_reach_scaling: bool = False  # mouths far from COM get bigger eat radius
    bone_reach_factor: float = 0.3  # eat_radius += distance_from_com * factor

    # Part 3: Armor damage reflection
    armor_damage_reflection: float = 0.0  # fraction of damage reflected to attacker

    # Part 3: Kin recognition
    enable_kin_recognition: bool = False
    offspring_immunity_ticks: int = 50  # newborns immune from parent attack

    # Part 3: Environmental dynamics
    enable_seasons: bool = False
    season_length: int = 10000  # ticks per full season cycle
    season_food_amplitude: float = 0.4  # ±fraction of base spawn rate
    enable_food_shocks: bool = False
    food_shock_probability: float = 0.0001  # per tick chance
    food_shock_duration: int = 2000  # ticks
    food_shock_severity: float = 0.3  # reduce spawn to this fraction
    enable_spatial_gradient: bool = False
    gradient_center_x: float = 0.5  # normalized [0,1]
    gradient_center_y: float = 0.5
    gradient_strength: float = 0.5  # 0=uniform, 1=strong gradient
    gradient_shift_rate: float = 0.00005  # center moves per tick

    # Part 3: Limb chain mutations
    limb_chain_bias: float = 0.0  # bias for attaching to peripheral nodes


@dataclass
class BrainConfig:
    max_hidden_size: int = 32
    max_inputs: int = 32  # buffer size for variable input count
    max_outputs: int = 16  # buffer size for variable output count
    n_memory: int = 2  # number of memory registers
    default_hidden_size: int = 16
    default_activation: str = "tanh"  # tanh, relu, sigmoid
    hidden_size_mutation_rate: float = 0.05  # chance to grow/shrink hidden layer
    activation_mutation_rate: float = 0.02  # chance to switch activation fn
    inputs_per_sensor: int = 3  # sensor inputs (Part 1: 3, Part 2: 6)
    n_action_outputs: int = 2  # action outputs beyond muscles (Part 1: eat+repro, Part 2: eat+attack+repro)


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
    body_mutation_rate: float = 0.12
    brain_mutation_rate: float = 0.08
    weight_perturb_scale: float = 0.5
    meta_mutation_rate: float = 0.05  # fixed rate for mutating meta-params

    # Body mutations
    add_node_prob: float = 0.45
    remove_node_prob: float = 0.15
    change_type_prob: float = 0.15
    add_remove_edge_prob: float = 0.10
    perturb_pos_prob: float = 0.15
    max_body_nodes: int = 30

    # Speciation
    speciation_threshold: float = 3.0
    body_distance_weight: float = 1.0
    brain_distance_weight: float = 0.5
    topology_distance_weight: float = 2.0

    # Aging
    max_age: int = 5000  # ticks before death
    senescence_age: float = 0.6  # fraction of max_age where metabolic cost starts rising
    senescence_max_multiplier: float = 3.0  # metabolic cost multiplier at max_age

    # Interaction
    attack_damage_per_mouth: float = 8.0
    eat_efficiency: float = 0.8  # fraction of food energy absorbed
    predation_energy_transfer: float = 0.5  # fraction of damage attacker gains as energy

    # Sexual reproduction
    enable_sexual_reproduction: bool = False
    sexual_proximity: float = 50.0  # max distance for mating


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

    @classmethod
    def part2(cls, seed: int = 137) -> "SimConfig":
        """Create Part 2 config with redesigned parameters for body diversity."""
        return cls(
            world=WorldConfig(
                plant_spawn_rate=1.2,
                plant_energy=20.0,
                plant_min_spawn_rate=0.8,
                initial_food_count=400,
                meat_decay_rate=0.002,
            ),
            body=BodyConfig(
                max_velocity=80.0,
                brownian_force=5.0,
                # Reduced structural node costs
                cost_bone=0.01,
                cost_sensor=0.025,
                cost_armor=0.02,
                cost_fat=0.005,
                # Larger bodies
                initial_spread=8.0,
                new_node_offset_sigma=6.0,
                new_node_outward_bias=0.5,
                position_perturb_sigma=2.0,
                # Anti-minimalism
                size_force_scaling=True,
                bone_drag_reduction=0.15,
                fat_repro_bonus=0.05,
            ),
            brain=BrainConfig(
                max_inputs=48,
                inputs_per_sensor=6,
                n_action_outputs=3,  # eat + attack + reproduce
            ),
            evolution=EvolutionConfig(
                max_age=15000,
                senescence_age=0.7,
                enable_sexual_reproduction=True,
                sexual_proximity=50.0,
            ),
            initial_population=50,
            max_population=500,
            max_ticks=300_000,
            seed=seed,
        )

    @classmethod
    def part3(cls, seed: int = 271) -> "SimConfig":
        """Create Part 3 config: bodies that matter.

        Builds on Part 2 with changes that make every node type earn its cost:
        - Muscle -> speed scaling
        - Bone -> foraging reach
        - Stronger predation + armor reflection
        - Kin recognition (evolvable)
        - Environmental dynamics (seasons, shocks, gradients)
        - Limb chain mutation bias
        """
        return cls(
            world=WorldConfig(
                plant_spawn_rate=1.2,
                plant_energy=20.0,
                plant_min_spawn_rate=0.8,
                initial_food_count=400,
                meat_decay_rate=0.002,
            ),
            body=BodyConfig(
                # Base velocity lower; muscle ratio adds speed
                max_velocity=80.0,
                base_max_velocity=40.0,
                muscle_speed_scaling=True,
                muscle_velocity_bonus=100.0,
                # Bone reach
                bone_reach_scaling=True,
                bone_reach_factor=0.3,
                # Armor reflection
                armor_damage_reflection=0.2,
                # Kin recognition
                enable_kin_recognition=True,
                offspring_immunity_ticks=50,
                # Environmental dynamics
                enable_seasons=True,
                season_length=10000,
                season_food_amplitude=0.4,
                enable_food_shocks=True,
                food_shock_probability=0.0001,
                food_shock_duration=2000,
                food_shock_severity=0.3,
                enable_spatial_gradient=True,
                gradient_strength=0.5,
                gradient_shift_rate=0.00005,
                # Limb chain bias
                limb_chain_bias=0.4,
                # Larger bodies (same as Part 2)
                brownian_force=5.0,
                initial_spread=8.0,
                new_node_offset_sigma=6.0,
                new_node_outward_bias=0.5,
                position_perturb_sigma=2.0,
                # Anti-minimalism (same as Part 2)
                size_force_scaling=True,
                bone_drag_reduction=0.15,
                fat_repro_bonus=0.05,
                # Reduced structural costs (same as Part 2)
                cost_bone=0.01,
                cost_sensor=0.025,
                cost_armor=0.02,
                cost_fat=0.005,
            ),
            brain=BrainConfig(
                max_inputs=48,
                inputs_per_sensor=6,
                n_action_outputs=3,  # eat + attack + reproduce
            ),
            evolution=EvolutionConfig(
                max_age=15000,
                senescence_age=0.7,
                enable_sexual_reproduction=True,
                sexual_proximity=50.0,
                # Stronger predation
                attack_damage_per_mouth=15.0,
                predation_energy_transfer=0.7,
            ),
            initial_population=50,
            max_population=500,
            max_ticks=300_000,
            seed=seed,
        )
