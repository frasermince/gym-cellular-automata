from dataclasses import dataclass


@dataclass
class PPOArgs:
    """PPO algorithm specific arguments"""

    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


@dataclass
class EnvArgs:
    """Environment configuration arguments"""

    env_id: str
    num_envs: int
    size: int = 256
    speed_move: float = 0.12
    speed_multiplier: float = 1.0
    use_hidden: bool = True
    enable_extensions: bool = False


@dataclass
class VisualizationArgs:
    """Visualization and recording arguments"""

    gif: bool = False
    steps: int = 40  # DEFAULT_UPDATES
    duration: float = 80  # DEFAULT_MILISECOND_FRAME
    recording_times: int = 8
    frames_per_recording: int = 8


@dataclass
class ExperimentArgs:
    """Experiment tracking and setup arguments"""

    exp_name: str = "ppo"
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "extended-mind"
    wandb_entity: str = "glen-berseth"
    device: int = 0
    profile: bool = False
    total_timesteps: int = 10000000
    num_ppo_steps: int = 128
    no_train: bool = False
    params_path: str = None
    description: str = ""
    conv_count: int = 3
    maxpool_count: int = 2


@dataclass
class Args:
    """Main arguments container"""

    ppo: PPOArgs
    env: EnvArgs
    viz: VisualizationArgs
    exp: ExperimentArgs

    # Computed at runtime
    batch_size: int = 0  # num_envs * num_ppo_steps
    minibatch_size: int = 0  # batch_size // num_minibatches
    num_iterations: int = 0  # total_timesteps // batch_size

    def __post_init__(self):
        """Compute derived values after initialization"""
        self.batch_size = self.env.num_envs * self.exp.num_ppo_steps
        self.minibatch_size = self.batch_size // self.ppo.num_minibatches
        self.num_iterations = self.exp.total_timesteps // self.batch_size
