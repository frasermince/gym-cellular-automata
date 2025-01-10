# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import pickle
import orbax.checkpoint

from jax.sharding import PartitionSpec as P, NamedSharding
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

padding_type = "SAME"
# padding_type = "VALID"

if jax.devices()[0].platform == "tpu":
    jax.config.update("jax_platform_name", "tpu")
    jax.config.update("jax_enable_x64", False)  # Use float32/bfloat16 on TPU
# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771

SHARD_STORAGE = False
SHOULD_SHARD = False
# jax.config.update("jax_default_matmul_precision", "bfloat16")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "firefighter"
    """the wandb's project name"""
    wandb_entity: str = "unchart"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Firefighter"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class Network(nn.Module):
    log_grid_shapes: bool = False

    @nn.compact
    def __call__(self, grid, position):
        if self.log_grid_shapes:
            print(f"Initial grid shape: {grid.shape}")
            print(f"Initial position shape: {position.shape}")

        grid = grid[..., None]
        if self.log_grid_shapes:
            print(f"Grid shape after adding channel dim: {grid.shape}")

        grid = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding=padding_type,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after first conv: {grid.shape}")

        grid = nn.relu(grid)
        grid = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=padding_type,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after second conv: {grid.shape}")

        grid = nn.relu(grid)
        grid = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=padding_type,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(grid)
        if self.log_grid_shapes:
            print(f"Grid shape after third conv: {grid.shape}")

        grid = nn.relu(grid)

        grid_features = grid.reshape(grid.shape[0], -1)
        if self.log_grid_shapes:
            print(f"Grid features shape after flatten: {grid_features.shape}")

        grid_features = nn.Dense(256)(grid_features)
        if self.log_grid_shapes:
            print(f"Grid features shape after dense: {grid_features.shape}")

        grid_features = nn.relu(grid_features)

        # Process position input
        position_features = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(position)
        if self.log_grid_shapes:
            print(f"Position features shape after dense: {position_features.shape}")

        position_features = nn.relu(position_features)

        # Combine features
        combined = jnp.concatenate([grid_features, position_features], axis=-1)
        if self.log_grid_shapes:
            print(f"Combined features shape: {combined.shape}")

        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            combined
        )
        if self.log_grid_shapes:
            print(f"Final features shape: {x.shape}")

        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        features = nn.Dense(64)(x)
        features = nn.relu(features)
        features = nn.Dense(64)(features)
        features = nn.relu(features)

        max_dim = max(self.action_dims)

        # Create separate Dense layers for each action dimension
        logits = []
        for dim in self.action_dims:
            head = nn.Dense(dim)(features)
            if max_dim > dim:
                # This -1e10 and masking concerns me a little. Make sure this
                # won't mess up action sampling.
                head = jnp.pad(
                    head, ((0, 0), (0, max_dim - dim)), constant_values=-1e10
                )
            logits.append(head)

        # Stack along action dimension
        logits = jnp.stack(logits, axis=1)  # [batch, num_action_types, max_dim]

        return logits


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    grid_obs: jnp.array
    position_obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    contexts: dict


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array
    amount_finished: jnp.array = 0


def build_storage_return(storage, env):
    # Save grid observations to JSON file
    grid_obs = jax.device_get(storage.grid_obs).transpose(
        1, 0, 2, 3
    )  # Get array from device and swap first two dims
    position_obs = jax.device_get(storage.position_obs).transpose(
        1, 0, 2
    )  # Get array from device and swap first two dims
    contexts = {}
    for context_key in env.per_env_context_keys:
        contexts[context_key] = jax.device_get(storage.contexts[context_key]).transpose(
            1, 0, *range(2, storage.contexts[context_key].ndim)
        )
    contexts["time"] = jax.device_get(storage.contexts["time"]).transpose(1, 0)
    return {"grid_obs": grid_obs, "position_obs": position_obs, "contexts": contexts}


# 100,000 should be sufficient 100,000 / 128 = 781.25
def run_rollout_loop(
    env,
    num_iterations,
    num_envs=8,
    recording_times=8,
    frames_per_recording=8,
    use_gif=False,
    key=jax.random.key(0),
    learning_rate=2.5e-4,
    device_index=0,
):
    args = Args()
    args.batch_size = int(num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True
    )
    registry = orbax.checkpoint.handlers.DefaultCheckpointHandlerRegistry()
    registry.add(
        "default",
        orbax.checkpoint.args.StandardSave,
        orbax.checkpoint.StandardCheckpointHandler,
    )
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    # envs = envpool.make(
    #     args.env_id,
    #     env_type="gym",
    #     num_envs=num_envs,
    #     episodic_life=True,
    #     reward_clip=True,
    #     seed=args.seed,
    # )
    # envs.num_envs = num_envs
    # envs.action_space = envs.action_space
    # envs.observation_space = envs.observation_space
    # envs.is_vector_env = True
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )
    # handle, recv, send, step_env = envs.xla()

    @jax.jit
    def step_env_wrapped(current_episode_stats, action, obs, info):
        step_tuple = env.stateless_step(action, obs, info)
        (
            next_obs,
            reward,
            next_done,
            truncated,
            next_info,
        ) = step_tuple

        new_episode_return = current_episode_stats.episode_returns + next_info["reward"]
        new_episode_length = current_episode_stats.episode_lengths + 1
        episode_stats = current_episode_stats.replace(
            amount_finished=current_episode_stats.amount_finished
            + jnp.sum(next_info["terminated"]),
            episode_returns=(new_episode_return)
            * (1 - next_info["terminated"])
            * (1 - next_info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length)
            * (1 - next_info["terminated"])
            * (1 - next_info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                next_info["terminated"] + next_info["TimeLimit.truncated"],
                new_episode_return,
                current_episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_info["terminated"] + next_info["TimeLimit.truncated"],
                new_episode_length,
                current_episode_stats.returned_episode_lengths,
            ),
        )
        # if jnp.any(next_info["terminated"]):
        #     import pdb

        #     pdb.set_trace()

        (
            next_obs,
            reward,
            next_done,
            truncated,
            next_info,
        ) = env.conditional_reset(step_tuple)
        # episode_stats = episode_stats.replace(
        #     episode_returns=(new_episode_return)
        #     * (1 - next_info["terminated"])
        #     * (1 - next_info["TimeLimit.truncated"])
        # )

        return episode_stats, (
            next_obs,
            reward,
            next_done,
            next_info,
        )

    # assert isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs))
            / args.num_iterations
        )
        return learning_rate * frac

    network = Network()
    actor = Actor(action_dims=env.action_space.nvec[0])
    critic = Critic()

    grid_sample, context = env.observation_space.sample()
    network_params = network.init(
        network_key, grid_sample, jnp.array(context["position"])
    )
    grid_sample, context = env.observation_space.sample()
    actor_params = actor.init(
        actor_key,
        network.apply(
            network_params,
            grid_sample,
            jnp.array(context["position"]),
        ),
    )
    grid_sample, context = env.observation_space.sample()
    critic_params = critic.init(
        critic_key,
        network.apply(
            network_params,
            grid_sample,
            jnp.array(context["position"]),
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.freeze(
            {
                "network_params": network_params,
                "actor_params": actor_params,
                "critic_params": critic_params,
            }
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else learning_rate,
                eps=1e-5,
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    grid, context = env.observation_space
    per_env_context = context["per_env_context"]
    if len(jax.devices()) >= 4 and SHARD_STORAGE and SHOULD_SHARD:
        mesh = jax.make_mesh((4,), ("devices"))
        with mesh:
            contexts = {}
            if use_gif:
                for context_key in env.per_env_context_keys:
                    contexts[context_key] = jax.device_put(
                        jnp.zeros(
                            (args.num_steps,) + per_env_context[context_key].shape
                        ),
                        NamedSharding(mesh, P("devices", None)),
                    )
                contexts["time"] = jax.device_put(
                    jnp.zeros((args.num_steps,) + context["time"].shape),
                    NamedSharding(mesh, P("devices", None)),
                )
            storage = Storage(
                grid_obs=jax.device_put(
                    jnp.zeros((args.num_steps,) + grid.shape),
                    NamedSharding(mesh, P("devices", None, None, None)),
                ),
                position_obs=jax.device_put(
                    jnp.zeros((args.num_steps,) + context["position"].shape),
                    NamedSharding(mesh, P("devices", None)),
                ),
                contexts=contexts,
                actions=jax.device_put(
                    jnp.zeros(
                        (args.num_steps,) + env.action_space.shape,
                        dtype=jnp.int32,
                    ),
                    NamedSharding(mesh, P("devices", None)),
                ),
                logprobs=jax.device_put(
                    jnp.zeros(((args.num_steps,) + env.action_space.shape)),
                    NamedSharding(mesh, P("devices", None)),
                ),
                dones=jax.device_put(
                    jnp.zeros((args.num_steps, num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                values=jax.device_put(
                    jnp.zeros((args.num_steps, num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                advantages=jax.device_put(
                    jnp.zeros((args.num_steps, num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                returns=jax.device_put(
                    jnp.zeros((args.num_steps, num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                rewards=jax.device_put(
                    jnp.zeros((args.num_steps, num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
            )
    else:
        contexts = {}
        if use_gif:
            for context_key in env.per_env_context_keys:
                contexts[context_key] = jnp.zeros(
                    (args.num_steps,) + per_env_context[context_key].shape
                )
            contexts["time"] = jax.device_put(
                jnp.zeros((args.num_steps,) + context["time"].shape),
            )
        storage = Storage(
            grid_obs=jnp.zeros((args.num_steps,) + grid.shape),
            position_obs=jnp.zeros((args.num_steps,) + context["position"].shape),
            actions=jnp.zeros(
                (args.num_steps,) + env.action_space.shape,
                dtype=jnp.int32,
            ),
            logprobs=jnp.zeros(((args.num_steps,) + env.action_space.shape)),
            dones=jnp.zeros((args.num_steps, num_envs)),
            values=jnp.zeros((args.num_steps, num_envs)),
            advantages=jnp.zeros((args.num_steps, num_envs)),
            returns=jnp.zeros((args.num_steps, num_envs)),
            rewards=jnp.zeros((args.num_steps, num_envs)),
            contexts=contexts,
        )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        next_grid_obs, context = next_obs
        next_position_obs = context["position"]
        hidden = network.apply(
            agent_state.params["network_params"],
            next_grid_obs,
            next_position_obs,
        )
        action_logits = actor.apply(agent_state.params["actor_params"], hidden)

        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=action_logits.shape)

        def sample_action(logits, u):
            action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
            return action, logprob

        actions, logprobs = jax.vmap(sample_action, in_axes=(1, 1), out_axes=1)(
            action_logits, u
        )

        value = critic.apply(agent_state.params["critic_params"], hidden)

        new_contexts = dict(storage.contexts)  # Create a new dict
        if use_gif:
            per_env_context = context["per_env_context"]
            for context_key in env.per_env_context_keys:
                new_contexts[context_key] = (
                    storage.contexts[context_key]
                    .at[step]
                    .set(per_env_context[context_key])
                )
            new_contexts["time"] = (
                storage.contexts["time"].at[step].set(context["time"])
            )
        storage = storage.replace(
            grid_obs=storage.grid_obs.at[step].set(next_grid_obs),
            position_obs=storage.position_obs.at[step].set(next_position_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(actions),
            logprobs=storage.logprobs.at[step].set(logprobs),
            values=storage.values.at[step].set(value.squeeze()),
            contexts=new_contexts,
        )
        return storage, actions, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: tuple[np.ndarray, np.ndarray],
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        grid, position = x
        hidden = network.apply(params["network_params"], grid, position)
        logits_set = actor.apply(params["actor_params"], hidden)
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        entropies = []
        logprobs = []

        def process_logits(logit, act):
            logprob = jax.nn.log_softmax(logit)[jnp.arange(act.shape[0]), act]
            logits = logit - jax.scipy.special.logsumexp(logit, axis=-1, keepdims=True)
            logits = logits.clip(min=jnp.finfo(logits.dtype).min)
            p_log_p = logits * jax.nn.softmax(logits)
            entropy = -p_log_p.sum(-1)
            return logprob, entropy

        logprobs, entropies = jax.vmap(process_logits, in_axes=(1, 1), out_axes=1)(
            logits_set, action
        )

        value = critic.apply(params["critic_params"], hidden).squeeze()
        return logprobs, entropies, value

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))

        grid, context = env.observation_space.sample()
        network_output = network.apply(
            agent_state.params["network_params"],
            grid,
            context["position"],
        )
        next_value = critic.apply(
            agent_state.params["critic_params"],
            network_output,
        ).squeeze(-1)

        lastgaelam = jnp.zeros(next_done.shape)

        def gae_step(carry, t):
            lastgaelam, storage = carry
            nextnonterminal = jax.lax.cond(
                t == args.num_steps - 1,
                lambda _: 1.0 - next_done,
                lambda _: 1.0 - storage.dones[t + 1],
                None,
            )
            nextvalues = jax.lax.cond(
                t == args.num_steps - 1,
                lambda _: next_value,
                lambda _: storage.values[t + 1],
                None,
            )
            delta = (
                storage.rewards[t]
                + args.gamma * nextvalues * nextnonterminal
                - storage.values[t]
            )
            lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )

            storage = storage.replace(
                advantages=storage.advantages.at[t].set(lastgaelam)
            )
            return (lastgaelam, storage), None

        (lastgaelam, storage), _ = jax.lax.scan(
            gae_step, (lastgaelam, storage), jnp.arange(args.num_steps - 1, -1, -1)
        )
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_grid_obs = storage.grid_obs.reshape(
            (-1,) + env.observation_space[0].shape[1:]
        )
        b_position_obs = storage.position_obs.reshape(
            (-1,) + env.observation_space[1]["position"].shape[1:]
        )
        b_logprobs = storage.logprobs.reshape((-1,) + (env.action_space.shape[-1],))
        b_actions = storage.actions.reshape((-1,) + (env.action_space.shape[-1],))
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)
        # Generate keys for all epochs
        keys = jax.random.split(key, args.update_epochs + 1)
        key = keys[0]
        permutation_keys = keys[1:]

        num_minibatches = args.batch_size // args.minibatch_size
        # vmap the permutation generation over epoch keys
        permutations = jax.vmap(
            lambda k: jax.random.permutation(k, args.batch_size).reshape(
                (num_minibatches, args.minibatch_size)
            )
        )(permutation_keys)

        if len(jax.devices()) >= 4 and not SHARD_STORAGE and SHOULD_SHARD:
            # print("Sharding visualization grid shape", b_grid_obs.shape)
            # flattened = b_grid_obs.reshape(
            #     -1, b_grid_obs.shape[-2] * b_grid_obs.shape[-1]
            # )
            # print("Sharding visualization flattened grid shape", flattened.shape)
            # jax.debug.visualize_array_sharding(flattened)

            mesh = jax.make_mesh((4,), ("devices"))
            batch_sharding = NamedSharding(mesh, P("devices"))
            b_grid_obs = jax.lax.with_sharding_constraint(b_grid_obs, batch_sharding)
            b_position_obs = jax.lax.with_sharding_constraint(
                b_position_obs, batch_sharding
            )
            b_logprobs = jax.lax.with_sharding_constraint(b_logprobs, batch_sharding)
            b_actions = jax.lax.with_sharding_constraint(b_actions, batch_sharding)
            b_advantages = jax.lax.with_sharding_constraint(
                b_advantages, batch_sharding
            )
            b_returns = jax.lax.with_sharding_constraint(b_returns, batch_sharding)

            permutations = jax.lax.with_sharding_constraint(
                permutations,
                NamedSharding(
                    mesh, P(None, None, "devices")
                ),  # Shard minibatch dimension
            )

        @jax.jit
        def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            mb_advantages = mb_advantages[:, None, None]

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            # TODO: I'm very suspicious of this line. We are meaning the two
            # logprobs for the actions together. Soon as we add extensions this
            # will be a third action. We need to confirm this is right.
            # jax.debug.visualize_array_sharding(
            #     pg_loss1.reshape(pg_loss1.shape[:-2] + (-1,))
            # )
            # jax.debug.visualize_array_sharding(
            #     pg_loss2.reshape(pg_loss2.shape[:-2] + (-1,))
            # )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            # jax.debug.visualize_array_sharding(
            #     pg_loss.reshape(pg_loss.shape[:-2] + (-1,))
            # )

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (
                pg_loss,
                v_loss,
                entropy_loss,
                jax.lax.stop_gradient(approx_kl),
            )

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        def update_epoch(carry, scan_inputs):
            agent_state = carry
            epoch_idx, epoch_permutation = scan_inputs
            # permutation = jax.random.permutation(
            #     subkey, args.batch_size, independent=True
            # ).reshape((num_minibatches, args.minibatch_size))

            def update_minibatch(carry, perm_indices):
                agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl = carry
                (
                    new_loss,
                    (new_pg_loss, new_v_loss, new_entropy_loss, new_approx_kl),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    (b_grid_obs[perm_indices], b_position_obs[perm_indices]),
                    b_actions[perm_indices],
                    b_logprobs[perm_indices],
                    b_advantages[perm_indices],
                    b_returns[perm_indices],
                )
                # print("grads", grads.shape)
                # jax.debug.visualize_array_sharding(
                #     new_loss.reshape(new_loss.shape[:-2] + (-1,))
                # )
                agent_state = agent_state.apply_gradients(grads=grads)
                return (
                    agent_state,
                    loss + new_loss,
                    pg_loss + new_pg_loss,
                    v_loss + new_v_loss,
                    entropy_loss + new_entropy_loss,
                    approx_kl + new_approx_kl,
                ), None

            # Initialize with dummy values for losses since they'll be overwritten
            init_carry = (agent_state, 0.0, 0.0, 0.0, 0.0, 0.0)
            (agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl), _ = (
                jax.lax.scan(update_minibatch, init_carry, epoch_permutation)
            )
            return (agent_state), (
                loss / num_minibatches,
                pg_loss / num_minibatches,
                v_loss / num_minibatches,
                entropy_loss / num_minibatches,
                approx_kl / num_minibatches,
            )

        init_carry = agent_state
        (final_agent_state), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(
            update_epoch, init_carry, (jnp.arange(args.update_epochs), permutations)
        )

        # Take the mean across epochs
        loss = loss.mean()
        pg_loss = pg_loss.mean()
        v_loss = v_loss.mean()
        entropy_loss = entropy_loss.mean()
        approx_kl = approx_kl.mean()
        return (
            final_agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            key,
        )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, report = env.reset()
    if len(jax.devices()) >= 4 and SHOULD_SHARD:
        print(f"SHARDING to {jax.devices()}")
        mesh = jax.make_mesh((4,), ("devices"))
        grid = jax.device_put(next_obs[0], NamedSharding(mesh, P("devices")))
        context = next_obs[1]
        per_env_context = context["per_env_context"]
        for context_key in env.per_env_context_keys:
            per_env_context[context_key] = jax.device_put(
                per_env_context[context_key], NamedSharding(mesh, P("devices"))
            )
        next_obs = (grid, context)

    next_done = jnp.full(num_envs, False)
    next_info = {
        "TimeLimit.truncated": jnp.full(num_envs, False),
        "terminated": jnp.full(num_envs, False),
        "steps_elapsed": jnp.zeros(num_envs),
        "reward_accumulated": jnp.zeros(num_envs),
        "reward": jnp.zeros(num_envs),
    }

    @jax.jit
    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        next_info,
        storage,
        key,
        global_step,
    ):
        # Remove @jax.jit and use jax.lax.fori_loop instead

        def body_fun(step, carry):
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            ) = carry

            global_step += num_envs
            storage, action, key = get_action_and_value(
                agent_state, next_obs, next_done, storage, step, key
            )

            episode_stats, (next_obs, reward, next_done, next_info) = step_env_wrapped(
                episode_stats, action, next_obs, next_info
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

            return (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            )

        init_carry = (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_info,
            storage,
            key,
            global_step,
        )
        return jax.lax.fori_loop(0, args.num_steps, body_fun, init_carry)

    progress_bar = tqdm(
        range(1, num_iterations + 1),
        desc="Training",
        postfix={
            "SPS": "0",
            "avg_return": "0.0",
            # "current_return": "0.0",
            "games_finished": 0,
            "avg_episode_length": "0.0",
            "avg_returned_episode_length": "0.0",
            "avg_return_per_timestep": "0.0",
        },
    )
    storage_returns = []
    with orbax.checkpoint.CheckpointManager(
        "/tmp/flax_ckpt/orbax/managed",
        options=checkpoint_options,
        handler_registry=registry,
    ) as checkpoint_manager:
        for iteration in progress_bar:
            # if len(jax.devices()) >= 4:
            #     flattened = next_obs[0].reshape(-1, next_obs[0].shape[-1])
            #     jax.debug.visualize_array_sharding(flattened)
            iteration_time_start = time.time()
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            ) = rollout(
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_info,
                storage,
                key,
                global_step,
            )
            if use_gif:
                checkpoint_spacing = max(1, num_iterations // recording_times)
                frame_iteration = iteration - 1  # Offset iteration by 1

                if (
                    frame_iteration % checkpoint_spacing == 0
                    and len(storage_returns) < recording_times
                ):
                    # print("Iteration video start", iteration)
                    storage_returns.append([])
                    storage_returns[-1].append(build_storage_return(storage, env))
                # If we're within 7 steps after a checkpoint, keep adding steps
                elif (
                    len(storage_returns) > 0
                    and len(storage_returns[-1]) < frames_per_recording
                    and frame_iteration % checkpoint_spacing < frames_per_recording
                ):
                    # print("Iteration video continue", iteration)
                    storage_returns[-1].append(build_storage_return(storage, env))
            storage = compute_gae(agent_state, next_obs, next_done, storage)
            agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = (
                update_ppo(
                    agent_state,
                    storage,
                    key,
                )
            )
            if len(jax.devices()) >= 4 and SHOULD_SHARD:
                # Gather stats from all devices
                mesh = jax.make_mesh((4,), ("devices",))

                # Create a mesh context for the all_gather operation
                @jax.jit
                def gather_stats(stats):
                    return jax.tree_map(
                        lambda x: (
                            jax.lax.all_gather(x, "devices").reshape(-1)
                            if hasattr(x, "sharding")
                            and x.sharding.spec == P("devices")
                            else x
                        ),
                        stats,
                    )

                with mesh:
                    sharded_stats = gather_stats(episode_stats)
                    total_finished = jax.device_get(episode_stats.amount_finished)
            else:
                # No sharding, use episode_stats directly
                sharded_stats = episode_stats
                total_finished = episode_stats.amount_finished
            avg_episodic_return = np.mean(
                jax.device_get(sharded_stats.returned_episode_returns)
            )
            current_episodic_return = np.mean(
                jax.device_get(sharded_stats.episode_returns)
            )
            avg_episode_length = np.mean(jax.device_get(sharded_stats.episode_lengths))

            avg_returned_episode_length = np.mean(
                jax.device_get(sharded_stats.returned_episode_lengths)
            )

            # Add NaN checks here
            def check_values(name, value):
                if jnp.any(jnp.isnan(value)):
                    print(f"WARNING: NaN detected in {name}")
                    print(f"Value: {value}")
                    # Could also save state for debugging
                    with open(f"debug_nan_{name}_{iteration}.pkl", "wb") as f:
                        pickle.dump(
                            {
                                "agent_state": agent_state,
                                "storage": storage,
                                "stats": episode_stats,
                            },
                            f,
                        )
                    return True
                return False

            has_nan = False
            # Check key metrics
            has_nan |= check_values("policy_loss", pg_loss)
            has_nan |= check_values("value_loss", v_loss)
            has_nan |= check_values("entropy_loss", entropy_loss)
            has_nan |= check_values("approx_kl", approx_kl)
            has_nan |= check_values("advantages", storage.advantages)
            has_nan |= check_values("returns", storage.returns)

            if has_nan:
                print(f"NaN detected at iteration {iteration}")
                break  # Or implement recovery logic
            # print("current:", sharded_stats.episode_lengths)
            # print("returned:", sharded_stats.returned_episode_lengths)
            # Record rewards and metrics
            writer.add_scalar(
                "charts/avg_episodic_return", avg_episodic_return, global_step
            )
            writer.add_scalar(
                "charts/avg_episodic_length",
                np.mean(jax.device_get(sharded_stats.returned_episode_lengths)),
                global_step,
            )
            writer.add_scalar(
                "charts/learning_rate",
                agent_state.opt_state[1].hyperparams["learning_rate"].item(),
                global_step,
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/loss", loss.item(), global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(num_envs * args.num_steps / (time.time() - iteration_time_start)),
                global_step,
            )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "SPS": sps,
                    "avg_return": f"{avg_episodic_return:.2f}",
                    # "current_return": f"{current_episodic_return:.2f}",
                    "games_finished": int(jax.device_get(total_finished)),
                    "avg_returned_episode_length": f"{avg_returned_episode_length:.2f}",
                    "avg_return_per_timestep": f"{avg_episodic_return/max(avg_returned_episode_length, 1e-8):.4f}",
                },
                refresh=True,
            )
            if iteration % 200 == 0 or iteration == 1:
                # Save model parameters using checkpoint manager
                checkpoint_manager.save(
                    iteration,
                    args=orbax.checkpoint.args.StandardSave(agent_state.params),
                )
    # envs.close()
    writer.close()
    return storage_returns, agent_state, run_name


def load_actor(params_path: str, env):
    """
    Loads saved parameters and returns a function that can be used for inference.

    Args:
        params_path: Path to the saved parameters

    Returns:
        function: A function that takes (grid_obs, position_obs) and returns actions
    """
    # Initialize models
    network = Network()
    actor = Actor(action_dims=env.action_space.nvec[0])

    critic = Critic()

    # Load parameters
    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     params_path,  # This should be the base directory, e.g. "/tmp/flax_ckpt/orbax/managed"
    #     orbax_checkpointer,
    #     options,
    # )

    # Create initial state with proper network initialization
    grid_sample, context = env.observation_space.sample()
    network_key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(0), 3)

    network_params = network.init(
        network_key, grid_sample, jnp.array(context["position"])
    )
    actor_params = actor.init(
        actor_key,
        network.apply(network_params, grid_sample, jnp.array(context["position"])),
    )
    critic_params = critic.init(
        critic_key,
        network.apply(network_params, grid_sample, jnp.array(context["position"])),
    )
    agent_state = {
        "network_params": network_params,
        "actor_params": actor_params,
        "critic_params": critic_params,
    }
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)

    # Create registry for standard PyTree handling
    registry = orbax.checkpoint.handlers.DefaultCheckpointHandlerRegistry()
    registry.add(
        "default",
        orbax.checkpoint.args.StandardRestore,
        orbax.checkpoint.StandardCheckpointHandler,
    )

    mesh = jax.make_mesh((1,), ("devices",))
    single_device_sharding = NamedSharding(mesh, P(None))

    def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        if isinstance(x, jax.ShapeDtypeStruct):
            x.sharding = single_device_sharding
        return x

    abstract_state_with_sharding = jax.tree_util.tree_map(set_sharding, agent_state)

    # Get latest checkpoint
    with orbax.checkpoint.CheckpointManager(
        params_path,
        options=options,
        handler_registry=registry,
    ) as manager:
        step = manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {params_path}")

        restored_state = manager.restore(
            step,
            args=orbax.checkpoint.args.StandardRestore(abstract_state_with_sharding),
        )

    @jax.jit
    def get_action(grid_obs, position_obs):
        """
        Get action for a single observation.

        Args:
            grid_obs: Grid observation
            position_obs: Position observation

        Returns:
            actions: Selected actions
        """
        # Get hidden features from network
        hidden = network.apply(
            restored_state["network_params"],
            grid_obs,
            position_obs,
        )

        # Get action logits
        action_logits = actor.apply(restored_state["actor_params"], hidden)

        # Sample action deterministically by taking argmax
        actions = jnp.argmax(action_logits, axis=-1)

        return actions

    return get_action
