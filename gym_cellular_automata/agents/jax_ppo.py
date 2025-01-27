# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import orbax.checkpoint

from jax.sharding import PartitionSpec as P, NamedSharding
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from jax.experimental import host_callback
import math
from gym_cellular_automata.agents.args import Args
from functools import partial


def policy_printer(args):
    policy_loss = args[0]
    if policy_loss == 0.00:
        import pdb

        pdb.set_trace()
    print(f"Policy Loss: {policy_loss:.2f}")


def debug_printer(args):
    newlogprob, logprob = args
    print(f"\nNewlogprob - {newlogprob}")
    print(f"Logprob  - min: {logprob}")


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


class Network(nn.Module):
    @nn.compact
    def __call__(self, grid):
        if grid.shape[1] <= 16:
            # For small grids, use smaller strides
            x = nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=(2, 2),  # Reduced stride
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(grid)
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(2, 2),  # Reduced stride
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        else:
            x = nn.Conv(
                32,
                kernel_size=(8, 8),
                strides=(4, 4),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(grid)
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dims: Sequence[int]  # Regular categorical dimensions
    choose_k: Sequence[tuple[int, int]] = (
        None  # List of (n, k) tuples for "choose k from n" actions
    )

    @nn.compact
    def __call__(self, x):
        # features = nn.Dense(
        #     64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(x)
        # features = nn.relu(features)
        # features = nn.Dense(
        #     64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(features)
        # features = nn.relu(features)
        features = x

        # Create logits for each action dimension
        logits = []

        # Handle regular categorical actions
        for dim in self.action_dims:
            head = nn.Dense(dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
                features
            )
            logits.append(head)

        # Handle "choose k" actions
        if len(self.choose_k) > 0:
            for n, k in self.choose_k:
                # Calculate number of possible combinations (n choose k)
                num_combinations = sum(math.comb(n, i) for i in range(k + 1))
                head = nn.Dense(
                    num_combinations,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                )(features)
                logits.append(head)

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
    recent_returns: jnp.array = None
    recent_lengths: jnp.array = None
    recent_idx: jnp.array = 0
    # Current episode tracking
    current_day_correct: jnp.array = None  # Track current episodes
    current_night_correct: jnp.array = None
    current_day_steps: jnp.array = None
    current_night_steps: jnp.array = None
    # Recent episode stats
    recent_day_correct: jnp.array = None  # Store completed episodes
    recent_night_correct: jnp.array = None
    recent_day_steps: jnp.array = None
    recent_night_steps: jnp.array = None


def build_storage_return(storage, env):
    # Save grid observations to JSON file
    grid_obs = jax.device_get(storage.grid_obs).transpose(
        1, 0, *range(2, storage.grid_obs.ndim)
    )  # Get array from device and swap first two dims
    position_obs = jax.device_get(storage.position_obs).transpose(
        1, 0, 2
    )  # Get array from device and swap first two dims
    contexts = {}
    for context_key in env.per_env_context_keys:
        if context_key in storage.contexts.keys():
            contexts[context_key] = jax.device_get(
                storage.contexts[context_key]
            ).transpose(1, 0, *range(2, storage.contexts[context_key].ndim))
    contexts["time"] = jax.device_get(storage.contexts["time"]).transpose(1, 0)
    return {"grid_obs": grid_obs, "position_obs": position_obs, "contexts": contexts}


def run_rollout_loop(
    env,
    args: Args,  # Accept structured Args directly
    key=jax.random.key(0),
):
    """Run PPO training loop

    Args:
        env: Environment to train on
        args: Structured arguments container
        key: JAX random key
    """
    host = os.environ.get("EXTENDED_MIND_HOST", "")
    if not host:
        host = "local"
    run_name = f"{args.env.env_id}__lr={args.ppo.learning_rate}__host={host}__seed={args.exp.seed}__speed={args.env.speed_multiplier}__size={args.env.size}__{int(time.time())}"
    checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True
    )
    registry = orbax.checkpoint.handlers.DefaultCheckpointHandlerRegistry()
    registry.add(
        "default",
        orbax.checkpoint.args.StandardSave,
        orbax.checkpoint.StandardCheckpointHandler,
    )
    if args.exp.track:
        import wandb

        wandb.init(
            project=args.exp.wandb_project_name,
            entity=args.exp.wandb_entity,
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
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
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
        episode_returns=jnp.zeros(args.env.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.env.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        recent_returns=jnp.zeros(10, dtype=jnp.float32),
        recent_lengths=jnp.zeros(10, dtype=jnp.int32),
        recent_idx=jnp.array(0, dtype=jnp.int32),
        # Current episode tracking
        current_day_correct=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        current_night_correct=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        current_day_steps=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        current_night_steps=jnp.zeros(args.env.num_envs, dtype=jnp.int32),
        # Recent episode stats
        recent_day_correct=jnp.zeros(10, dtype=jnp.int32),
        recent_night_correct=jnp.zeros(10, dtype=jnp.int32),
        recent_day_steps=jnp.zeros(10, dtype=jnp.int32),
        recent_night_steps=jnp.zeros(10, dtype=jnp.int32),
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

        # Track correct extension usage (last action in tuple)
        is_night = obs[1]["per_env_context"]["is_night"]
        extension_action = action[:, -1]  # Get last action from tuple

        # Count correct actions (2 for day, 1 for night)
        day_correct = (1 - is_night) * (extension_action == 2)
        night_correct = is_night * (extension_action == 1)

        # Update current episode stats
        current_episode_stats = current_episode_stats.replace(
            current_day_correct=current_episode_stats.current_day_correct + day_correct,
            current_night_correct=current_episode_stats.current_night_correct
            + night_correct,
            current_day_steps=current_episode_stats.current_day_steps + (1 - is_night),
            current_night_steps=current_episode_stats.current_night_steps + is_night,
        )

        def update_recent_stats(stats, returns, lengths, mask):
            # Instead of using boolean indexing, we'll use a scan
            num_finished = jnp.sum(mask)

            def body_fun(carry, env_idx):
                stats, returns, lengths, mask = carry
                # Get the index of the next finished episode
                new_idx = stats.recent_idx
                new_returns = jnp.where(
                    mask[env_idx],
                    stats.recent_returns.at[new_idx].set(returns[env_idx]),
                    stats.recent_returns,
                )
                new_lengths = jnp.where(
                    mask[env_idx],
                    stats.recent_lengths.at[new_idx].set(lengths[env_idx]),
                    stats.recent_lengths,
                )
                # Update recent day/night stats when episode finishes
                new_day_correct = jnp.where(
                    mask[env_idx],
                    stats.recent_day_correct.at[new_idx].set(
                        stats.current_day_correct[env_idx]
                    ),
                    stats.recent_day_correct,
                )
                new_night_correct = jnp.where(
                    mask[env_idx],
                    stats.recent_night_correct.at[new_idx].set(
                        stats.current_night_correct[env_idx]
                    ),
                    stats.recent_night_correct,
                )
                new_day_steps = jnp.where(
                    mask[env_idx],
                    stats.recent_day_steps.at[new_idx].set(
                        stats.current_day_steps[env_idx]
                    ),
                    stats.recent_day_steps,
                )
                new_night_steps = jnp.where(
                    mask[env_idx],
                    stats.recent_night_steps.at[new_idx].set(
                        stats.current_night_steps[env_idx]
                    ),
                    stats.recent_night_steps,
                )
                new_idx = (new_idx + mask[env_idx].astype(jnp.int32)) % 10

                return (
                    (
                        stats.replace(
                            recent_returns=new_returns,
                            recent_lengths=new_lengths,
                            recent_day_correct=new_day_correct,
                            recent_night_correct=new_night_correct,
                            recent_day_steps=new_day_steps,
                            recent_night_steps=new_night_steps,
                            recent_idx=new_idx,
                        ),
                        returns,
                        lengths,
                        mask,
                    ),
                    None,
                )

            # Scan through all environments
            (final_stats, _, _, _), _ = jax.lax.scan(
                body_fun, (stats, returns, lengths, mask), jnp.arange(mask.shape[0])
            )

            # Reset current episode stats for finished episodes
            return final_stats.replace(
                recent_idx=(stats.recent_idx + num_finished) % 10,
                # Reset current stats for finished episodes
                # current_day_correct=final_stats.current_day_correct * (1 - mask),
                # current_night_correct=final_stats.current_night_correct * (1 - mask),
                # current_day_steps=final_stats.current_day_steps * (1 - mask),
                # current_night_steps=final_stats.current_night_steps * (1 - mask),
            )

        finished_mask = next_info["terminated"] + next_info["TimeLimit.truncated"]
        current_episode_stats = update_recent_stats(
            current_episode_stats, new_episode_return, new_episode_length, finished_mask
        )
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
        ) = env.conditional_reset(step_tuple, action)
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
            - (count // (args.ppo.num_minibatches * args.ppo.update_epochs))
            / args.num_iterations
        )
        return learning_rate * frac

    network = Network()
    actor = Actor(action_dims=env.action_space.nvec[0], choose_k=env.extension_choices)
    critic = Critic()

    grid_sample, context = env.observation_space.sample()
    network_params = network.init(network_key, grid_sample)
    grid_sample, context = env.observation_space.sample()
    actor_params = actor.init(
        actor_key,
        network.apply(
            network_params,
            grid_sample,
        ),
    )
    grid_sample, context = env.observation_space.sample()
    critic_params = critic.init(
        critic_key,
        network.apply(
            network_params,
            grid_sample,
        ),
    )
    print(
        sum(x.size for x in jax.tree_leaves(network_params)),
        sum(x.size for x in jax.tree_leaves(actor_params)),
        sum(x.size for x in jax.tree_leaves(critic_params)),
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
            optax.clip_by_global_norm(args.ppo.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=(
                    linear_schedule if args.ppo.anneal_lr else args.ppo.learning_rate
                ),
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
            if args.viz.gif:
                for context_key in env.per_env_context_keys:
                    contexts[context_key] = jax.device_put(
                        jnp.zeros(
                            (args.exp.num_ppo_steps,)
                            + per_env_context[context_key].shape
                        ),
                        NamedSharding(mesh, P("devices", None)),
                    )
                contexts["time"] = jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps,) + context["time"].shape),
                    NamedSharding(mesh, P("devices", None)),
                )
            storage = Storage(
                grid_obs=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps,) + grid.shape),
                    NamedSharding(mesh, P("devices", None, None, None)),
                ),
                position_obs=jax.device_put(
                    jnp.zeros(
                        (args.exp.num_ppo_steps,) + context["position"].shape,
                        dtype=jnp.int32,
                    ),
                    NamedSharding(mesh, P("devices", None)),
                ),
                contexts=contexts,
                actions=jax.device_put(
                    jnp.zeros(
                        (args.exp.num_ppo_steps,) + env.total_action_space.shape,
                        dtype=jnp.int32,
                    ),
                    NamedSharding(mesh, P("devices", None)),
                ),
                logprobs=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, *env.total_action_space.shape)),
                    NamedSharding(mesh, P("devices", None)),
                ),
                dones=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                values=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                advantages=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                returns=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
                rewards=jax.device_put(
                    jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
                    NamedSharding(mesh, P("devices")),
                ),
            )
    else:
        contexts = {}
        if args.viz.gif:
            for context_key in env.per_env_context_keys:
                if context_key in per_env_context.keys():
                    contexts[context_key] = jnp.zeros(
                        (args.exp.num_ppo_steps,) + per_env_context[context_key].shape
                    )
            contexts["time"] = jax.device_put(
                jnp.zeros((args.exp.num_ppo_steps,) + context["time"].shape),
            )
        storage = Storage(
            grid_obs=jnp.zeros((args.exp.num_ppo_steps,) + grid.shape),
            position_obs=jnp.zeros(
                (args.exp.num_ppo_steps,) + context["position"].shape, dtype=jnp.int32
            ),
            actions=jnp.zeros(
                (args.exp.num_ppo_steps,) + env.total_action_space.shape,
                dtype=jnp.int32,
            ),
            logprobs=jnp.zeros((args.exp.num_ppo_steps, *env.total_action_space.shape)),
            dones=jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
            values=jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
            advantages=jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
            returns=jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
            rewards=jnp.zeros((args.exp.num_ppo_steps, args.env.num_envs)),
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
        )
        action_logits = actor.apply(agent_state.params["actor_params"], hidden)

        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        actions = []
        logprobs = []
        for i in range(len(action_logits)):
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey, shape=action_logits[i].shape)
            logits = action_logits[i]
            action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
            actions.append(action)
            logprobs.append(logprob)
        actions = jnp.stack(actions, axis=1)

        logprobs = jnp.concat(logprobs)

        value = critic.apply(agent_state.params["critic_params"], hidden)

        new_contexts = dict(storage.contexts)  # Create a new dict
        if args.viz.gif:
            per_env_context = context["per_env_context"]
            for context_key in env.per_env_context_keys:
                if (
                    context_key in per_env_context.keys()
                    and context_key in storage.contexts.keys()
                ):
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
        hidden = network.apply(params["network_params"], grid)
        logits_set = actor.apply(params["actor_params"], hidden)
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        logprobs = []
        entropies = []
        for i in range(len(logits_set)):
            logit = logits_set[i]
            act = action[:, i]
            logprob = jax.nn.log_softmax(logit)[jnp.arange(act.shape[0]), act]
            logits = logit - jax.scipy.special.logsumexp(logit, axis=-1, keepdims=True)
            logits = logits.clip(min=jnp.finfo(logits.dtype).min)
            p_log_p = logits * jax.nn.softmax(logits)
            entropy = -p_log_p.sum(-1)
            logprobs.append(logprob)
            entropies.append(entropy)
        logprobs = jnp.stack(logprobs, axis=1)
        entropies = jnp.stack(entropies, axis=1)
        # logprobs = logprobs.sum(axis=1).squeeze()

        value = critic.apply(params["critic_params"], hidden).squeeze()
        return logprobs, entropies, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once, gamma=args.ppo.gamma, gae_lambda=args.ppo.gae_lambda
    )

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params["critic_params"],
            network.apply(agent_state.params["network_params"], next_obs[0]),
        ).squeeze(-1)

        advantages = jnp.zeros((args.env.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    @jax.jit
    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)

        # Calculate approx_kl per action and take mean
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.ppo.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        # Policy loss calculated per action
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(
            ratio, 1 - args.ppo.clip_coef, 1 + args.ppo.clip_coef
        )

        # Take mean across both batch and action dimensions
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        if args.ppo.clip_vloss:
            v_loss_unclipped = 0.5 * ((newvalue - mb_returns) ** 2).mean()
            v_clipped = mb_values + jnp.clip(
                newvalue - mb_values,
                -args.ppo.clip_coef,
                args.ppo.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ppo.ent_coef * entropy_loss + v_loss * args.ppo.vf_coef
        return loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            jax.lax.stop_gradient(approx_kl),
        )

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.ppo.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)
            shuffled_storage = shuffled_storage.replace(
                advantages=jnp.repeat(
                    jnp.expand_dims(shuffled_storage.advantages, axis=2),
                    env.total_action_space.shape[-1],
                    axis=2,
                )
            )

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = (
                    ppo_loss_grad_fn(
                        agent_state.params,
                        (minibatch.grid_obs, minibatch.position_obs),
                        minibatch.actions,
                        minibatch.logprobs,
                        minibatch.advantages,
                        minibatch.returns,
                        minibatch.values,
                    )
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    grads,
                )

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = (
                jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            )
            return (agent_state, key), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                grads,
            )

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = (
            jax.lax.scan(
                update_epoch, (agent_state, key), (), length=args.ppo.update_epochs
            )
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

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

    next_done = jnp.full(args.env.num_envs, False)
    next_info = {
        "TimeLimit.truncated": jnp.full(args.env.num_envs, False),
        "terminated": jnp.full(args.env.num_envs, False),
        "steps_elapsed": jnp.zeros(args.env.num_envs),
        "reward_accumulated": jnp.zeros(args.env.num_envs),
        "reward": jnp.zeros(args.env.num_envs),
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

            global_step += args.env.num_envs
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
        return jax.lax.fori_loop(0, args.exp.num_ppo_steps, body_fun, init_carry)

    progress_bar = tqdm(
        range(1, args.exp.total_timesteps // args.env.num_envs + 1),
        desc="Training",
        postfix={
            "SPS": "0",
            "avg_episodic_return": "0.0",
            "games_finished": 0,
            "avg_episode_length": "0.0",
            "avg_return_per_timestep": "0.0",
            "recent_20_return": "0.0",
            "recent_20_length": "0.0",
            "value_loss": "0.0",
            "policy_loss": "0.0",
            "entropy_loss": "0.0",
            "approx_kl": "0.0",
        },
    )
    storage_returns = []
    action_count = [0 for i in range(9)]
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
            if args.viz.gif:
                checkpoint_spacing = max(
                    1, args.exp.total_timesteps // args.viz.recording_times
                )
                frame_iteration = iteration - 1  # Offset iteration by 1

                if (
                    frame_iteration % checkpoint_spacing == 0
                    and len(storage_returns) < args.viz.recording_times
                ):
                    # print("Iteration video start", iteration)
                    storage_returns.append([])
                    storage_returns[-1].append(build_storage_return(storage, env))
                # If we're within 7 steps after a checkpoint, keep adding steps
                elif (
                    len(storage_returns) > 0
                    and len(storage_returns[-1]) < args.viz.frames_per_recording
                    and frame_iteration % checkpoint_spacing
                    < args.viz.frames_per_recording
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

            # actions = jax.device_get(storage.actions)
            # for action in actions:
            #     action_count[action[0][0].item()] += 1

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
            writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
            writer.add_scalar(
                "losses/entropy", entropy_loss[-1, -1].item(), global_step
            )
            writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.env.num_envs
                    * args.exp.num_ppo_steps
                    / (time.time() - iteration_time_start)
                ),
                global_step,
            )
            # Calculate recent statistics
            recent_returns = jax.device_get(sharded_stats.recent_returns)
            recent_lengths = jax.device_get(sharded_stats.recent_lengths)

            # Calculate per-episode return/timestep ratios
            mask = (recent_returns != 0) & (recent_lengths != 0)
            if jnp.any(mask):
                per_episode_ratios = jnp.where(
                    mask, recent_returns / jnp.maximum(recent_lengths, 1e-8), 0
                )
                avg_return_per_timestep = jnp.mean(per_episode_ratios[mask]).item()
                recent_avg_return = jnp.mean(recent_returns[mask]).item()
                recent_avg_length = jnp.mean(recent_lengths[mask]).item()
                recent_standard_error = (
                    jnp.std(recent_returns[mask]) / jnp.sqrt(len(recent_returns[mask]))
                ).item()
            else:
                avg_return_per_timestep = 0
                recent_avg_return = 0
                recent_avg_length = 0
                recent_standard_error = 0

            writer.add_scalar(
                "charts/avg_return_per_timestep",
                avg_return_per_timestep,
                global_step,
            )

            writer.add_scalar(
                "charts/recent_avg_return",
                recent_avg_return,
                global_step,
            )
            writer.add_scalar(
                "charts/recent_avg_length",
                recent_avg_length,
                global_step,
            )
            writer.add_scalar(
                "charts/recent_standard_error",
                recent_standard_error,
                global_step,
            )

            # Calculate extension correctness statistics
            recent_day_correct = jax.device_get(sharded_stats.recent_day_correct)
            recent_night_correct = jax.device_get(sharded_stats.recent_night_correct)
            recent_day_steps = jax.device_get(sharded_stats.recent_day_steps)
            recent_night_steps = jax.device_get(sharded_stats.recent_night_steps)

            mask = (recent_returns != 0) & (recent_lengths != 0)
            if jnp.any(mask):
                # Calculate percentage of correct actions
                day_correct_rate = jnp.where(
                    recent_day_steps > 0,
                    recent_day_correct / (recent_day_steps + 1e-8),
                    0.0,
                )[mask].mean()
                night_correct_rate = jnp.where(
                    recent_night_steps > 0,
                    recent_night_correct / (recent_night_steps + 1e-8),
                    0.0,
                )[mask].mean()

                writer.add_scalar(
                    "extensions/night_correct_rate",
                    night_correct_rate.item() * 100,
                    global_step,
                )
                writer.add_scalar(
                    "extensions/day_correct_rate",
                    day_correct_rate.item() * 100,
                    global_step,
                )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "SPS": sps,
                    # "avg_return": f"{avg_episodic_return:.2f}",
                    # "current_return": f"{current_episodic_return:.2f}",
                    "games_finished": int(jax.device_get(total_finished)),
                    # "avg_returned_episode_length": f"{avg_returned_episode_length:.2f}",
                    "avg_episodic_return": f"{avg_episodic_return:.2f}",
                    "avg_return_per_timestep": f"{avg_return_per_timestep:.4f}",
                    "recent_10_return": f"{recent_avg_return:.2f}",
                    "recent_10_length": f"{recent_avg_length:.2f}",
                    "recent_10_standard_error": f"{recent_standard_error:.2f}",
                    "value_loss": f"{v_loss[-1, -1].item():.4f}",
                    "policy_loss": f"{pg_loss[-1, -1].item():.4f}",
                    "entropy_loss": f"{entropy_loss[-1, -1].item():.4f}",
                    "approx_kl": f"{approx_kl[-1, -1].item():.4f}",
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
    actor = Actor(action_dims=env.action_space.nvec[0], choose_k=env.extension_choices)

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

    network_params = network.init(network_key, grid_sample)
    actor_params = actor.init(
        actor_key,
        network.apply(network_params, grid_sample),
    )
    critic_params = critic.init(
        critic_key,
        network.apply(network_params, grid_sample),
    )
    print(
        sum(x.size for x in jax.tree_leaves(network_params)),
        sum(x.size for x in jax.tree_leaves(actor_params)),
        sum(x.size for x in jax.tree_leaves(critic_params)),
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
        )

        # Get action logits
        action_logits = actor.apply(restored_state["actor_params"], hidden)

        actions = []
        for logits in action_logits:
            actions.append(jnp.argmax(logits, axis=-1))

        return jnp.expand_dims(jnp.concatenate(actions), axis=0)

    return get_action
