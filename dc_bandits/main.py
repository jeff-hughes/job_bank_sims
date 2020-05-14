import time
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from bandits.core.contextual_bandit2 import run_contextual_bandit
from bandits.data.data_sampler import sample_mushroom_data

# algorithms
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling

def context_bandit_gen_context():
    """Randomly selects a gender (0 or 1) and normalized age (between 18
    and 65) from a uniform distribution.
    """
    gender = np.random.choice([0, 1])
    age = np.random.choice(range(18, 65))
    norm_age = (age - 18) / (65 - 18)
    return np.array([gender, norm_age])

def context_bandit_prob(ctx, action):
    """Given a context (gender, age) and an action (0 or 1), provides the
    probability of receiving a reward. Probability is determined based on
    a logistic function with pre-specified parameters.
    """
    gender = ctx[0]
    age = ctx[1]
    alpha = (-0.3) + (-0.6)*action + (-0.4)*gender + 0.8*gender*action + 0.7*age + 0.5*age*action
    prob = 1 / (1 + np.exp(-alpha))
    return prob

def context_bandit_reward(ctx, action):
    """Given a context (gender, age) and an action (0 or 1), provides a
    reward based on a pre-specified probability distribution.
    """
    prob = context_bandit_prob(ctx, action)
    return int(np.random.rand() < prob)

data_route = "datasets"

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/bandits/logs/', 'Base directory to save output')

def display_results(algos, opt_rewards, opt_actions, h_rewards, times, name):
    """Displays summary statistics of the performance of each algorithm."""

    print("===================================================================")
    print("{} bandit completed after {:.3f} seconds.".format(name, np.sum(times)))
    print("-------------------------------------------------------------------")

    performance_pairs = []
    for j, a in enumerate(algos):
        performance_pairs.append((a.name, np.sum(h_rewards[:, j]), times[j]))
    performance_pairs = sorted(performance_pairs,
                               key=lambda elt: elt[1],
                               reverse=True)
    for i, (name, reward, time) in enumerate(performance_pairs):
        print("{:2}) {:20} | time = {:7.2f} | total reward = {:10}".format(i+1, name, time, int(reward)))

    print("-------------------------------------------------------------------")
    print(f"Optimal total reward = {np.sum(opt_rewards)}.")
    print("Frequency of optimal actions (action, frequency):")
    print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
    print("===================================================================\n")


def main(_):
    # create dataset
    data_type = "job_bank"
    num_contexts = 2000
    num_actions = 2
    context_dim = 2
    dataset = np.empty((num_contexts, 4), dtype=np.float)
    opt_actions = np.empty(num_contexts, dtype=np.int)
    opt_rewards = np.empty(num_contexts, dtype=np.float)
    for iter in range(num_contexts):
        ctx = context_bandit_gen_context()
        all_probs = [context_bandit_prob(ctx, a) for a in range(num_actions)]
        optimal = np.argmax(all_probs)
        rewards = [context_bandit_reward(ctx, a) for a in range(num_actions)]
        dataset[iter, :] = np.array(ctx.tolist() + rewards)
        opt_actions[iter] = optimal
        opt_rewards[iter] = all_probs[optimal]
    
    hparams = HParams(num_actions=num_actions)

    hparams_linear = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        a0=6,
        b0=6,
        lambda_prior=0.25,
        initial_pulls=2)

    hparams_rms = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        optimizer='RMS',
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=50,
        training_epochs=100,
        p=0.95,
        q=3,
        verbose=False)

    hparams_dropout = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        optimizer='RMS',
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=50,
        training_epochs=100,
        use_dropout=True,
        keep_prob=0.80,
        verbose=False)

    hparams_bbb = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        optimizer='RMS',
        use_sigma_exp_transform=True,
        cleared_times_trained=10,
        initial_training_steps=100,
        noise_sigma=0.1,
        reset_lr=False,
        training_freq=50,
        training_epochs=100,
        verbose=False)

    hparams_nlinear = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=1,
        training_freq_network=50,
        training_epochs=100,
        a0=6,
        b0=6,
        lambda_prior=0.25,
        verbose=False)

    hparams_nlinear2 = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=10,
        training_freq_network=50,
        training_epochs=100,
        a0=6,
        b0=6,
        lambda_prior=0.25,
        verbose=False)

    hparams_pnoise = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        optimizer='RMS',
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=50,
        training_epochs=100,
        noise_std=0.05,
        eps=0.1,
        d_samples=300,
        verbose=False)

    hparams_alpha_div = HParams(
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        optimizer='RMS',
        use_sigma_exp_transform=True,
        cleared_times_trained=10,
        initial_training_steps=100,
        noise_sigma=0.1,
        reset_lr=False,
        training_freq=50,
        training_epochs=100,
        alpha=1.0,
        k=20,
        prior_variance=0.1,
        verbose=False)

    hparams_gp = HParams(
        num_actions=num_actions,
        num_outputs=num_actions,
        context_dim=context_dim,
        reset_lr=False,
        learn_embeddings=True,
        max_num_points=1000,
        show_training=False,
        freq_summary=1000,
        batch_size=512,
        keep_fixed_after_max_obs=True,
        training_freq=50,
        initial_pulls=2,
        training_epochs=100,
        lr=0.01,
        buffer_s=-1,
        initial_lr=0.001,
        lr_decay_rate=0.0,
        optimizer='RMS',
        task_latent_dim=5,
        activate_decay=False,
        verbose=False)

    algos = [
        UniformSampling('Uniform Sampling', hparams),
        FixedPolicySampling('Fixed 1', [0.75, 0.25], hparams),
        FixedPolicySampling('Fixed 2', [0.25, 0.75], hparams),
        PosteriorBNNSampling('RMS', hparams_rms, 'RMSProp'),
        PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
        PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
        NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
        NeuralLinearPosteriorSampling('NeuralLinear2', hparams_nlinear2),
        LinearFullPosteriorSampling('LinFullPost', hparams_linear),
        BootstrappedBNNSampling('BootRMS', hparams_rms),
        ParameterNoiseSampling('ParamNoise', hparams_pnoise),
        PosteriorBNNSampling('BBAlphaDiv', hparams_alpha_div, 'AlphaDiv'),
        PosteriorBNNSampling('MultitaskGP', hparams_gp, 'GP'),
    ]

    _, h_rewards, times = run_contextual_bandit(context_dim, num_actions, dataset, algos)

    display_results(algos, opt_rewards, opt_actions, h_rewards, times, data_type)


if __name__ == '__main__':
  app.run(main)