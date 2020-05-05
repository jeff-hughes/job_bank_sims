# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from contextual_bandit import ContextBanditSim

# %%
def moving_average(a, n=5):
    """Calculate moving average with window of `n`."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# %%
class QAgent:
    """A basic agent estimating Q-values to weight non-contextual
    multi-armed bandits.

    Parameters:
    num_bandits: Specifies the number of bandits in the current problem.
    eps: Using epsilon-greedy method, this determines the probability of
         the agent choosing an action at random.
    """
    def __init__(self, num_bandits, eps=0.1):
        self.num_bandits = num_bandits
        self.eps = eps

        self.n = np.zeros(num_bandits, dtype=np.int)  # action counts
        self.Q = np.zeros(num_bandits, dtype=np.float)  # Q-values for each action

        self.actions = []
        self.rewards = []

    def choose_action(self, iter_num, _ctx):
        if np.random.rand() < self.eps:
            action = int(np.random.randint(self.num_bandits))
        else:
            action = int(np.argmax(self.Q))
        self.n[action] += 1

        if iter_num == 0:
            totals = np.zeros(self.num_bandits)
            totals[action] = 1
            self.actions.append(totals)
        else:
            totals = self.actions[iter_num-1].copy()
            totals[action] += 1
            self.actions.append(totals)

        return action

    def update_rewards(self, iter_num, _ctx, action, reward):
        self.Q[action] += (1.0/self.n[action]) * (reward - self.Q[action])
        self.rewards.append(reward)

# %%
def multi_bandit_gen_context():
    """Dummy function, returns empty context."""
    return np.array([], dtype=np.float)

def make_multi_bandit_reward(probs):
    """Creates a multi-armed bandit reward function with the specified
    probabilities.
    """
    def multi_bandit_reward(_ctx, _num_bandits, action):
        """Randomly selects a reward for the given action, based on the
        specified probability for that action.
        """
        regret = np.max(probs) - probs[action]
        if np.random.rand() < probs[action]:
            return (1, regret)
        else:
            return (0, regret)
    return multi_bandit_reward

# %%
# basic multi-armed (non-contextual) bandit problem. Four bandits with 
# differing probabilities of receiving reward (either 0 or 1).
num_bandits = 4
multi_bandit_reward = make_multi_bandit_reward([0.2, 0, 0.5, 0.7])

q_agent = QAgent(num_bandits, eps=0.01)
q_multi_sim = ContextBanditSim(multi_bandit_gen_context,
    multi_bandit_reward, num_bandits, q_agent)

results_q = q_multi_sim.run_sim(5000)

print("Mean: {:.2f}, SD: {:.2f}".format(
    np.mean(results_q["rewards"]), np.std(results_q["rewards"])))
print("Min: {:.2f}, Max: {:.2f}".format(
    np.min(results_q["rewards"]), np.max(results_q["rewards"])))

# %%
q_rew_mv_avg = moving_average(q_agent.rewards, n=50)
plt.plot(range(len(q_rew_mv_avg)), q_rew_mv_avg)
plt.show()

print(q_agent.Q)

# %%
bandit_props = np.zeros((5000, num_bandits))
for i, totals in enumerate(q_agent.actions):
    bandit_props[i] = totals / np.sum(totals)

for b in range(num_bandits):
    plt.plot(range(5000), bandit_props[:, b], label=f"Arm {b}")
plt.legend()
plt.show()

# %%
q_regret = [q_multi_sim.regret(T) for T in range(5000)]
plt.plot(range(5000), q_regret)
plt.show()



# %%
class RandomAgent:
    """Chooses an action at random."""
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits
    
    def choose_action(self, _iter_num, _ctx):
        return np.random.choice(self.num_bandits)
    
    def update_rewards(self, _iter_num, _ctx, _action, _reward):
        pass

# %%
class QContextAgent:
    """A basic agent estimating Q-values to weight (discrete) contextual
    multi-armed bandits.

    Instead of a single estimate of Q-values for each bandit, this stores
    an estimate of Q-values for each bandit in a given context. Thus,
    contexts must be discrete groups (e.g., age groups rather than age).

    Parameters:
    num_bandits: Specifies the number of bandits in the current problem.
    age_bins: Numpy array of age groups to split age into. (See
              numpy.digitize() for format.)
    eps: Using epsilon-greedy method, this determines the probability of
         the agent choosing an action at random.
    """
    def __init__(self, num_bandits, age_bins, eps=0.1):
        self.num_bandits = num_bandits
        self.age_bins = age_bins
        self.eps = eps

        self.records = {}

        self.actions = []  # overall actions, regardless of context
        self.rewards = []

    def _parse_ctx(self, ctx):
        age_bin = np.digitize(ctx[1], self.age_bins)
        return (ctx[0], age_bin)

    def choose_action(self, iter_num, ctx):
        ctx = self._parse_ctx(ctx)
        if ctx not in self.records.keys():
            self.records[ctx] = {
                "n": np.zeros(self.num_bandits, dtype=np.int),  # action counts
                "Q": np.ones(self.num_bandits, dtype=np.float)  # Q-values for each action
            }

        if np.random.rand() < self.eps:
            action = int(np.random.randint(self.num_bandits))
        else:
            action = int(np.argmax(self.records[ctx]))
        self.records[ctx]["n"][action] += 1

        if iter_num == 0:
            totals = np.zeros(self.num_bandits)
            totals[action] = 1
            self.actions.append(totals)
        else:
            totals = self.actions[iter_num-1].copy()
            totals[action] += 1
            self.actions.append(totals)

        return action

    def update_rewards(self, iter_num, ctx, action, reward):
        ctx = self._parse_ctx(ctx)
        self.records[ctx]["Q"][action] += (1.0/self.records[ctx]["n"][action]) * (reward - self.records[ctx]["Q"][action])
        self.rewards.append(reward)

# %%
def context_bandit_gen_context():
    """Randomly selects a gender (0 or 1) and age (between 18 and 65)
    from a uniform distribution.
    """
    gender = np.random.choice([0, 1])
    age = np.random.choice(range(18, 65))
    return np.array([gender, age])

def context_bandit_prob(ctx, action):
    """Given a context (gender, age) and an action (0 or 1), provides the
    probability of receiving a reward. Probability is determined based on
    a logistic function with pre-specified parameters.
    """
    gender = ctx[0]
    age = ctx[1]
    norm_age = (age - 18) / (65 - 18)
    alpha = (-0.3) + (-0.6)*action + (-0.4)*gender + 0.8*gender*action + 0.7*norm_age + 0.5*norm_age*action
    prob = 1 / (1 + np.exp(-alpha))
    return prob

def context_bandit_reward(ctx, num_bandits, action):
    """Given a context (gender, age) and an action (0 or 1), provides a
    reward based on a pre-specified probability distribution.
    """
    all_probs = [context_bandit_prob(ctx, a) for a in range(num_bandits)]
    regret = np.max(all_probs) - all_probs[action]
    if np.random.rand() < all_probs[action]:
        return (1, regret)
    else:
        return (0, regret)

# %%
# contextual bandit problem. Two bandits with probabilities differing by
# gender and age.
num_bandits = 2

rand_agent = RandomAgent(num_bandits)
rand_sim = ContextBanditSim(context_bandit_gen_context,
    context_bandit_reward, num_bandits, rand_agent)

results_rand = rand_sim.run_sim(10000)

print("Mean: {:.2f}, SD: {:.2f}".format(
    np.mean(results_rand["rewards"]), np.std(results_rand["rewards"])))
print("Min: {:.2f}, Max: {:.2f}".format(
    np.min(results_rand["rewards"]), np.max(results_rand["rewards"])))

# %%
q_context_agent = QContextAgent(num_bandits,
    age_bins=np.array([18, 26, 36, 46, 56, 66]), eps=0.05)
q_context_sim = ContextBanditSim(context_bandit_gen_context,
    context_bandit_reward, num_bandits, q_context_agent)

results_qc = q_context_sim.run_sim(50000)

print("Mean: {:.2f}, SD: {:.2f}".format(
    np.mean(results_qc["rewards"]), np.std(results_qc["rewards"])))
print("Min: {:.2f}, Max: {:.2f}".format(
    np.min(results_qc["rewards"]), np.max(results_qc["rewards"])))

# %%
probs_men = [
    [context_bandit_prob(np.array([0, age]), 0) for age in range(18, 66)],
    [context_bandit_prob(np.array([0, age]), 1) for age in range(18, 66)]]
probs_women = [
    [context_bandit_prob(np.array([1, age]), 0) for age in range(18, 66)],
    [context_bandit_prob(np.array([1, age]), 1) for age in range(18, 66)]]

for b in range(num_bandits):
    plt.plot(range(18, 66), probs_men[b], label=f"Arm {b}")
    vals = [q_context_agent.records[(0, x)]["Q"][b] for x in range(1, 6)]
    plt.scatter([21.5, 30.5, 40.5, 50.5, 60.5], vals)
plt.legend()
plt.show()

for b in range(num_bandits):
    plt.plot(range(18, 66), probs_women[b], label=f"Arm {b}")
    vals = [q_context_agent.records[(1, x)]["Q"][b] for x in range(1, 6)]
    plt.scatter([21.5, 30.5, 40.5, 50.5, 60.5], vals)
plt.legend()
plt.show()

# %%
qc_rew_mv_avg = moving_average(q_context_agent.rewards, n=100)
plt.plot(range(len(qc_rew_mv_avg)), qc_rew_mv_avg)
plt.show()

# %%
qc_regret = [q_context_sim.regret(T) for T in range(0, 50000, 100)]
plt.plot(range(0, 50000, 100), qc_regret)
plt.show()
print(f"Final total regret: {qc_regret[-1]}")


# %%
class LinUCBAgent:
    """An agent using the Linear UCB algorithm to weight contextual
    multi-armed bandits.

    This variant of the UCB algorithm for contextual bandits estimates a
    disjoint (i.e., not shared between arms) linear model with ridge
    regression: https://arxiv.org/abs/1003.0146

    Parameters:
    num_bandits: Specifies the number of bandits in the current problem.
    num_dim: Number of dimensions for the context.
    alpha: Tuning parameter on the variance of the estimates.
    """
    def __init__(self, num_bandits, num_dim, alpha):
        self.num_bandits = num_bandits
        self.num_dim = num_dim
        self.alpha = alpha

        self.A = np.array([np.eye(self.num_dim) for _ in range(self.num_bandits)])  # a x dim x dim
        self.b = np.array([np.zeros((self.num_dim, 1)) for _ in range(self.num_bandits)])  # a x dim

        self.actions = []  # overall actions, regardless of context
        self.rewards = []

    def _parse_ctx(self, ctx):
        norm_age = (ctx[1] - 18) / (65 - 18)
        return np.array((ctx[0], norm_age))

    def choose_action(self, iter_num, ctx):
        ctx = self._parse_ctx(ctx)
        p = np.zeros(self.num_bandits)
        for a in range(self.num_bandits):
            Ainv = np.linalg.inv(self.A[a])  # dim x dim
            theta = np.dot(Ainv, self.b[a])  # dim x 1
            p[a] = (np.dot(theta.T, self.b[a]) +
                self.alpha * np.sqrt(np.dot(np.dot(ctx.T, Ainv), ctx)))
                # 1 x 1

        # selects max, except where options are close to being tied, in
        # which case, it selects one of the max values at random
        action = int(np.random.choice(np.flatnonzero(np.isclose(p, p.max()))))
        # action = int(np.argmax(p))

        if iter_num == 0:
            totals = np.zeros(self.num_bandits)
            totals[action] = 1
            self.actions.append(totals)
        else:
            totals = self.actions[iter_num-1].copy()
            totals[action] += 1
            self.actions.append(totals)

        return action

    def update_rewards(self, iter_num, ctx, action, reward):
        ctx = self._parse_ctx(ctx)
        self.A[action] = self.A[action] + np.dot(ctx, ctx.T)  # dim x dim
        self.b[action] = self.b[action] + reward * ctx[:, np.newaxis]  # dim x 1
        self.rewards.append(reward)

# %%
linucb_agent = LinUCBAgent(num_bandits, num_dim=2, alpha=0.1)
linucb_sim = ContextBanditSim(context_bandit_gen_context,
    context_bandit_reward, num_bandits, linucb_agent)

results_ucb = linucb_sim.run_sim(50000)

print("Mean: {:.2f}, SD: {:.2f}".format(
    np.mean(results_ucb["rewards"]), np.std(results_ucb["rewards"])))
print("Min: {:.2f}, Max: {:.2f}".format(
    np.min(results_ucb["rewards"]), np.max(results_ucb["rewards"])))

# %%
linucb_rew_mv_avg = moving_average(linucb_agent.rewards, n=100)
plt.plot(range(len(linucb_rew_mv_avg)), linucb_rew_mv_avg)
plt.show()

# %%
linucb_regret = [linucb_sim.regret(T) for T in range(0, 50000, 100)]
plt.plot(range(0, 50000, 100), linucb_regret)
plt.show()
print(f"Final total regret: {linucb_regret[-1]}")

# %%
# grid search over alpha
linucb_tests = {
    "alpha": [],
    "regret": []
}

for alpha in range(0, 30, 2):
    l_agent = LinUCBAgent(num_bandits, num_dim=2, alpha=alpha/10.0)
    l_sim = ContextBanditSim(context_bandit_gen_context,
        context_bandit_reward, num_bandits, l_agent)
    l_results = l_sim.run_sim(50000, verbose=False)
    linucb_tests["alpha"].append(alpha/10.0)
    linucb_tests["regret"].append(l_sim.regret(50000))

# %%
print(linucb_tests["regret"])
plt.plot(linucb_tests["alpha"], linucb_tests["regret"])
plt.show()

# this seems to indicate that the algorithm either "gets it" or "doesn't",
# and it's not really particularly sensitive to the alpha parameter. Should
# try averaging this over several iterations for each alpha level.