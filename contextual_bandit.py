import numpy as np

class ContextBanditSim:
    """Sets up a simulation of a contextual bandit. Must receive a
    function that generates valid contexts, and a function mapping
    contexts and actions to rewards. Must also specify the number of
    bandits (i.e., allowed actions), and the agent to be tested. Running
    the simulation is performed with the run_sim() method.

    Agent provided to the Simulation must implement two methods:
    - choose_action(iter_num, context)
    - update_rewards(iter_num, action, reward)
    """
    def __init__(self, gen_context_fn, reward_fn, num_bandits, agent):
        """Initialize the ContextBanditSim parameters."""
        self.gen_context_fn = gen_context_fn
        self.reward_fn = reward_fn
        self.num_bandits = num_bandits

        self.agent = agent
        self.recorded = {
            "contexts": [],
            "actions": [],
            "rewards": []
        }
 
    def run_iter(self, iter_num):
        """Runs a single iteration, sending the context (obs) to the agent
        to select an action and retrieving the reward.
        """
        ctx = self.gen_context_fn()
        action = self.agent.choose_action(iter_num, ctx)
        if action not in list(range(self.num_bandits)):
            print(f"Error: action {action} greater than number of bandits.")
            return False
        reward = self.reward_fn(ctx, action)
        self.agent.update_rewards(iter_num, action, reward)

        # record results
        self.recorded["contexts"].append(ctx)
        self.recorded["actions"].append(action)
        self.recorded["rewards"].append(reward)
        return True
 
    def run_sim(self, num_iters):
        """Runs the simulation for the specified number of iterations and
        returns the results. Output dict includes "contexts", "actions",
        and "rewards", with an array of length `num_iters` for each."""
        for iter in range(num_iters):
            if (iter+1) % 100 == 0:
                print(f"Iteration: {iter+1}")
            ok = self.run_iter(iter)
            if not ok:
                print("Error occurred. Ending simulation.")
                break
        return self.recorded