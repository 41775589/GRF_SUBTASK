import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a transition-focused reward for mastering Sprint, Stop-Sprint, and Dribble actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize count for sticky actions

    def reset(self):
        """Reset the sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()  # Reset the environment

    def get_state(self, to_pickle):
        """Store the current state of the sticky action counters."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the sticky action counters from the saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward by adding a bonus for using Sprint, Stop-Sprint, and Dribble actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "transition_reward": 0.0}

        if observation is None or len(observation) == 0:
            return reward, components

        # Assume there's a single agent
        o = observation[0]
        
        # Check for sprint-related actions
        if 'sticky_actions' in o:
            sprint_index = 8  # Based on ordering provided in stick actions
            dribble_index = 9
            sprint_active = o['sticky_actions'][sprint_index]
            dribble_active = o['sticky_actions'][dribble_index]

            # Encourage usage of Sprint, Stop-Sprint, and Dribble
            if sprint_active:
                components['transition_reward'] += 0.1  # Bonus for sprinting
            if dribble_active:
                components['transition_reward'] += 0.05  # Bonus for dribbling

        reward[0] += components['transition_reward']
        
        return reward, components

    def step(self, action):
        """Process the environment step and add additional reward components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        return observation, reward, done, info
