import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on enhancing offensive skills by promoting 
    Short Pass, Long Pass, Shot, Dribble, and Sprint actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track each of the sticky actions over steps
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset the counter on a new episode
        return self.env.reset()

    def get_state(self, to_pickle):
        # Stores the sticky_actions_counter state
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Loads the sticky_actions_counter state
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Adjustment of the rewards to favor offensive skills."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Check for relevant actions taken: Short Pass (3), Long Pass (4), Shot (5), Dribble (9), Sprint (8)
        # Reward coefficients for each corresponding action
        reward_coefficients = {3: 0.05, 4: 0.05, 5: 0.15, 9: 0.05, 8: 0.02}

        components = {"base_score_reward": reward.copy(),
                      "offensive_action_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            relevant_actions = [3, 4, 5, 9, 8]  # Actions indices for Short Pass, Long Pass, Shot, Dribble, Sprint
            for action_idx in relevant_actions:
                if o['sticky_actions'][action_idx]:
                    components['offensive_action_reward'][rew_index] += reward_coefficients[action_idx]
                    reward[rew_index] += components['offensive_action_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
