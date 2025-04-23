import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a sliding tackle reward focusing on defensive actions near the defensive third.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.4  # Reward increment for successful defensive actions
        self.enemy_attack_zone = -0.5  # Define the defensive third boundary
    
    def reset(self):
        """
        Reset the environment and any internal variables.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Optionally save the internal state for checkpointing.
        """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Optionally restore internal state for checkpointing.
        """
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Compute the reward, enhancing defensive efforts near the goal area.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check if the opponent team is controlling ball in our defensive third.
            if (o['ball_owned_team'] == 1 and o['ball'][0] < self.enemy_attack_zone):
                # Check if any of our players did a sliding action (action 9)
                if o['sticky_actions'][9] == 1:
                    components["tackle_reward"][rew_index] = self.tackle_reward
                    reward[rew_index] += components["tackle_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """
        Take a step using the provided action, recalculating the reward with
        enhanced features based on sliding tackles.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Integrate sticky actions summary into the info for tracking purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
