import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive tactics reward focused on tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.performing_tackle_reward = 0.3
        self.avoid_foul_penalty = -0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset for new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State getter for serialization purposes."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State setter for deserialization purposes."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Augment the reward function by adding specific rewards/penalties for defensive behaviors.
        
        Specifically, rewards are given for successfully performing tackle actions and penalties
        for causing fouls.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward for successfully tackling
            tackles = o.get('sticky_actions')[7]  # Assuming 7 is the tackle action index
            if tackles:
                components["tackle_reward"][rew_index] = self.performing_tackle_reward
                reward[rew_index] += self.performing_tackle_reward
            
            # Penalty for committing a foul
            if o.get('game_mode') in (2, 6):  # Assuming these modes are due to fouls
                components["foul_penalty"][rew_index] = self.avoid_foul_penalty
                reward[rew_index] += self.avoid_foul_penalty

        return reward, components

    def step(self, action):
        """Process the action, modify the rewards and step through the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action

        return observation, reward, done, info
