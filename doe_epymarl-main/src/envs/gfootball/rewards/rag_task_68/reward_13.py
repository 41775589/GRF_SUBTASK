import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focusing on offensive strategies: 
    shooting accuracy, dribbling, and making effective passes."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.2
        self.passing_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        """Reset the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        """Serialize the wrapper state."""
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the wrapper state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper'].get("sticky_actions_counter", np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Rewards for offensive strategies."""
        observation = self.env.unwrapped.observation()
        modified_rewards = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return modified_rewards, components

        for i in range(len(reward)):
            # Extract current observation details for each agent
            o = observation[i]
            
            # Dribbling reward: check sticky actions related to dribbling
            if o['sticky_actions'][9]:  # action_dribble
                components["dribbling_reward"][i] = self.dribbling_reward
                modified_rewards[i] += components["dribbling_reward"][i]
            
            # Passing reward: Consider both long and high passes based on game situation
            if o['game_mode'] in [3, 4, 5]:  # FreeKick, Corner, ThrowIn - opportunities for strategic passes
                components["passing_reward"][i] = self.passing_reward
                modified_rewards[i] += components["passing_reward"][i]

            # Shooting reward: Goal scoring or attempts close to goal
            if o['ball'][0] > 0.8 and o['ball_owned_team'] == 1:  # Near opponent's goal and ball owned by own team
                components["shooting_reward"][i] = self.shooting_reward
                modified_rewards[i] += components["shooting_reward"][i]

        return modified_rewards, components

    def step(self, action):
        """Steps through the environment, applying the reward wrapper."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
