import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering offensive tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward = 0.5
        self.shot_reward = 2.0
        self.dribble_reward = 0.3

    def reset(self):
        """Reset and return the initial observation from the environment."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of this reward wrapper along with the environment's state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from the given state object."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on additional conditions related to offensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": np.zeros(len(reward)),
            "shot_reward": np.zeros(len(reward)),
            "dribble_reward": np.zeros(len(reward))
        }
        
        if observation is None:
            return reward, components

        for player_index, obs in enumerate(observation):
            # Check if the player performed a passing action
            if 'sticky_actions' in obs and len(obs['sticky_actions']) > 8:
                if obs['sticky_actions'][1] == 1 or obs['sticky_actions'][8] == 1:
                    components["pass_reward"][player_index] = self.pass_reward
                    reward[player_index] += self.pass_reward

            # Check if there was a shot action
            if obs['game_mode'] == 3 and obs['ball_owned_player'] == obs['active']:
                components["shot_reward"][player_index] = self.shot_reward
                reward[player_index] += self.shot_reward

            # Check for dribble actions
            if 'sticky_actions' in obs and len(obs['sticky_actions']) > 4:
                if obs['sticky_actions'][4] == 1:
                    components["dribble_reward"][player_index] = self.dribble_reward
                    reward[player_index] += self.dribble_reward

        return reward, components

    def step(self, action):
        """Step the environment by applying action, modifying the reward, and returning observables."""
        observation, reward, done, info = self.env.step(action)
        # Modify the reward as per the custom reward function
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        # Adding components to info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
