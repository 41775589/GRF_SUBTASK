import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for goalkeeper training tasks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions for the goalkeeper

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for checkpointing."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from checkpoint."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the goalkeeper's actions and game state."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_bonus": [0.0] * len(reward),
                      "distribution_bonus": [0.0] * len(reward)}

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_player' not in o:
                continue

            # Reward for successful saves
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == self.env._agent_index:
                # Penalty kick mode and goalkeeper saves the penalty
                components["save_bonus"][rew_index] = 2.0

            # Reward for effective distribution under pressure
            if o['game_mode'] in [3, 4] and o['ball_owned_team'] == 0 and o['sticky_actions'][-1]: # Consider last action as a pass
                components["distribution_bonus"][rew_index] = 1.5

            # Aggregate reward modifications
            reward[rew_index] += components["save_bonus"][rew_index] + components["distribution_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """Step through environment with decorated reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        self.sticky_actions_counter.fill(0)
        # Keep track of sticky actions
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
