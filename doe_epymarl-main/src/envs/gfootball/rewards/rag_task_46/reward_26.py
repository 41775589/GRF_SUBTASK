import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful tackles without fouls during game modes with either normal play or set-pieces."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Not used, this wrapper doesn't maintain state that needs restoring.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        # Iterate over each agent's observations.
        for rew_index, o in enumerate(observation):
            # Game modes where tackles are more important and should be rewarded:
            if o['game_mode'] in (0, 2, 3, 4, 6):  # Normal, GoalKick, FreeKick, Corner, and Penalty
                if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                    # If it's a successful tackle and no foul is committed
                    if not o['left_team_yellow_card'][o['active']]:
                        components["tackle_reward"][rew_index] = 1.0
                        reward[rew_index] += 1.0  # Gives extra reward for a good tackle

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # Summing up to provide a single number for final reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
