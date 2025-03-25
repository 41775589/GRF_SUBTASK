import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that improves the skills of a 'stopper' role by adding a specific reward for man-marking, blocking shots, and stalling forward moves."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            components.setdefault(f"defensive_rewards_{rew_index}", 0.0)

            # Reward for being close to the ball if it's controlled by an opposing team
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                reward[rew_index] += 0.2
                components[f"defensive_rewards_{rew_index}"] += 0.2

            # Reward for interacting with the ball when an opposing player is close
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball'] - o['right_team'][o['right_team_active']]) < 0.1:
                reward[rew_index] += 0.3
                components[f"defensive_rewards_{rew_index}"] += 0.3

            # Extra reward for blocking shots
            if o['game_mode'] == 2 and o['ball_owned_player'] != o['active']:
                reward[rew_index] += 0.5
                components[f"defensive_rewards_{rew_index}"] += 0.5

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
