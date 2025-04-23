import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for successfully executing attacking plays,
       focusing on finishing and creative offensive play under match-like pressures."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._offensive_play_points = {}
        self._num_segments = 5
        self._segment_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._offensive_play_points = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['offensive_play_points'] = self._offensive_play_points
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._offensive_play_points = from_pickle['offensive_play_points']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["offensive_play_reward"][rew_index] = 0.0

            if o['ball_owned_team'] != 1 or o['ball_owned_player'] != o['active']:
                continue
            
            # Measure player's position and player's distance to the goal
            player_pos = o['right_team'][o['active']]
            goal_pos = [1, 0]  # The opponent’s goal location
            dist_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))

            if dist_to_goal < 0.3:  # If a player is quite close to the goal
                segment_index = int((0.3 - dist_to_goal) / 0.06)
                if segment_index > self._offensive_play_points.get(rew_index, -1):
                    components["offensive_play_reward"][rew_index] = self._segment_reward
                    self._offensive_play_points[rew_index] = segment_index
                    reward[rew_index] += self._segment_reward

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
