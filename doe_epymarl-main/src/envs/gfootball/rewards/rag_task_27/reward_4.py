import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focusing on defensive responsiveness and interception skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize parameters for calculating defensive rewards
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_distance = None
        self._interception_counter = [0] * self.env.unwrapped.num_agents
    
    def reset(self):
        # Reset interception counters and count of sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_distance = None
        self._interception_counter = [0] * self.env.unwrapped.num_agents
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_distance': self._previous_ball_distance,
            'interception_counter': self._interception_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._previous_ball_distance = from_pickle['CheckpointRewardWrapper']['previous_ball_distance']
        self._interception_counter = from_pickle['CheckpointRewardWrapper']['interception_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for reducing the distance between the active player and the ball when ball is possessed by the opponent
            if o['ball_owned_team'] == 1:  # Assuming agent team is 0
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                current_ball_distance = np.linalg.norm(player_pos - ball_pos)
                
                if self._previous_ball_distance is not None and (current_ball_distance < self._previous_ball_distance[rew_index]):
                    components["defensive_reward"][rew_index] = 0.1  # Reward for closing the distance
                
                self._previous_ball_distance[rew_index] = current_ball_distance
            
            # Reward for intercepting the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if self._interception_counter[rew_index] == 0:
                    components["defensive_reward"][rew_index] = 1.0  # Reward interception first time
                    self._interception_counter[rew_index] = 1  # Mark interception is rewarded
            
            reward[rew_index] += components["defensive_reward"][rew_index]

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
