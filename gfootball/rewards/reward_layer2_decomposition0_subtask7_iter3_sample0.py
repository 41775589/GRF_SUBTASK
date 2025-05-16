import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards focused on 'Stop-Moving' strategies, aiding in the interception of passes and
    maintenance of strategic positions by reacting dynamically to the movement of the ball and observing player's
    proximity to opponent.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward when the controlled player achieves a 'Stop-Moving' action near the opponent with the ball 
        self.stop_near_opponent_reward = 0.2  
        
        # Reward when the controlled player effectively stops moving without any nearby opponent.
        self.stop_without_opponent_reward = 0.05  
        
        # This threshold defines 'closeness' to an opponent for additional reward
        self.proximity_threshold = 0.1  

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_near_opponent_reward": [0.0],
            "stop_without_opponent_reward": [0.0]
        }

        if observation is None:
            return reward, components

        o = observation[0]
        own_position = o['left_team'][o['active']]

        # Identifying whether any opponent is close
        opponent_positions = o['right_team']
        distances = np.linalg.norm(opponent_positions - own_position, axis=1)
        close_opponents = np.any(distances < self.proximity_threshold)

        # Check the stopping condition based on sticky actions ('action_left' and 'action_right')
        if o['sticky_actions'][0] == 1 and o['sticky_actions'][4] == 0:
            if close_opponents:
                # Reward more if stopping near an opponent
                components["stop_near_opponent_reward"][0] = self.stop_near_opponent_reward
            else:
                # Reward for stopping strategically with no opponents in close proximity
                components["stop_without_opponent_reward"][0] = self.stop_without_opponent_reward

        # Update the total reward for the first agent
        reward[0] += components["stop_near_opponent_reward"][0] + components["stop_without_opponent_reward"][0]

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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
