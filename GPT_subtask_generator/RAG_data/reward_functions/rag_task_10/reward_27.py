import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive gameplay nuances such as tackling,
    blocking dribbles, and effective positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
        components = {"base_score_reward": reward.copy(),
                      "defensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            player_position = o['left_team'][o['active']] if o['active'] != -1 else None
            
            # Additional rewards based on defensive actions
            if player_position and o['ball_owned_team'] == 1:
                # Calculate distance from ball to simulate tackling effort
                distance_to_ball = np.linalg.norm(ball_position - player_position[:2])
                if distance_to_ball < 0.05:
                    components["defensive_play_reward"][rew_index] = 0.1
                    
                # Check player's action related to defense
                if o['sticky_actions'][6] > 0 or o['sticky_actions'][7] > 0:  # Sliding or body collision
                    components["defensive_play_reward"][rew_index] += 0.1

            # Compound both rewards
            reward[rew_index] += components["defensive_play_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                if action_value:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action_value

        return observation, reward, done, info
