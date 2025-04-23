import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for accurate long passing.
    This focuses on areas of the field segmented into different zones and rewards accurate passing
    between these zones.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._pass_zones = {
            'defensive': (-1.0, -0.33),
            'midfield': (-0.33, 0.33),
            'offensive': (0.33, 1.0)
        }
        self._last_ball_position = None
        self._last_possessing_zone = None
        self.long_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._last_ball_position = None
        self._last_possessing_zone = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self._last_ball_position
        to_pickle['last_possessing_zone'] = self._last_possessing_zone
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle.get('last_ball_position')
        self._last_possessing_zone = from_pickle.get('last_possessing_zone')
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            ball_pos_x = o['ball'][0]
            current_zone = None
            for zone, bounds in self._pass_zones.items():
                if bounds[0] <= ball_pos_x <= bounds[1]:
                    current_zone = zone
                    break
            
            if current_zone and self._last_possessing_zone and self._last_possessing_zone != current_zone:
                if o['ball_owned_team'] == 0:  # Assuming '0' is the team of interest
                    components["long_pass_reward"][rew_index] = self.long_pass_reward
                    reward[rew_index] += self.long_pass_reward
            
            self._last_ball_position = o['ball'].copy()
            self._last_possessing_zone = current_zone

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
