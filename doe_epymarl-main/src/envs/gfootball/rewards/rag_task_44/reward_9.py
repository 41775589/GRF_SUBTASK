import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on stopping dribble under pressure as a defensive measure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pressure_threshold = 0.3  # Threshold distance to consider under pressure
        self._stop_dribble_reward = 0.5  # Reward for stopping dribble under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['PressureRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['PressureRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Determine if under pressure by the proximity of opponents
            player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            opponents = o['left_team'] if o['ball_owned_team'] == 1 else o['right_team']
            
            min_distance = np.min(np.linalg.norm(opponents - player_pos, axis=1))
            under_pressure = min_distance < self._pressure_threshold

            # Check if player has stopped dribbling under pressure
            if under_pressure and o['sticky_actions'][9] == 0:  # action_dribble == 0
                components["stop_dribble_reward"][rew_index] = self._stop_dribble_reward
                reward[rew_index] += components["stop_dribble_reward"][rew_index]

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
