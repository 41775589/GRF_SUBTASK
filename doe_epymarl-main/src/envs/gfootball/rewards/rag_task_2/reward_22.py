import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive play and ball control."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.defensive_zones = 10  # Divide the field into 10 defensive zones
        self.zone_reward = 0.05  # Reward for holding zone defensively
        self.control_reward = 0.1  # Reward for maintaining control of the ball
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._zone_control = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._zone_control = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._zone_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._zone_control = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "control_reward": [0.0] * len(reward), 
                      "defensive_zone_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for control (team owning the ball)
            if o['ball_owned_team'] == 0:  # Assuming 0 is the team of the agent
                components['control_reward'][rew_index] = self.control_reward
                reward[rew_index] += components['control_reward'][rew_index]

            # Reward based on defensive positioning
            player_x = o['left_team'][o['active']][0]
            zone_index = min(int((player_x + 1) * self.defensive_zones / 2), self.defensive_zones - 1)
            if zone_index not in self._zone_control:
                self._zone_control[zone_index] = True
                components['defensive_zone_reward'][rew_index] = self.zone_reward
                reward[rew_index] += components['defensive_zone_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        # Slow the action on each discrete step
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
