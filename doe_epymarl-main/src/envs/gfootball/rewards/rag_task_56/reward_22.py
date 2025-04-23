import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances goalkeeper and defender training by rewarding effective defense actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.gk_ball_stops = 0
        self.defensive_actions = 0
        self._gk_reward = 1.0
        self._defender_reward = 0.5

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.gk_ball_stops = 0
        self.defensive_actions = 0
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['gk_ball_stops'] = self.gk_ball_stops
        to_pickle['defensive_actions'] = self.defensive_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.gk_ball_stops = from_pickle['gk_ball_stops']
        self.defensive_actions = from_pickle['defensive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "gk_bonus": 0, "defense_bonus": 0}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            role = o['right_team_roles'] if o['active'] in o['right_team'] else o['left_team_roles']
            if role[o['active']] == 0:  # Goalkeeper role index
                if o['ball_owned_team'] == o['active'] and o['ball_owned_player'] == o['active']:
                    # Ball stopped by goalkeeper
                    self.gk_ball_stops += 1
                    components['gk_bonus'] = self._gk_reward
                    reward[rew_index] += components['gk_bonus']
                
            elif role[o['active']] in [1, 2, 3, 4]:  # Defender indices (CB, LB, RB, DM)
                if o['ball_owned_team'] == o['active'] and o['ball_owned_player'] == o['active']:
                    # Successful tackle or interception by a defender
                    self.defensive_actions += 1
                    components['defense_bonus'] = self._defender_reward
                    reward[rew_index] += components['defense_bonus']

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
