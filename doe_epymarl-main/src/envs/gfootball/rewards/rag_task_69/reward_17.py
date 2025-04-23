import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes offensive play including accurate shooting, dribbling, and making strategic passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_factor = 0.2
        self.shoot_reward_factor = 0.3
        self.dribble_reward_factor = 0.1

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
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for high or long passes which might break opponent lines
            if o['game_mode'] in {3, 6} and o['sticky_actions'][1] == 1:
                components["pass_reward"][rew_index] = self.pass_reward_factor
                reward[rew_index] += components["pass_reward"][rew_index]

            # Reward for shots towards the goal
            if o['ball_owned_player'] == o['active'] and o['game_mode'] == 0 and o['ball'][0] > 0.5:
                components["shoot_reward"][rew_index] = self.shoot_reward_factor
                reward[rew_index] += components["shoot_reward"][rew_index]

            # Reward for dribbling effectively
            if o['sticky_actions'][9] == 1 and o['right_team_active'][o['active']]:
                components["dribble_reward"][rew_index] = self.dribble_reward_factor
                reward[rew_index] += components["dribble_reward"][rew_index]

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
