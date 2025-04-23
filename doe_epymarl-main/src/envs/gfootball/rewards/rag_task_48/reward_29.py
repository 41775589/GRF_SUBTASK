import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective high passes from midfield."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward parameters
        self.high_pass_reward = 0.5
        self.scoring_opportunity_reward = 1.0

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
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "scoring_opportunity_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Assume 0 = own goal area, 1 = opponent's goal area (field y-coordinate scale from -1 to 1)
            midfield_zone = (-0.2 < o['left_team'][o['active']][0] < 0.2)
            ongoing_high_pass = (o['ball'][2] > 0.2 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])
            aiming_close_to_goal = (0.9 < o['ball'][0] < 1.0)

            # Reward high passes made from the midfield
            if midfield_zone and ongoing_high_pass:
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Further reward if the high pass translates into a direct scoring chance (i.e., ball near the opposition goal)
            if aiming_close_to_goal and ongoing_high_pass:
                components["scoring_opportunity_reward"][rew_index] = self.scoring_opportunity_reward
                reward[rew_index] += components["scoring_opportunity_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
