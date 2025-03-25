import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that modifies the reward to emphasize synergy between midfielders in maintaining possession
        and controlling the game pace effectively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "possession_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_central_midfielder = (o['active'] in o['left_team_roles'] and o['left_team_roles'][o['active']] == 5) or (
                                    o['active'] in o['right_team_roles'] and o['right_team_roles'][o['active']] == 5)
            
            # Enhance reward for maintaining possession by the central midfielder
            if o['ball_owned_player'] == o['active'] and is_central_midfielder:
                components["possession_reward"][rew_index] = 0.1
                reward[rew_index] += components["possession_reward"][rew_index]
                
            # Control game pace reward: Reward fewer sticky actions while holding possession
            if o['ball_owned_player'] == o['active'] and is_central_midfielder:
                active_sticky_actions = np.sum(o['sticky_actions'])
                components["possession_reward"][rew_index] -= 0.02 * active_sticky_actions
                reward[rew_index] += components["possession_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
