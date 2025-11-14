import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for strategic positioning and high pass events."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_trigger = False
        self.strategy_reward_coefficient = 0.5
        self.high_pass_reward_coefficient = 0.8

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_trigger = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['high_pass_trigger'] = self.high_pass_trigger
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.high_pass_trigger = from_pickle['high_pass_trigger']
        return from_pickle

    def reward(self, reward):
        """Modify rewards based on observable strategic movements and effective high pass actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "strategic_positioning_reward": [0] * len(reward),
            "high_pass_reward": [0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward strategic positioning close to key game zones
            distance_to_opp_goal = abs(o['ball'][0] - 1) # Assuming opponent goal is at x=1
            if o['ball_owned_team'] == 1 and distance_to_opp_goal < 0.2:
                components["strategic_positioning_reward"][rew_index] = self.strategy_reward_coefficient
                reward[rew_index] += components["strategic_positioning_reward"][rew_index]

            # High pass specific reward
            if 'ball_direction' in o and o['ball_direction'][2] > 0.15: # Assuming 0.15 is threshold for high pass
                if self.high_pass_trigger == False:
                    self.high_pass_trigger = True
                    components["high_pass_reward"][rew_index] = self.high_pass_reward_coefficient
                    reward[rew_index] += components["high_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Process environment step with custom rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
