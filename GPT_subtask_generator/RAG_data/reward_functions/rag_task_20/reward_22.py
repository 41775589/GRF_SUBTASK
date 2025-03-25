import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward in terms of offensive play. 
    It rewards agents for advancing towards the opponent's goal, maintaining possession,
    and successful passing strategies, encouraging teamwork and dynamic positioning.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_checkpoints = 5  # Divide field in 5 regions approaching the opponent's goal
        self._checkpoint_reward = 0.2
        self._passing_reward = 0.1
        self._possession_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {
            "base_score_reward": reward.copy(),
            "advance_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # x-coordinate of the ball's position

            # Encourage advancing towards the opponent's goal and controlling the ball
            if o['ball_owned_team'] == 1 and ball_position > 0:  # Assuming right side is the aiming goal
                region_index = min(int((ball_position + 1) / 0.4), self._num_checkpoints - 1)
                components["advance_reward"][rew_index] = (self._num_checkpoints - region_index) * self._checkpoint_reward

            # Reward for maintaining possession
            if o['ball_owned_team'] == 1:
                components["possession_reward"][rew_index] = self._possession_reward

            # Observing successful passing
            if 'action' in o and o['action'] in [football_action_set.action_short_pass, football_action_set.action_long_pass] and o['ball_owned_team'] == 1:
                components["passing_reward"][rew_index] += self._passing_reward
            
            reward[rew_index] += components["advance_reward"][rew_index] + components["passing_reward"][rew_index] + components["possession_reward"][rew_index]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
