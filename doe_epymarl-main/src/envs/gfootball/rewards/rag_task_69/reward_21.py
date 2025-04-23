import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards for offensive actions including shooting, dribbling, and passing effectively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define coefficients for each type of action to tweak the reward sensitivity
        self.shoot_coefficient = 3.0
        self.dribble_coefficient = 1.0
        self.passing_coefficient = 2.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            if 'sticky_actions' in o:
                if o['sticky_actions'][9] > 0:  # Dribble action is active
                    components["dribble_reward"][rew_index] = self.dribble_coefficient
                    reward[rew_index] += components["dribble_reward"][rew_index]

                if o['game_mode'] in [3, 5]:  # FreeKick or ThrowIn - high chances of a pass
                    components["passing_reward"][rew_index] = self.passing_coefficient
                    reward[rew_index] += components["passing_reward"][rew_index]

            # Reward shots towards the goal
            if 'right_team_roles' in o:
                goal_target = abs(o['right_team'][o['active']][1])  # y-position close to 0
                if o['ball_direction'][0] > 0.5 and goal_target < 0.2:  # ball is moving forward fast, towards center goal
                    components["shoot_reward"][rew_index] = self.shoot_coefficient
                    reward[rew_index] += components["shoot_reward"][rew_index]

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
