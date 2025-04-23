import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense rewards for specific offensive skills in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for different actions
        self.pass_coef = 0.2           # reward for successful passes
        self.shot_coef = 1.0           # higher reward for successful shots on target
        self.dribble_coef = 0.1        # reward for successful dribbles
        self.sprint_coef = 0.05        # small reward for effective use of sprint

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            active_player = o['active']

            # Check if any action is taken by the active player
            if o['sticky_actions'][7] == 1 or o['sticky_actions'][1] == 1 or o['sticky_actions'][3] == 1:  # Pass actions
                components["pass_reward"][idx] = self.pass_coef
            if o['sticky_actions'][9] == 1:  # Shot action
                components["shot_reward"][idx] = self.shot_coef
            if o['sticky_actions'][8] == 1:  # Dribble action
                components["dribble_reward"][idx] = self.dribble_coef
            if o['sticky_actions'][4] == 1:  # Sprint action
                components["sprint_reward"][idx] = self.sprint_coef

            reward[idx] += components["pass_reward"][idx] + components["shot_reward"][idx] + components["dribble_reward"][idx] + components["sprint_reward"][idx]

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
