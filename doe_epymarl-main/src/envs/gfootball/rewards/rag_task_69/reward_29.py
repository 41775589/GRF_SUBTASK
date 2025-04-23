import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that augments the reward based on offensive skills such as shooting accuracy,
    dribbling past opponents, and making effective passes.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_multiplier = 0.2
        self.dribble_reward_multiplier = 0.3
        self.shoot_reward_multiplier = 0.5

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
                      "dribble_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Check for successful passes (i.e., a player switches control within the same team)
            if obs['sticky_actions'][7] or obs['sticky_actions'][1]:  # long pass or high pass attempts
                # Additional reward if team still has ball possession
                if obs['ball_owned_team'] == obs['active']:
                    components["pass_reward"][i] += self.pass_reward_multiplier
                    reward[i] += components["pass_reward"][i]

            # Check for successful dribbling (i.e., maintaining ball possession under pressure)
            if obs['sticky_actions'][9]:  # dribble action
                components["dribble_reward"][i] += self.dribble_reward_multiplier
                reward[i] += components["dribble_reward"][i]

            # Check for shooting attempts (i.e., attempts to score)
            if obs['sticky_actions'][5]:  # shooting attempt
                # More reward for shots that end up closer to the goal, increases if scores
                goal_distance = abs(obs['ball'][0] - 1)  # approximating distance to the opponent's goal in x-direction
                shooting_effectiveness = max(0, 1 - goal_distance)  # closer to goal equals higher effectiveness
                components["shoot_reward"][i] += self.shoot_reward_multiplier * shooting_effectiveness
                reward[i] += components["shoot_reward"][i]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
