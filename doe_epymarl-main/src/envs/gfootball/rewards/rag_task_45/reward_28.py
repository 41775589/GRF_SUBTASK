import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering abrupt stopping 
    and direction changes defensively (Stop-Sprint and Stop-Moving)."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "stop_sprint_reward": [0.0] * 2}

        for i, o in enumerate(observation):
            # Detect abrupt stops and sprints
            if o['sticky_actions'][8] == 1 and np.linalg.norm(o['right_team_direction'][o['active']]) < 0.01:
                # Reward for stopping suddenly while sprinting
                components["stop_sprint_reward"][i] = 0.2
            if o['sticky_actions'][:8].sum() == 0 and np.linalg.norm(o['right_team_direction'][o['active']]) < 0.01:
                # Reward for stopping all movements abruptly
                components["stop_sprint_reward"][i] += 0.1

            # Update total reward for each agent
            reward[i] += components["stop_sprint_reward"][i]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
