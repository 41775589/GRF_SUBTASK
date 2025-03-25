import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward focusing on midfielder/defender player dynamics."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # Initialize reward component dictionary
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Promote high and long passes actions
            if o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1:  # indexes for high and long passes
                components["pass_reward"][rew_index] = 0.05
                reward[rew_index] += components["pass_reward"][rew_index]

            # Reinforce dribbling when in control of the ball and under pressure
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                close_opponents = np.any(np.linalg.norm(o['right_team'] - o['left_team'][o['active']], axis=-1) < 0.1)
                if close_opponents:
                    components["dribble_reward"][rew_index] = 0.05
                    reward[rew_index] += components["dribble_reward"][rew_index]

            # Sprint and stop actions based on dynamic game context
            if o['sticky_actions'][8] or o['sticky_actions'][9]:  # Sprinting (moving fast)
                components["sprint_reward"][rew_index] = 0.03
            else:  # Stopping or slowing down
                components["sprint_reward"][rew_index] = -0.01

            reward[rew_index] += components["sprint_reward"][rew_index]

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
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_val
        return observation, reward, done, info
