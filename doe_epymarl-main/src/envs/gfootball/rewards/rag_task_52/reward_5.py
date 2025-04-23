import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages advanced defending strategies, 
       including tackling proficiency, efficient movement control, 
       and pressured passing in a football game."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define internal state to keep track of rewarded tackles and passes
        self.tackles_rewarded = set()
        self.passes_rewarded = set()

    def reset(self):
        # Reset internal state when the environment is reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_rewarded.clear()
        self.passes_rewarded.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save state information to pickle for reproducibility
        to_pickle['tackles_rewarded'] = self.tackles_rewarded
        to_pickle['passes_rewarded'] = self.passes_rewarded
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set state information from unpickled data
        from_pickle = self.env.set_state(state)
        self.tackles_rewarded = state['tackles_rewarded']
        self.passes_rewarded = state['passes_rewarded']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_idx, agent_reward in enumerate(reward):
            o = observation[agent_idx]
            # Check for successful tackle
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and agent_idx not in self.tackles_rewarded:
                components['tackle_reward'][agent_idx] = 0.2  # tackling reward
                reward[agent_idx] += components['tackle_reward'][agent_idx]
                self.tackles_rewarded.add(agent_idx)

            # Check for successful pressured pass
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['sticky_actions'][1] == 1 and agent_idx not in self.passes_rewarded:
                components['pass_reward'][agent_idx] = 0.1  # pressure passing reward
                reward[agent_idx] += components['pass_reward'][agent_idx]
                self.passes_rewarded.add(agent_idx)

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
