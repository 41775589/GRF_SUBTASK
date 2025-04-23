import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on the Stop-Dribble action under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._num_actions = 10  # Number of sticky actions (sprint, dribble, etc.)
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)
        self.stop_dribble_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(self._num_actions, dtype=int)
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
                      "stop_dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if 'active' in o and 'sticky_actions' in o:
                    # Check if dribble action is taken and then stopped
                    dribble_index = 9  # Assuming dribble is at the 9th index
                    if o['sticky_actions'][dribble_index] == 1:
                        self.sticky_actions_counter[dribble_index] += 1
                    else:
                        if self.sticky_actions_counter[dribble_index] > 0:
                            # Reward for stopping the dribble
                            components["stop_dribble_reward"][rew_index] = self.stop_dribble_reward
                            reward[rew_index] += components["stop_dribble_reward"][rew_index]
                            self.sticky_actions_counter[dribble_index] = 0
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include reward components in the info dictionary for detailed traceability
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions seen so far
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
