import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long pass accuracy."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the long pass range as distance between passer and receiver.
        self.long_pass_minimum_distance = 0.5
        self.long_pass_accuracy_reward = 0.5

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
        # Getting underlying environment observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                # Assuming player indices match their array positions
                ball_owner_position = np.array(o['left_team'][o['active']])
                # Iterate through all players to check for potential receivers
                for idx, teammate_pos in enumerate(o['left_team']):
                    if idx != o['active']:
                        teammate_position = np.array(teammate_pos)
                        distance = np.linalg.norm(teammate_position - ball_owner_position)
                        if distance >= self.long_pass_minimum_distance:
                            # Reward for executing a successful long pass
                            components["long_pass_accuracy_reward"][rew_index] = self.long_pass_accuracy_reward
                            break

        # Update rewards
        reward = [base + comp for base, comp in zip(components["base_score_reward"], components["long_pass_accuracy_reward"])]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update the sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
