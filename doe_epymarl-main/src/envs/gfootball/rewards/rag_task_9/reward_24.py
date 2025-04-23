import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward related to offensive skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining specific rewards for different offensive actions
        self.action_rewards = {
            'shot': 0.3,         # Reward for taking a shot
            'pass': 0.1,         # Reward for making a pass
            'dribble_sprint': 0.05  # Reward for dribbling and sprinting effectively
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper_sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Check for corresponding actions and increment the reward accordingly
            if o['sticky_actions'][9]:  # Check dribble
                components["dribble_sprint_reward"][rew_index] = self.action_rewards['dribble_sprint']
            if o['sticky_actions'][8]:  # Check sprint
                components["dribble_sprint_reward"][rew_index] += self.action_rewards['dribble_sprint']
            if o['sticky_actions'][2]:  # Check long pass
                components["pass_reward"][rew_index] = self.action_rewards['pass']
            if o['sticky_actions'][1]:  # Check short pass
                components["pass_reward"][rew_index] = self.action_rewards['pass']

            if o['game_mode'] == 6:  # Check if game mode is in a potential shooting state
                components["shot_reward"][rew_index] = self.action_rewards['shot']

            # Aggregate all rewards
            reward[rew_index] += (components["pass_reward"][rew_index] +
                                  components["shot_reward"][rew_index] +
                                  components["dribble_sprint_reward"][rew_index])
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
