import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for offensive strategies including mastering accurate shooting,
    effective dribbling to evade opponents, and practicing different pass types to break defensive lines.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_threshold = 0.9  # Close to the opponent's goal
        self.dribble_rewards = 0
        self.passing_effectiveness = 0

    def reset(self):
        """
        Reset the reward wrapper's state.
        """
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Serialize the current state of the wrapper.
        """
        to_pickle['dribble_rewards'] = self.dribble_rewards
        to_pickle['passing_effectiveness'] = self.passing_effectiveness
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Deserialize the state of the wrapper.
        """
        from_pickle = self.env.set_state(state)
        self.dribble_rewards = from_pickle['dribble_rewards']
        self.passing_effectiveness = from_pickle['passing_effectiveness']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward function by adding specific rewards for good offensive maneuvers.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for shooting accuracy near opponent's goal
            if o['ball'][0] > self.shooting_threshold:
                components["shooting_reward"][rew_index] = 0.5
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Reward for dribbling based on sticky actions usage (e.g., dribbling near opponents)
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is the dribble action
                self.dribble_rewards += 0.01
                components["dribble_reward"][rew_index] = self.dribble_rewards
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Reward for effective passing (assumption: effective passing increases with action variety and correct positioning)
            pass_effective = np.any([o['sticky_actions'][i] for i in (0, 2, 4, 6)])  # Left, Top, Right, Bottom passes
            if pass_effective and o['ball_owned_team'] == 0:
                self.passing_effectiveness += 0.02
                components["pass_reward"][rew_index] = self.passing_effectiveness
                reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take a step in the environment, modifying the reward.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add detailed reward components to info for debugging purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
