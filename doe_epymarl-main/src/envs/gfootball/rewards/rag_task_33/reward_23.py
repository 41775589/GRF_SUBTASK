import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting efficiently from long distances, particularly outside the penalty box."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_shot_distance_threshold = 0.6  # Threshold to consider shots as long-range
        self.long_shot_reward = 0.5  # Reward for making long-range attempts
        self.goal_reward = 1.0  # Reward for scoring a goal
        
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_shot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, (o, r) in enumerate(zip(observation, reward)):
            if o['game_mode'] == 0:  # Game is in normal play mode
                # Calculate the distance of the ball from opponent's goal
                distance_to_goal = abs(o['ball'][0] - 1)  # Assuming the goal is at x = 1
                if distance_to_goal > self.long_shot_distance_threshold and o['ball_owned_team'] == 1:
                    # Long shot attempted by the right team (assuming agent's team)
                    components["long_shot_reward"][rew_index] = self.long_shot_reward

                # Additional reward if the ball goes into the goal
                if 'goal' in o:  # You might need to change this based on how goal info is stored
                    components["long_shot_reward"][rew_index] += self.goal_reward

            # Summarize and add respective components to the reward
            reward[rew_index] += components["long_shot_reward"][rew_index]

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
