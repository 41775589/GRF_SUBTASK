import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting skill reward focusing on accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_power_threshold = 0.5  # Define a threshold for what we consider a 'powerful' shot
        self.distance_goal_threshold = 0.2  # Distance threshold to the goal to consider it a near goal shot

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0] * len(reward),
            "shooting_power_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            distance_to_goal = np.linalg.norm(np.array(o['ball']) - np.array([1, 0]))
            shot_power = np.linalg.norm(o['ball_direction'][:2])

            if o['game_mode'] == 6:  # Penalty mode
                if shot_power > self.shot_power_threshold and distance_to_goal < self.distance_goal_threshold:
                    components["shooting_power_reward"][rew_index] = 0.5
                    components["shooting_accuracy_reward"][rew_index] = 0.5
                elif shot_power > self.shot_power_threshold:
                    components["shooting_power_reward"][rew_index] = 0.3
                elif distance_to_goal < self.distance_goal_threshold:
                    components["shooting_accuracy_reward"][rew_index] = 0.3

            reward[rew_index] += components["shooting_power_reward"][rew_index] + components["shooting_accuracy_reward"][rew_index]

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
