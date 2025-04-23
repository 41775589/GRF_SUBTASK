import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for enhancing shot precision skills in tight spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.reward_thresholds = [0.2, 0.1]  # Thresholds in proximity to goal to check
        self.shot_power_reward = 0.1
        self.shot_precision_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.has_shot = False  # To control the reward given for attempted shots

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.has_shot = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'has_shot': self.has_shot
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.has_shot = from_pickle['CheckpointRewardWrapper']['has_shot']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Components dictionary to track different parts of the reward
        components = {"base_score_reward": reward.copy(), "shot_power_reward": 0.0, "shot_precision_reward": 0.0}

        if observation is None:
            return reward, components

        ball_pos = observation[0]['ball'][:2]  # Assume 2D position
        goal_pos = [1, 0]  # Assuming right goal position
        dist_to_goal = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))

        # Check the shot power and precision based on the ball's proximity to goal and player's current action.
        if not self.has_shot and observation[0]['active'] != -1:
            if dist_to_goal <= self.reward_thresholds[0] and 'action_shot' in observation[0]['sticky_actions']:
                components["shot_power_reward"] = self.shot_power_reward
                reward[0] += 1.5 * components["shot_power_reward"]
                self.has_shot = True

            if dist_to_goal <= self.reward_thresholds[1] and 'action_dribble' in observation[0]['sticky_actions']:
                components["shot_precision_reward"] = self.shot_precision_reward
                reward[0] += components["shot_precision_reward"]
                self.has_shot = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
