import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting skill enhancement reward for central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.power_shooting_zone = {
            'min_x': -0.25,  # central zone limits, expressed in normalized coordinates
            'max_x': 0.25,
            'min_y': -0.42,  # including the whole width
            'max_y': 0.42
        }
        self.shooting_skill_multiplier = 5.0  # Multiplier for shooting from the central field

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
        components = {'base_score_reward': reward.copy(), 'accuracy_enhancement': [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            ball_pos_x, ball_pos_y = o['ball'][0], o['ball'][1]
            # Check if the action is a shot and if it's within the designated central field zone
            if self.env.action_space.contains('shot') and \
               self.power_shooting_zone['min_x'] <= ball_pos_x <= self.power_shooting_zone['max_x'] and \
               self.power_shooting_zone['min_y'] <= ball_pos_y <= self.power_shooting_zone['max_y']:
                # Calculate additional reward by considering the distance from central line
                distance_from_center = np.abs(ball_pos_x)  # Central line is at x=0
                # Closer to center gives higher reward
                components['accuracy_enhancement'][idx] = self.shooting_skill_multiplier * (1 - distance_from_center)
                reward[idx] += components['accuracy_enhancement'][idx]

        return reward.copy(), components.copy()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
