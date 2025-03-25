import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on training agents in defensive scenarios near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = -1
        self.defensive_checkpoint_reward = 0.2
        self.penalty_area_coords = {'xmin': -1.0, 'xmax': 1.0, 'ymin': -0.22, 'ymax': 0.22}
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['CheckpointRewardWrapper_previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state['CheckpointRewardWrapper_sticky_actions_counter']
        self.previous_ball_owner = state['CheckpointRewardWrapper_previous_ball_owner']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy()}
        defensive_reward = np.zeros(len(reward))
        
        for idx in range(len(reward)):
            o = observation[idx]
            ball_x, ball_y = o['ball'][0], o['ball'][1]

            if self.previous_ball_owner != o['ball_owned_team'] and o['ball_owned_team'] in [0, 1]:
                # Ball ownership change happened
                if self.penalty_area_coords['xmin'] <= ball_x <= self.penalty_area_coords['xmax'] and self.penalty_area_coords['ymin'] <= ball_y <= self.penalty_area_coords['ymax']:
                    if o['ball_owned_team'] == 0:  # If left team owns the ball in their own penalty area
                        defensive_reward[idx] = self.defensive_checkpoint_reward

            # Update the previous ball owner for next step
            self.previous_ball_owner = o['ball_owned_team']

        # Update the reward with the additional defensive reward
        reward += defensive_reward
        components['defensive_rewards'] = defensive_reward.tolist()  # convert array to list for consistency in output
        
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
