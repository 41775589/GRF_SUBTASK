import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes accurate long passes across predefined zones of the field.
    It encourages mastering the precision of long passes by rewarding the ball travel over different lengths and accuracy.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints related to passing distance and position accuracy
        self._num_zones = 10
        self._zone_length = 2 / self._num_zones  # Football field in x axis goes from -1 to 1
        self.pass_threshold_distance = 0.3  # Long passes must be at least this fraction of total field length
        self.pass_accuracy_reward = 0.2
        self.prev_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['prev_ball_position'] = self.prev_ball_position
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_ball_position = from_pickle['prev_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_reward = reward.copy()
        components = {"base_score_reward": new_reward.copy(), "pass_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            curr_obs = observation[i]
            if self.prev_ball_position is not None and curr_obs['ball_owned_team'] == 1:
                ball_pos = curr_obs['ball'][:2]
                pass_distance = np.linalg.norm(ball_pos - self.prev_ball_position)
                if pass_distance >= self.pass_threshold_distance:
                    # Check if the pass ends within a designated zone
                    final_x_pos = abs(ball_pos[0])
                    if final_x_pos >= 1 - self._zone_length:  # Pass ended close to goal
                        components['pass_accuracy_reward'][i] = self.pass_accuracy_reward
                        new_reward[i] += components['pass_accuracy_reward'][i]
            self.prev_ball_position = curr_obs['ball'][:2] if curr_obs['ball_owned_team'] != -1 else None

        return new_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
