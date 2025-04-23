import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_precision": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if it's a game mode where passing is feasible, not in set piece situations.
            if o['game_mode'] not in [0, 5]:  # Normal play or throw in
                continue
            
            # Assuming ball possession and long pass calculation simplification:
            if o['ball_owned_team'] == 1 or o['ball_owned_team'] == -1:  # right team or no possession
                continue

            ball_position = o['ball']
            ball_direction = o['ball_direction']
            last_ball_position = np.subtract(ball_position, ball_direction)  # Approximation

            # Calculate how far the ball has moved
            ball_travel_distance = np.linalg.norm(last_ball_position[:2] - ball_position[:2])

            # Define a threshold for what constitutes a "long pass"
            long_pass_threshold = 0.5  # 50% of field width
            if ball_travel_distance > long_pass_threshold:  # Simplified, consider plane travel
                # Reward accuracy - checking whether the ball is closer to a teammate
                # Assume simulation provides co-ordinates in normalized range [-1, 1]
                teammates_positions = o['left_team']
                distance_to_teammates = np.linalg.norm(teammates_positions - ball_position[:2], axis=1)
                closest_teammate_distance = np.min(distance_to_teammates)

                accuracy_threshold = 0.1  # distance must be within 10% of field width
                if closest_teammate_distance < accuracy_threshold:
                    components["long_pass_precision"][rew_index] = 0.5  # Reward for accurate long pass
                    reward[rew_index] += components["long_pass_precision"][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
