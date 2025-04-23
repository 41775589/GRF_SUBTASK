import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes and crosses."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_threshold = 0.3  # Distance threshold to consider a high pass
        self.crossing_reward = 0.2    # Reward for completing a cross
        self.high_pass_reward = 0.1   # Reward for a successful high pass
        self.previous_ball_pos = None

    def reset(self):
        super().reset()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_pos'] = self.previous_ball_pos
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.previous_ball_pos = from_pickle['previous_ball_pos']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_pos = np.array(o['ball'][:2])  # Ignore Z for simplicity

            if self.previous_ball_pos is not None:
                distance_travelled = np.linalg.norm(current_ball_pos - self.previous_ball_pos)

                # Check for a high pass
                if distance_travelled > self.passing_threshold and o['ball_direction'][2] > 0:
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    reward[rew_index] += components["high_pass_reward"][rew_index]

                # Check for a cross into the box
                if o['right_team'][2][0] > 0.7:  # assume player 2 is a striker closer to the goal
                    # Check if the ball is near the opponent's box
                    if abs(current_ball_pos[1]) < 0.2 and current_ball_pos[0] > 0.7:
                        components["crossing_reward"][rew_index] = self.crossing_reward
                        reward[rew_index] += components["crossing_reward"][rew_index]

            self.previous_ball_pos = current_ball_pos

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
