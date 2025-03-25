import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function to encourage correct defensive maneuvers focusing on sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackles_counter = np.array([0.0] * 2)
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackles_counter = np.array([0.0] * 2)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sliding_tackles': self.sliding_tackles_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackles_counter = np.array(from_pickle['CheckpointRewardWrapper']['sliding_tackles'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward[:],
                      "sliding_tackle_reward": [0.0] * len(reward)}

        for idx in range(len(reward)):
            # Registers events related to successful sliding tackles
            tackle_detected = observation[idx]['sticky_actions'][6]  # assume index 6 is the sliding tackle action
            ball_position = observation[idx]['ball'].copy()

            if self.previous_ball_position is not None:
                # Using decreasing distance to our own goal and successful sliding as a triggering mechanism
                previous_distance_to_goal = np.linalg.norm(self.previous_ball_position[:2] + [1, 0])  # assuming goal at x = -1
                current_distance_to_goal = np.linalg.norm(ball_position[:2] + [1, 0])
                
                if tackle_detected and previous_distance_to_goal > current_distance_to_goal:
                    components["sliding_tackle_reward"][idx] = 1.0  # simple reward for moving ball closer through sliding
                    self.sliding_tackles_counter[idx] += 1
            self.previous_ball_position = ball_position

            reward[idx] += components["sliding_tackle_reward"][idx]
        
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
