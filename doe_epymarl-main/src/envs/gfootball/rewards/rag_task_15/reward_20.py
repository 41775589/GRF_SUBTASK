import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long passes in the football environment."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Assume the field ranges from -1 to 1 in x-coordinate, setting distinct regions for passing
        self.long_pass_threshold = 0.3  # Threshold for x-distance to be considered a long pass
        self.long_pass_reward = 1.0  # Reward for successful long pass
        self.previous_ball_position = None
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle.get('CheckpointRewardWrapper_previous_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            current_ball_position = o['ball'][:2]  # using x, y position
            if self.previous_ball_position is not None:
                dx = abs(current_ball_position[0] - self.previous_ball_position[0])

                if dx >= self.long_pass_threshold and o['ball_owned_team'] == o['active']:
                    # Check if the ball traveled beyond the long pass threshold and was in possession of the team
                    components["long_pass_reward"][rew_index] = self.long_pass_reward
                    reward[rew_index] += components["long_pass_reward"][rew_index]

            self.previous_ball_position = current_ball_position[:]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
