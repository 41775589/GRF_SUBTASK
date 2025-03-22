import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering short pass techniques under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_precision_threshold = 0.1  # Threshold distance to consider a pass precise

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_precision_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if the player is performing a passing action and has ownership of the ball
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                ball_position = np.array(o['ball'])
                teammates_positions = o['left_team']

                # Calculate distances between ball and all teammates to find the minimum distance
                distances = [np.linalg.norm(ball_position - teammate) for teammate in teammates_positions]
                min_distance = np.allclose(min(distances), 0, atol=self.pass_precision_threshold)

                # Reward for making a precise pass under pressure (considered if within a threshold distance)
                if min_distance:
                    components["pass_precision_reward"][rew_index] = 1.0  # Reward for a precise pass

        final_reward = [reward[idx] + components["pass_precision_reward"][idx] for idx in range(len(reward))]
        return final_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
