import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering long passes and handling ball dynamics over different lengths.
    Specifically, rewards are provided based on the precision and length of the passes under various match conditions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_length_thresholds = [0.3, 0.5, 0.7]  # Thresholds to determine short, medium, and long passes
        self.pass_precision_reward = [0.1, 0.2, 0.3]  # Rewards for passing the ball accurately over different lengths

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            prev_ball_position = o['ball'] if 'last_ball_position' in self.env.unwrapped.info else None
            current_ball_position = o['ball']
            ball_owner = o['ball_owned_player']

            # Only consider scenarios where the ball is owned and transitions happen
            if prev_ball_position is not None and ball_owner == o['active']:
                pass_distance = np.linalg.norm(np.array(prev_ball_position[:2]) - np.array(current_ball_position[:2]))
                
                # Determine the reward based on passing length
                extended_reward = 0
                for i, threshold in enumerate(self.pass_length_thresholds):
                    if pass_distance > threshold:
                        extended_reward = self.pass_precision_reward[i]
                
                components["long_pass_precision_reward"][rew_index] += extended_reward
                reward[rew_index] += extended_reward
        # Update the last ball position in info for the next calculation
        self.env.unwrapped.info['last_ball_position'] = current_ball_position

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
