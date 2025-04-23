import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically targeted at mastering long passes in football.
       This includes dynamics of the ball travel such as speed and accuracy to a distant player.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_threshold_distance = 0.5  # Assuming a reasonable distance threshold for a "long pass"
        self.long_pass_speed_threshold = 0.03  # Speed threshold to qualify for a long pass
        self.long_pass_accuracy_reward = 1.0  # Reward for accurate long pass
        self.pass_completion_reward = 0.5  # Additional reward if the pass completes successfully
        self.distance_accuracy_factor = 0.5  # Modulates the reward based on the accuracy of the pass

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "accuracy_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            if 'ball_direction' in o and 'ball_owned_team' in o:
                if o['ball_owned_team'] == 0:  # Team 0 has the ball
                    ball_speed = np.linalg.norm(o['ball_direction'])
                    # Check if the pass is long and fast enough
                    if ball_speed > self.long_pass_speed_threshold:
                        distance = np.linalg.norm(o['ball'])
                        if distance > self.long_pass_threshold_distance:
                            components["long_pass_reward"][rew_index] = self.long_pass_accuracy_reward
                            # Assess accuracy based on closeness to a teammate
                            closest_teammate_distance = np.min(np.linalg.norm(o['left_team'] - o['ball'], axis=1))
                            accuracy_bonus = (self.distance_accuracy_factor / (1 + closest_teammate_distance))
                            components["accuracy_bonus"][rew_index] = accuracy_bonus
                            reward[rew_index] += components["long_pass_reward"][rew_index] + components["accuracy_bonus"][rew_index]

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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle
