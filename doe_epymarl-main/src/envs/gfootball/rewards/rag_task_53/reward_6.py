import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on strategic ball control and distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_duration = 0
        self.prev_ball_position = None
        self.control_reward_threshold = 5  # threshold in steps to reward for continuous control

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_duration = 0
        self.prev_ball_position = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "strategic_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for maintaining ball control
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Reward increases after each step of controlling the ball
                self.ball_control_duration += 1
                if self.ball_control_duration > self.control_reward_threshold:
                    components["strategic_control_reward"][rew_index] = 0.1
                    reward[rew_index] += components["strategic_control_reward"][rew_index]

                # Reward for not just holding onto the ball but moving it
                current_ball_position = o['ball'][:2]  # Get X, Y coordinates
                if self.prev_ball_position is not None:
                    distance = np.linalg.norm(current_ball_position - self.prev_ball_position)
                    if distance > 0.01:  # moved significantly
                        components["strategic_control_reward"][rew_index] += 0.05
                        reward[rew_index] += components["strategic_control_reward"][rew_index]

                self.prev_ball_position = current_ball_position
            else:
                self.ball_control_duration = 0
                self.prev_ball_position = None

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
