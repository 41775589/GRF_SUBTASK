import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on advanced dribbling techniques and sprint usage.
    This is designed to encourag...
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = [0.0, 0.0, 0.0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = [0.0, 0.0, 0.0]
        return super().reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return super().get_state(to_pickle)

    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'], dtype=int)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_usage_reward": [0.0] * len(reward),
                      "dribbling_advance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for using sprint effectively in offensive positions
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # action_sprint index is 8
                if o['ball_owned_team'] == 1 and o['right_team'][o['active']][0] > 0.5:  # Offensive half
                    components["sprint_usage_reward"][rew_index] = 0.02

            # Reward for progressing forward with the ball
            current_ball_position = o['ball'][0]
            ball_progress = current_ball_position - self.previous_ball_position[0]
            if o['ball_owned_team'] == 1 and current_ball_position > 0.5:
                components["dribbling_advance_reward"][rew_index] = max(0.0, ball_progress)

            # Combine all rewards
            reward[rew_index] += components["sprint_usage_reward"][rew_index] + components["dribbling_advance_reward"][rew_index]

        self.previous_ball_position = observation[0]['ball'] # assuming single agent centric observation
        return reward, components

    def step(self, action):
        observation, reward, done, info = super().step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
