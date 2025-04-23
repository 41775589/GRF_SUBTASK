import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that encourages executing high passes from midfield,
    optimizing placement and timing for creating direct scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.3  # Reward for successful high passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Verify if the controlled player is in midfield
            midfield_x_coordinate_range = [-0.2, 0.2]
            if (o['active'] is not None and
                midfield_x_coordinate_range[0] <= o['left_team'][o['active']][0] <= midfield_x_coordinate_range[1]):

                # Checking for high pass action being executed
                if o['sticky_actions'][9] == 1:  # Assume index 9 is 'action_high_pass'
                    # Check if the ball is going towards the opponent's box
                    ball_destination_x, ball_destination_y = o['ball'][0] + o['ball_direction'][0], o['ball'][1] + o['ball_direction'][1]
                    opponent_box_x_coordinate_range = [0.6, 1.0]

                    # Check ball landing within opponent's box range in x and within the field in y
                    if (opponent_box_x_coordinate_range[0] <= ball_destination_x <= opponent_box_x_coordinate_range[1] and
                        -0.42 <= ball_destination_y <= 0.42):
                        components['high_pass_reward'][rew_index] = self.high_pass_reward
                        reward[rew_index] += components['high_pass_reward'][rew_index]

        return reward, components

    def step(self, action):
        """
        The overridden step function adds the reward modifications to the info dictionary.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
