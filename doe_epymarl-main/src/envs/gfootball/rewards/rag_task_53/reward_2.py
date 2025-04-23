import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for maintaining ball control and strategic ball distribution.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_duration = {}
        self.zone_control_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_duration = {}
        self.zone_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_duration'] = self.ball_control_duration
        to_pickle['zone_control_rewards'] = self.zone_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_duration = from_pickle.get('ball_control_duration', {})
        self.zone_control_rewards = from_pickle.get('zone_control_rewards', {})
        return from_pickle

    def reward(self, reward):
        # Fetch the current observations from the environment
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'control_and_distribution_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Process each agent's observation to determine rewards
        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Reward for maintaining control
                duration = self.ball_control_duration.get(idx, 0) + 1
                self.ball_control_duration[idx] = duration
                components['control_and_distribution_reward'][idx] += 0.01 * duration

                # Reward for ball distribution by checking positions
                player_x, player_y = o['right_team'][o['active']]
                zone = self.determine_zone(player_x, player_y)
                if self.zone_control_rewards.get(idx) != zone:
                    self.zone_control_rewards[idx] = zone
                    components['control_and_distribution_reward'][idx] += 0.05

            reward[idx] += components['control_and_distribution_reward'][idx]

        return reward, components

    def step(self, action):
        # Execute a step using the wrapped environment
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        return observation, reward, done, info

    def determine_zone(self, x, y):
        """Determine the control zone based on x, y coordinates on the field."""
        if x > 0.5:
            return 'attacking_third'
        elif x > -0.5:
            return 'midfield_third'
        else:
            return 'defensive_third'
