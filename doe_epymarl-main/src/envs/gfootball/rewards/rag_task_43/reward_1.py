import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds sophisticated rewards based on defensive performance and transition to counterattacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = self.initialize_defensive_zones()
        self.defensive_rewards_given = set()
        self.transition_rewards_given = set()
        self.defensive_reward = 0.2
        self.transition_reward = 0.3

    def initialize_defensive_zones(self):
        # Define zones based on typical defensive positions in football
        return {
            'goalkeeper_zone': [-1, -0.1, 0.1],  # x_position, min_y, max_y
            'defense_zone': [-0.7, -0.42, 0.42]
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards_given = set()
        self.transition_rewards_given = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_rewards_given': self.defensive_rewards_given, 
            'transition_rewards_given': self.transition_rewards_given
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards_given = from_pickle['CheckpointRewardWrapper']['defensive_rewards_given']
        self.transition_rewards_given = from_pickle['CheckpointRewardWrapper']['transition_rewards_given']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for being in the right defensive position
            if o['ball_owned_team'] == 0: # if left team owns the ball
                player_x, player_y = o['left_team'][o['active']]
            else:
                player_x, player_y = o['right_team'][o['active']]

            # Check defensive positioning
            if (player_x <= self.defensive_positions['defense_zone'][0] 
                and self.defensive_positions['defense_zone'][1] <= player_y <= self.defensive_positions['defense_zone'][2]
                and rew_index not in self.defensive_rewards_given):
                components["defensive_reward"][rew_index] = self.defensive_reward
                reward[rew_index] += components["defensive_reward"][rew_index]
                self.defensive_rewards_given.add(rew_index)

            # Reward for successful transition
            if (o['ball_owned_team'] == 1 and  # if right team owns the ball
                player_x >= -0.5 and  # crossing the midfield line
                rew_index not in self.transition_rewards_given
                ):
                components["transition_reward"][rew_index] = self.transition_reward
                reward[rew_index] += components["transition_reward"][rew_index]
                self.transition_rewards_given.add(rew_index)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    if action:
                        self.sticky_actions_counter[i] += 1
            else:  # If no sticky_actions, assume no specific ones are used
                pass
        return observation, reward, done, info
