import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function to focus on mastering High Pass 
    and positioning, particularly for wide midfield players, to expand the field of play 
    and support lateral transitions.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        # Initialize the reward components
        components = {"base_score_reward": reward.copy(),
                      "positional_expansion_reward": [0.0] * len(reward),
                      "high_pass_execution_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_position = o['left_team'] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])]

            # Reward wide midfielders for their lateral movement
            if o['left_team_roles'][o['active']] == 6 or o['right_team_roles'][o['active']] == 7:  # Assuming roles 6 and 7 indicate wide midfield roles
                # Encourage using the full width of the pitch
                width_position = abs(player_position[1])  # y-coordinate
                components['positional_expansion_reward'][rew_index] = 0.1 * width_position
            
            # Reward execution of high passes
            if o['game_mode'] == 5 and o['sticky_actions'][9] == 1:  # Assuming index 9 indicates a high pass action
                components['high_pass_execution_reward'][rew_index] = 0.5
            
            # Aggregate the rewards
            reward[rew_index] += (components['positional_expansion_reward'][rew_index] +
                                  components['high_pass_execution_reward'][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
