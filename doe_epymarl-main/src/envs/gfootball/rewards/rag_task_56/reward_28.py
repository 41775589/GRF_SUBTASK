import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards to encourage specific defensive behaviors for football players."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define the number of defensive zones and initial reward for positioning and shot-stopping
        self.defensive_zones = 5
        self.positioning_reward = 0.05
        self.shot_stopping_reward = 0.2

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
        components = {'base_score_reward': reward.copy(),
                      'positional_reward': [0.0] * len(reward),
                      'shot_stopping_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]
            
            # Encourage the goalkeeper to be close to the goal line for better shot-stopping
            if o['right_team_roles'][o['active']] == 0:  # 0 is typically the goalkeeper
                goalie_pos = o['right_team'][o['active']]
                if goalie_pos[0] > 0.7:  # Closer to the goal line on the x-axis
                    components['shot_stopping_reward'][rew_index] = self.shot_stopping_reward
                    reward[rew_index] += components['shot_stopping_reward'][rew_index]

            # Reward defenders for being in strategic defensive positions
            for player_index, role in enumerate(o['right_team_roles']):
                if role in [1, 2, 3]:  # Typically defenders
                    player_pos = o['right_team'][player_index]
                    if player_pos[0] > 0.5 and abs(player_pos[1]) < 0.3:  # Strategic defensive zones
                        components['positional_reward'][rew_index] += self.positioning_reward
                        reward[rew_index] += components['positional_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_value
        
        return observation, reward, done, info
