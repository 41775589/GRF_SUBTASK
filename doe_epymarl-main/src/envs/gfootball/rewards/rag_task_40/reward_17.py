import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focused on defensive strategies and counterattack positioning.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_zones = 6
        self.defensive_reward = 0.1

    def reset(self):
        """
        Reset the internal state upon starting a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the wrapper along with the environment's state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the wrapper from the saved state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Calculate enhanced rewards for defensive actions and strategic positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'defensive_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]
            if obs is None:
                continue
            
            # Reward for successful tackles
            if obs.get('game_mode') == 4:  # Assuming '4' might represent a defensive win like tackling
                components['defensive_reward'][idx] = self.defensive_reward
                reward[idx] += components['defensive_reward'][idx]

            # Evaluate defensive positioning: closer to their own goal increases the reward
            player_x = obs['left_team'][obs['active']][0]  # Assuming the active index points to current player coordinates
            if player_x < -0.5:  # Behind the midfield line
                components['defensive_reward'][idx] += self.defensive_reward * (0.5 + abs(player_x))

        return reward, components

    def step(self, action):
        """
        Step through the environment using the given action and calculate rewards.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
