import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on enhancing team defensive capabilities."""
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defensive behavior rewards
        self.tackles_successful_reward = 0.5
        self.goalkeeper_save_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Could store custom states if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Reward defensive actions specifically for goalkeepers and defenders:
        - Goalkeeper saves
        - Successful tackles by defenders
        """
        observation = self.env.unwrapped.observation()  # Receive decoded and individual obs
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_save_reward": [0.0] * len(reward),
                      "tackles_successful_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            # Reward goalkeeper for saves or good positioning
            if o['left_team_roles'][o['active']] == 0:  # Assuming role 0 is the goalkeeper
                if o['ball_owned_team'] == 1 and (o['ball'][0] < -0.9):  # Ball near own goal
                    components['goalkeeper_save_reward'][rew_index] = self.goalkeeper_save_reward

            # Reward defenders for successful tackles (assumed role codes 1 to 4 are defenders)
            if o['left_team_roles'][o['active']] in [1, 2, 3, 4]:
                # Simple heuristic: if last action was a directional movement towards the ball carrier from the opposing team
                direction_of_play = np.sign(o['ball_direction'][0])
                active_direction = np.sign(o['left_team_direction'][o['active']][0])
                if active_direction == direction_of_play:  # Moving towards ball means potentially tackled
                    components['tackles_successful_reward'][rew_index] = self.tackles_successful_reward

            # Calculate the final modified reward for each player
            reward[rew_index] += (components['goalkeeper_save_reward'][rew_index] +
                                  components['tackles_successful_reward'][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Append new rewards and reward components into info for tracking
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter for each possible action
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
