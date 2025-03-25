import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focused on enhancing defensive 
    coverage through sprinting to improve agents' ability to quickly reposition 
    across the field in response to dynamic game changes.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.num_sprint_zones = 5  # Divide the field into 5 zones for sprint checkpoints
        self.sprint_reward = 0.1  # Reward for reaching a new sprint zone
        self.last_player_positions = {}
        self.zone_boundaries = np.linspace(-1, 1, self.num_sprint_zones + 1)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_player_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.last_player_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_player_positions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_position = o['right_team'][o['active']][0]  # x coordinate of the active player
            last_position = self.last_player_positions.get(rew_index, -1.1)  # default outside left boundary

            # Determine current and last zones for the player's x position
            current_zone = np.digitize(current_position, self.zone_boundaries) - 1
            last_zone = np.digitize(last_position, self.zone_boundaries) - 1

            # Check sprint action is active and if player advanced to a new zone
            if o['sticky_actions'][8] > 0 and current_zone > last_zone:
                components["sprint_reward"][rew_index] = self.sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

            # Update last position with the current
            self.last_player_positions[rew_index] = current_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add additional info for debugging and analytics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
