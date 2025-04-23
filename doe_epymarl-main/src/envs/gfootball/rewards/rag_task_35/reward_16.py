import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on strategic positioning and movement patterns.
    Encourages pivoting between defensive and attacking strategies effectively.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize counter for actions to trace directional changes
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the sticky actions counter on environment reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store the current state of sticky actions
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Retrieve the last state of sticky actions from stored state
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Base reward
            components.setdefault("base_score_reward", [0.0] * len(reward))
            
            # Reward for moving towards strategic positions and pivoting between roles
            strategic_positioning_reward = 0.0
            if o['game_mode'] == 0:  # Normal gameplay
                # Assume positions near midfield are strategic pivot positions
                midfield_x = 0.0
                strategy_zone_y_low, strategy_zone_y_high = -0.25, 0.25
                
                # Check if player is in a strategic pivot area aligning with middle of the field
                if any((midfield_x - 0.1 <= player_pos[0] <= midfield_x + 0.1) and
                       (strategy_zone_y_low <= player_pos[1] <= strategy_zone_y_high)
                       for player_pos in o['right_team']):
                    strategic_positioning_reward += 0.05

                # Encourage switching between offensive and defensive strategies
                if self.sticky_actions_counter[4] or self.sticky_actions_counter[3]:  # moving right or left
                    strategic_positioning_reward += 0.02

            # Store individual components
            components.setdefault('strategic_positioning_reward', []).append(strategic_positioning_reward)
            
            # Update total reward for the current environment step
            reward[idx] += strategic_positioning_reward

            # Update sticky_actions_counter
            self.sticky_actions_counter = o['sticky_actions']

        return reward, components

    def step(self, action):
        # Standard step with rewiring to include additional reward components in the info dictionary
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
