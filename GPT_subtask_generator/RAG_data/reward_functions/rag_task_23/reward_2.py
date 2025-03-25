import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a targeted reward for defensive role synergy and penalty area skills."""

    def __init__(self, env):
        super().__init__(env)  # Initialize the base class
        # Initialize the counter for sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the environment including custom components."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment including custom components."""
        from_pickle = self.env.set_state(state)
        # Placeholder for custom components, if needed.
        return from_pickle

    def reward(self, reward):
        """Adjust the rewards based on strategic defensive positions and actions near the penalty area."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, copy=True)}

        for idx, o in enumerate(observation):
            is_defender = (o['left_team_roles'][o['active']] == 2 or  # 2 is typically a defensive role
                           o['left_team_roles'][o['active']] == 3)
            near_penalty_area = o['left_team'][o['active']][0] > 0.7  # X position greater signifies closeness to penalty

            if is_defender and near_penalty_area:
                # Check if the controlled player is blocking an opponent or intercepting the ball
                opponent_positions = o['right_team']
                own_position = o['left_team'][o['active']]
                distances = np.linalg.norm(opponent_positions - own_position, axis=1)
                close_opponents = np.any(distances < 0.1)  # Close to any opponent

                if close_opponents:
                    # Provide extra reward for good defensive positioning and actions
                    reward[idx] += 0.5

        return reward, components

    def step(self, action):
        """Apply the action, adjust the rewards, and return the observations and rewards."""
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter for debugging purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, new_reward, done, info
