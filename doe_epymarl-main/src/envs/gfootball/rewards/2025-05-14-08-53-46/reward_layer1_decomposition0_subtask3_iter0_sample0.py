import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for exploiting open spaces and sprinting towards the goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky action counters and environment state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the state of the environment."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Custom reward function that encourages running towards open spaces and sprinting."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "open_space_sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Check whether the player is sprinting and moving towards open space
            if 'sticky_actions' in o and 'ball_owned_team' in o:
                # Assuming 8th index in sticky_actions is for sprint
                is_sprinting = o['sticky_actions'][8]
                open_space_bonus = 0.0
                if is_sprinting and o['ball_owned_team'] == 1:
                    player_pos = o['right_team'][o['active']]
                    goal_pos_x = 1.0  # X position of opponent's goal
                    distance_to_goal = np.abs(goal_pos_x - player_pos[0])
                    open_space_bonus = max(0, (1 - distance_to_goal))  # Closer to goal gets higher reward

                # Calculate the additional reward
                components["open_space_sprint_reward"][rew_index] = open_space_bonus
                reward[rew_index] = 1 * components["base_score_reward"][rew_index] + 10 * components["open_space_sprint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Apply action, get the results, augment with reward modifications and return."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
