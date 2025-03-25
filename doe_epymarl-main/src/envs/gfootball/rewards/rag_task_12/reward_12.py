import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for an advanced midfielder/defender agent."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for agent_idx in range(len(reward)):
            o = observation[agent_idx]
            reward_delta = 0
            # Specific rewards for key actions and game states
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 1:  # if agent owns the ball
                # Encourage passing in advanced positions
                if abs(o['ball'][0]) > 0.5:  
                    reward_delta += 0.1

                # Encourage dribbling in pressured positions
                if o['sticky_actions'][9] == 1:  # is dribbling
                    reward_delta += 0.05

                # Sprinting/Dynamic movement reward weighted by distance to opponent goal
                if o['sticky_actions'][8] == 1:  # is sprinting
                    sprint_bonus = 0.05 * max(0, 1 - abs(o['ball'][1]))  # Higher reward closer to center y=0
                    reward_delta += sprint_bonus

            # Check for efficient positioning and ball control
            if o['designated'] == o['active']:
                # Encourage controlling the game pace by stopping sprint wisely
                if o['sticky_actions'][8] == 0:  # not sprinting (careful control)
                    reward_delta += 0.02

            # Update reward with calculated changes
            reward[agent_idx] += reward_delta
            components['advanced_midfielder_reward'] = [reward_delta] * len(reward)
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
