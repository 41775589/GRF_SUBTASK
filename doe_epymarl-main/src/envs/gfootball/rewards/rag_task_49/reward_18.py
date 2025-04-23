import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage shooting from central field positions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Middle third boundaries of the pitch in x-coordinate [-1, 1] system
        self.central_zone_min = -0.33
        self.central_zone_max = 0.33
        # Bonus reward for actions taken in this central zone
        self.central_zone_bonus = 0.5
        # Initialize sticky actions counter
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "central_field_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Process each agent's observation
        for rew_index, o in enumerate(observation):
            # Check if player is close to middle of the pitch
            if 'ball' in o and self.central_zone_min <= o['ball'][0] <= self.central_zone_max:
                # Check if the player's team owns the ball
                if o['ball_owned_team'] == 0:  # assuming '0' represents the controlled team
                    # Apply central field bonus
                    components["central_field_bonus"][rew_index] = self.central_zone_bonus
                    # Modify the reward
                    reward[rew_index] += components["central_field_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info.update({
            f"component_{key}": sum(value) for key, value in components.items()
        })
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action  
        return observation, reward, done, info
