import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the synergistic effectiveness of central midfielders
    by emphasizing controlled pace and seamless transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_reward = 0.05
        self.pace_management_reward = 0.03

    def reset(self):
        """Reset the sticky actions counter and environment at the start of each episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Custom reward function to enhance midfield synergy."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward),
                      "pace_management_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['game_mode'] != 0:  # Exclude non-normal modes.
                continue

            # Encourage players for quick transitions (passing efficiently).
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                if o['ball_owned_team'] == 0:  # Assumed '0' is central midfield
                    components['transition_reward'][rew_index] += self.transition_reward
                reward[rew_index] += components['transition_reward'][rew_index]

            # Encourage pace control by rewarding low ball speed near opponents
            ball_speed = np.linalg.norm(o['ball_direction'][0:2])  # ignore z-component
            if ball_speed < 0.1:  # threshold for low ball speed
                components['pace_management_reward'][rew_index] += self.pace_management_reward
            reward[rew_index] += components['pace_management_reward'][rew_index]
        
        return reward, components 

    def step(self, action):
        """Step function to process actions and update the environment state."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
