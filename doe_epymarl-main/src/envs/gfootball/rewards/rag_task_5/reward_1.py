import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function with defense and quick transition rewards."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_counter = 0
        self.defensive_position_reward = 0.05
    
    def reset(self):
        """Reset the environment and the reward parameters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_counter = 0
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Encode the state of the reward wrapper along with the environment's state."""
        to_pickle['transition_counter'] = self.transition_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Decode the state from the saved pickle and set it to the current state."""
        from_pickle = self.env.set_state(state)
        self.transition_counter = from_pickle['transition_counter']
        return from_pickle
    
    def reward(self, reward):
        """Custom reward logic focusing on defensive capabilities and rapid transitions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Reward defensive organization
                own_goal_x = -1  # Assume left team; change coordinates for right team if needed
                players_position = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                for player_pos in players_position:
                    if player_pos[0] < own_goal_x:
                        components["defensive_position_reward"][rew_index] += self.defensive_position_reward

                # Reward transitions
                if o['game_mode'] in [4, 6]:  # Free-kick or penalty modes
                    self.transition_counter += 1
                    components["transition_reward"][rew_index] = self.transition_counter * 0.1
            
            # Aggregate the rewards
            reward[rew_index] += (components["defensive_position_reward"][rew_index] +
                                  components["transition_reward"][rew_index])
        
        return reward, components

    def step(self, action):
        """Step function aggregates modification in rewards and push changes to environment's step."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
