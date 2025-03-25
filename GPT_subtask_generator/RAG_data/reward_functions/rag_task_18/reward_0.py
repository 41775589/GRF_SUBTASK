import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on central midfield effectiveness and controlled transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Retrieve any necessary state if needed, currently not used.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Base score
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Analyze positions and movement of central midfielders
            y_positions = o['left_team'][:, 1]  # all y-positions of players
            ball_y_pos = o['ball'][1]  # y-position of the ball
            player_roles = o['left_team_roles']
            # Focusing on central midfield roles (5 = CM)
            midfielders = np.where(player_roles == 5)[0]
            
            # Check for midfielders in control and near the ball in midfield areas
            if any((np.abs(y_positions[midfielder] - ball_y_pos) < 0.1 for midfielder in midfielders)):
                components["transition_reward"][rew_index] = 0.05  # small bonus for maintaining central control

            # Reward transitions: controlled transitions in ball movement and pace handling
            # Assuming ball_owned_team 0 is left
            if o['ball_owned_team'] == 0 and any([o['sticky_actions'][8], o['sticky_actions'][9]]): # check sprint or dribble
                components["transition_reward"][rew_index] += 0.05  # Reward controlled progression
            
            # Update total reward with computed components
            reward[rew_index] = sum(components.values())

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
