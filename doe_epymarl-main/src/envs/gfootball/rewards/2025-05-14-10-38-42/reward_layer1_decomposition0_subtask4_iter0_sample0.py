import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on enhancing defensive coordination and interaction among defending agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and sticky actions count."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the wrapper along with the environment state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper along with the environment state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Customize reward to emphasize defensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defense_interaction_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                player_pos = o['right_team'][o['active']]
                ball_pos = o['ball']
                
                # Calculate the distance of the player to the ball
                distance_to_ball = np.sqrt((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)
                
                # Reward players for being close to the ball defensively
                if distance_to_ball < 0.1:
                    components["defense_interaction_reward"][rew_index] = 1.0  # Encourage defensive proximity
                    
            # Reward if a player performs a successful tackle or intercept
            if 'game_mode' in o and o['game_mode'] == 4:  # Assuming game_mode 4 refers to interceptions/tackles
                components["defense_interaction_reward"][rew_index] += 2.0  # Higher reward for successful defensive actions
                
            reward[rew_index] += components["defense_interaction_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Step the environment and apply the reward modifications."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Record sticky actions for each team member
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
