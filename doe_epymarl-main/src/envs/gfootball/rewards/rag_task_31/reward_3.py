import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A RewwardWrapper that emphasizes defensive actions and penalizes unwarranted aggression or 
    missed tackles to incentivize better and more timely defensive responses.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_penalty = -0.1
        self.tackle_reward = 0.2
        self.slide_reward = 0.3
        self.ball_recovery_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_actions_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            action_taken = player_obs['sticky_actions']
            ball_owned_team = player_obs['ball_owned_team']
            ball_owned_player = player_obs['ball_owned_player']
            active_player = player_obs['active']

            # Special reward for making tackles
            if action_taken[6]:  # Assume index 6 corresponds to 'action_tackle'
                components["defensive_actions_reward"][rew_index] += self.tackle_reward

            # Special reward for slide actions leading to ball recovery
            if action_taken[9]:  # Assume index 9 corresponds to 'action_slide'
                if ball_owned_team == 0 and ball_owned_player == active_player:
                    components["defensive_actions_reward"][rew_index] += self.slide_reward
            
            # Corrective action if ball recovery is successful, irrespective of slide/tackle
            if player_obs['right_team_roles'][active_player] != 1:  # Not goalkeeper
                if ball_owned_team == 0 and ball_owned_player == active_player:
                    components["defensive_actions_reward"][rew_index] += self.ball_recovery_reward

            # Penalty for unwarranted sprint usage
            if action_taken[8]:  # Assume index 8 corresponds to 'action_sprint'
                components["defensive_actions_reward"][rew_index] += self.sprint_penalty
            
            # Update the final reward for this player with additional components
            reward[rew_index] += components["defensive_actions_reward"][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding additional info in info dict for all components
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
