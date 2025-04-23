import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance the coordination between midfielders and strikers in offensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.midfielder_to_striker_passes = {}
        self.striker_attempts = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        # Reset the sticky actions and other state variables
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_to_striker_passes = {}
        self.striker_attempts = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['midfielder_to_striker_passes'] = self.midfielder_to_striker_passes
        state['striker_attempts'] = self.striker_attempts
        return state

    def set_state(self, state):
        state_from_env = self.env.set_state(state)
        self.midfielder_to_striker_passes = state.get('midfielder_to_striker_passes', {})
        self.striker_attempts = state.get('striker_attempts', {})
        return state_from_env

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shot_reward": [0.0] * len(reward)
        }
        
        for idx, (r, o) in enumerate(zip(reward, obs)):
            # Check if the ball is being passed by a midfielder to a striker
            if self.is_midfielder(o['active'], o) and self.has_passed_to_striker(o):
                components["pass_reward"][idx] = 0.1
                self.midfielder_to_striker_passes[idx] = self.midfielder_to_striker_passes.get(idx, 0) + 1
            
            # Check if the striker is attempting to score
            if self.is_striker(o['active'], o) and self.has_attempted_shot(o):
                components["shot_reward"][idx] = 0.3
                self.striker_attempts[idx] = self.striker_attempts.get(idx, 0) + 1
        
        # Sum base reward with additional rewards
        total_reward = [
            base + pass_r + shot_r
            for base, pass_r, shot_r in zip(components["base_score_reward"], components["pass_reward"], components["shot_reward"])
        ]
        
        return total_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
    
    def is_midfielder(self, active_player_index, observation):
        return observation['left_team_roles'][active_player_index] in [4, 5, 6]  # Midfield roles

    def has_passed_to_striker(self, observation):
        # Simplified logic to decide if a pass has been made to a striker
        return observation['action'] == 'pass' and observation['pass_target_role'] == 9  # Striker role

    def is_striker(self, active_player_index, observation):
        return observation['left_team_roles'][active_player_index] == 9  # Striker role

    def has_attempted_shot(self, observation):
        # Simplified logic to decide if a shot has been attempted
        return 'shoot' in observation['action']
