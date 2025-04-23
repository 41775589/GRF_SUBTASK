import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function tailored for mastering short passes under defensive pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define a reward magnitude for successful pass under pressure
        self._pass_reward = 0.5
        self._ball_possession_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Initialize the detailed reward components
        components = {
            "base_score_reward": reward.copy(),
            "pass_under_pressure_reward": [0.0] * len(reward),
            "ball_possession_reward": [0.0] * len(reward)
        }
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            active_player_team = 'left_team' if o['active'] in o['left_team'] else 'right_team'
            team_mates_positions = o[active_player_team]
            
            # Checking ball possession under pressure situation
            if o['ball_owned_team'] == 0 and active_player_team == 'left_team' or o['ball_owned_team'] == 1 and active_player_team == 'right_team':
                # Calculate the distance from all opponents
                opponents = 'right_team' if active_player_team == 'left_team' else 'left_team'
                distances = np.linalg.norm(team_mates_positions - o[opponents], axis=1)
                
                # Reward for keeping the ball under pressure
                if np.any(distances < 0.1):  # Assuming 0.1 as a pressure distance threshold
                    components["ball_possession_reward"][rew_index] = self._ball_possession_reward
                    reward[rew_index] += components["ball_possession_reward"][rew_index]

                # Evaluate if a pass was made under pressure
                if o['sticky_actions'][9] == 1:  # Assuming index 9 is the passing action
                    if np.any(distances < 0.15):  # Passing under pressure if opponents are very close
                        components["pass_under_pressure_reward"][rew_index] = self._pass_reward
                        reward[rew_index] += components["pass_under_pressure_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
