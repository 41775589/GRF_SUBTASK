import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for precise Stop-Dribble under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
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
                      "pressure_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            # Check if the active player has the ball and is performing a stop-dribble under high pressure
            if 'sticky_actions' in o and 'right_team_active' in o:
                # Sticky action index 9 corresponds to the "dribble" action in the Football environment
                is_dribbling = o['sticky_actions'][9] == 1
                # Pressure is assumed high if multiple opponents are nearby the ball handler
                close_opponents = sum([np.linalg.norm(o['ball'] - opponent_pos) < 0.1 for opponent_pos in o['right_team']])
                
                if is_dribbling and close_opponents > 1:
                    components["pressure_control_reward"][idx] = 0.25  # additional reward for managing dribble under pressure
                    reward[idx] += components["pressure_control_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
