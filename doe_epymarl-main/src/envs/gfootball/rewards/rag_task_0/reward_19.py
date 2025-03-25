import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for offensive football strategies. 
    Rewards are given for accurate shooting, effective dribbling evasions, and differentiated passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for pickling."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from an unpickled state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward logic focused on improving offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": copy.deepcopy(reward), 
                      "shooting_accuracy": [0.0] * len(reward), 
                      "dribble_evasion": [0.0] * len(reward), 
                      "pass_breakthrough": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Handle accurate shooting rewards
            if o['game_mode'] in {6}:  # Penalty kick positions
                if o['score'][0] > o['score'][1]:  # Assuming left team is the agent team
                    components['shooting_accuracy'][idx] = 0.5
                    reward[idx] += components['shooting_accuracy'][idx]

            # Handle dribbling evasion rewards
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if self.sticky_actions_counter[9] == 1:  # Assuming action 9 is 'dribble'
                    components['dribble_evasion'][idx] = 0.3
                    reward[idx] += components['dribble_evasion'][idx]

            # Handle pass breakthrough rewards
            if 'game_mode' in o and o['game_mode'] == 4:  # Assuming game mode 4 is for special plays like corners
                components['pass_breakthrough'][idx] = 0.3
                reward[idx] += components['pass_breakthrough'][idx]

            self.sticky_actions_counter = o['sticky_actions']  # Update the sticky action counters

        return reward, components

    def step(self, action):
        """Execute a step in the environment, taking the defined action."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        return observation, reward, done, info
