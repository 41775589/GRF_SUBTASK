import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that refines rewards for goalkeeper training tasks, focusing on clearing strategies and high-pressure backups.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.goalkeeper_index = None
        self.last_cleared_to_player = None
        self.clear_ball_reward = 0.5

    def reset(self):
        super().reset()
        self.last_cleared_to_player = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = super().get_state(to_pickle)
        to_pickle['last_cleared_to_player'] = self.last_cleared_to_player
        return to_pickle

    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.last_cleared_to_player = from_pickle.get('last_cleared_to_player', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clear_ball_reward": [0.0, 0.0]
        }
        
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Identify goalkeeper by role
            if self.goalkeeper_index is None and 'right_team_roles' in o:
                self.goalkeeper_index = np.where(o['right_team_roles'] == 0)[0][0]

            # Efficient clearing reward logic
            if 'ball_owned_player' in o and o['ball_owned_player'] == self.goalkeeper_index:
                if 'action' in o:
                    # Assuming action index for 'clear_ball' command is predefined and known
                    if o['action'] == clear_ball_action_index:
                        # Check if the ball has been cleared to a specific player
                        cleared_player = self.detect_clear_target_player(o)
                        if cleared_player is not None and cleared_player != self.last_cleared_to_player:
                            components["clear_ball_reward"][idx] = self.clear_ball_reward
                            reward[idx] += components["clear_ball_reward"][idx]
                            self.last_cleared_to_player = cleared_player

        return reward, components

    def detect_clear_target_player(self, observation):
        """
        Logic to detect which player the ball is cleared towards.
        """
        # Simplified assumption code - defining how to map the clear action to a target player.
        # Implement specific logic based on environment observation details.
        return np.random.choice(range(len(observation['right_team'])))  # random example

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
