import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that adds a specialized reward for goalie coordination tasks.
    This reward focuses on the goalkeeper's ability to clear the ball during
    high-pressure situations and distribute it effectively to specific outfield players.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Specific variables for tracking goalkeeper and defense effectiveness
        self.goalie_clearances = 0
        self.successful_passes = 0
        self.goalkeeper_index = None
        self.outfield_targets = []

    def reset(self):
        # Resetting the counter for goalkeeper actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalie_clearances = 0
        self.successful_passes = 0
        self.goalkeeper_index = None
        self.outfield_targets = []
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save current state related to the new reward mechanisms
        current_state = self.env.get_state(to_pickle)
        current_state['goalie_clearances'] = self.goalie_clearances
        current_state['successful_passes'] = self.successful_passes
        return current_state

    def set_state(self, state):
        # Restore the state
        from_pickle = self.env.set_state(state)
        self.goalie_clearances = from_pickle['goalie_clearances']
        self.successful_passes = from_pickle['successful_passes']
        return from_pickle

    def reward(self, reward):
        # Access the observation to add rewards
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "goalkeeper_clearance_reward": 0.0, 
                      "successful_pass_reward": 0.0}

        if observation is None:
            return reward, components
        
        # Find the goalkeeper's index once
        if self.goalkeeper_index is None:
            for i, role in enumerate(observation['left_team_roles']):
                if role == 0:  # Goalkeeper role index
                    self.goalkeeper_index = i
                    break

        # Goalkeeper actions and clearing ball under pressure conditions
        if observation['ball_owned_player'] == self.goalkeeper_index and observation['ball_owned_team'] == 0:
            if observation['game_mode'] in [3, 4]:  # High-pressure modes like FreeKick or Corner
                self.goalie_clearances += 1
                components['goalkeeper_clearance_reward'] = 0.5  # +0.5 reward for clearing under pressure

        # Reward for successful passes from the goalkeeper to specific outfield players
        if observation['ball_owned_team'] == 0 and observation['last_action'] in [9, 10]:  # Passing actions
            if observation['ball_owned_player'] in self.outfield_targets:
                self.successful_passes += 1
                components['successful_pass_reward'] = 0.3  # +0.3 reward for successful targeting

        # Aggregate rewards
        total_reward = sum(reward) + components['goalkeeper_clearance_reward'] + components['successful_pass_reward']
        return [total_reward] * len(reward), components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info.update({f"component_{key}": value for key, value in components.items()})
        return observation, reward, done, info
