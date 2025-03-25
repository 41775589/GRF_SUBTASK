import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on dribbling, evasion, and effective sprint usage."""
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints based on zones in the opponent's half
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2
        
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
                      "checkpoint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            player_pos = o['right_team'][o['designated']]
            owned_by_player = (o['ball_owned_team'] == 1) and (o['ball_owned_player'] == o['designated'])
            
            # Calculate the distance from the player to the opponent's goal
            distance_to_goal = 1 - player_pos[0]  # Opponent's goal is at x=1
            
            # Checkpoint reward for dribbling and pushing forward
            if owned_by_player and distance_to_goal < 0.5:
                checkpoint_index = int((0.5 - distance_to_goal) / 0.1)  # 5 checkpoints each 10% of the way
                reward[rew_index] += self._checkpoint_reward
                components["checkpoint_reward"][rew_index] += self._checkpoint_reward * checkpoint_index
            
            # Bonus for using sprint effectively while dribbling around defenders
            if owned_by_player and o['sticky_actions'][8] == 1: # Sprint action index is 8
                reward[rew_index] += 0.05
                components["checkpoint_reward"][rew_index] += 0.05
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
