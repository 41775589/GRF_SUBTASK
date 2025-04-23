import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on team synergy during possession changes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        self.possession_changes = 0
        self.possession_change_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        self.possession_changes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_ball_owner'] = self.previous_ball_owner
        to_pickle['CheckpointRewardWrapper_possession_changes'] = self.possession_changes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['CheckpointRewardWrapper_ball_owner']
        self.possession_changes = from_pickle['CheckpointRewardWrapper_possession_changes']
        return from_pickle

    def reward(self, reward):
        """Update the reward based on strategic possession changes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            current_ball_owner = None
            if o['ball_owned_team'] == 0:  # Ball is owned by the left team
                current_ball_owner = o['left_team'][o['ball_owned_player']]
            elif o['ball_owned_team'] == 1:  # Ball is owned by the right team
                current_ball_owner = o['right_team'][o['ball_owned_player']]
            
            # Checking for change in possession
            if current_ball_owner is not None and self.previous_ball_owner is not None:
                if current_ball_owner != self.previous_ball_owner:
                    reward[i] += self.possession_change_reward
                    self.possession_changes += 1
                    components["possession_change_reward"][i] = self.possession_change_reward
            
            # Update previous owner
            self.previous_ball_owner = current_ball_owner

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
