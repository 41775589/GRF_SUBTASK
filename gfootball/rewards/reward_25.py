import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on offensive strategies in football."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_position = None
        self.dribbling_bonus = 0.1
        self.passing_bonus = 0.1
        self.shooting_bonus = 0.3

    def reset(self):
        """Reset environment and clean up variables."""
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize relevant state information for checkpointing."""
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize and set state from a checkpoint."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        """Reward function augmented with dense rewards for offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_bonus": [0.0] * len(reward),
                      "passing_bonus": [0.0] * len(reward),
                      "shooting_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        ball_position = observation['ball'][:2]  # only x, y coordinates
        if self.previous_ball_position is not None:
            ball_movement = np.linalg.norm(ball_position - self.previous_ball_position)
        else:
            ball_movement = 0
        
        # Update the reward based on game mode and ball control
        for i, o in enumerate(observation):
            # Encourage dribbling by rewarding movement with ball possession
            if o['ball_owned_team'] == 1 and ball_movement > 0:
                components["dribbling_bonus"][i] = self.dribbling_bonus * ball_movement
                reward[i] += components["dribbling_bonus"][i]
            
            # Encourage successful passes by detecting changes in ball ownership
            if o['ball_owned_team'] != self.previous_ball_position and o['ball_owned_team'] == 1:
                components["passing_bonus"][i] = self.passing_bonus
                reward[i] += components["passing_bonus"][i]
            
            # High reward for scoring
            if o['score'][1] > o['score'][0]:  # assuming '1' is the controlled team
                components["shooting_bonus"][i] = self.shooting_bonus
                reward[i] += components["shooting_bonus"][i]
        
        self.previous_ball_position = ball_position
        return reward, components

    def step(self, action):
        """Apply the action, adjust the reward, and return the updated observation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
