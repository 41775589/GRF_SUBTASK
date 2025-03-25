import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive reward focusing on position, interception, and tackling actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.interception_counter = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.interception_counter = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self.previous_ball_position,
            'interception_counter': self.interception_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        config = from_pickle['CheckpointRewardWrapper']
        self.previous_ball_position = config['previous_ball_position']
        self.interception_counter = config['interception_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_actions_reward": [0.0] * len(reward)
        }

        # Apply additional rewards based on defensive capabilities
        for idx, (rew, obs) in enumerate(zip(reward, observation)):
            # Detect and reward interceptions
            if self.previous_ball_position is not None and obs['ball_owned_team'] == 1:
                distance = np.linalg.norm(np.array(obs['ball']) - np.array(self.previous_ball_position))
                if distance < 0.1 and obs['ball_owned_player'] != -1:
                    components['defensive_actions_reward'][idx] += 0.5
                    self.interception_counter[idx] += 1

            # Reward for successful tackles or blocks
            defensive_actions = (obs['sticky_actions'][6],   # Slide
                                  obs['sticky_actions'][1],   # Stop dribble
                                  obs['sticky_actions'][0])  # Stop moving
            if any(defensive_actions):
                components['defensive_actions_reward'][idx] += 0.05
          
            # Update total reward with components
            reward[idx] = 1 * components['base_score_reward'][idx] + components['defensive_actions_reward'][idx]
        
        # Store ball position for next step's calculation
        self.previous_ball_position = [obs['ball'] for obs in observation]

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
