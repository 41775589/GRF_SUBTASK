import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards defensive actions typical of a 'sweeper' in football."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        # Default reward components to zero.
        components = {"base_score_reward": reward,
                      "clearance_reward": 0.0,
                      "tackle_reward": 0.0,
                      "recovery_position_reward": 0.0}

        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            # Reward for clearing the ball from the defensive zone.
            if 'ball' in o and o['ball_owned_player'] == o['active'] \
               and o['ball_owned_team'] == o['left_team_active'] and o['ball'][0] < -0.5:
                components['clearance_reward'] += 0.3

            # Reward for tackles (assuming a change in ball possession near the player triggers this)
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and np.linalg.norm(o['left_team'][o['active']] - o['ball'][:2]) < 0.1:
                components['tackle_reward'] += 0.5

            # Reward for position recovery.
            if 'right_team_direction' in o and np.linalg.norm(o['right_team_direction'][o['active']]) > 0.05:
                components['recovery_position_reward'] += 0.2

        # Calculate total rewards by summing up base reward and all additional rewards.
        total_reward = reward + components['clearance_reward'] + components['tackle_reward'] + components['recovery_position_reward']
        return total_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
