import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized goalkeeper coordination reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_activities = [0, 0]  # Tracks clearances and backups

    def reset(self):
        """Reset the reward-specific state when the environment is also reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_activities = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get state that includes the checkpoint state for saving the environment."""
        to_pickle['goalkeeper_activities'] = self.goalkeeper_activities
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state including checkpoints from the saved state."""
        from_pickle = self.env.set_state(state)
        self.goalkeeper_activities = from_pickle['goalkeeper_activities']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on goalkeeper performance in high-pressure scenarios."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'goalkeeper_clearance_reward': [0.0] * len(reward),
                      'goalkeeper_backup_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['active'] == 0:  # Assuming index 0 is the goalkeeper
                # For each successful clearance
                if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction'][0:2]) > 0.5:
                    components['goalkeeper_clearance_reward'][rew_index] = 1.0
                    reward[rew_index] += components['goalkeeper_clearance_reward'][rew_index]
                    self.goalkeeper_activities[0] += 1

                # If the goalkeeper moves back towards the goal under pressure
                goal_pos_y = o['right_team'][0][1] if o['right_team'][0][1] > 0 else -o['left_team'][0][1]
                if abs(o['left_team'][0][1] - goal_pos_y) > 0.1:
                    components['goalkeeper_backup_reward'][rew_index] = 0.5
                    reward[rew_index] += components['goalkeeper_backup_reward'][rew_index]
                    self.goalkeeper_activities[1] += 1
            
        return reward, components

    def step(self, action):
        """Step through the environment and enhance info with reward components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
