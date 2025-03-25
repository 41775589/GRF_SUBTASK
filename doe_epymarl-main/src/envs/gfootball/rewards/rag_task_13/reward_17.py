import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on defensive maneuvers for a 'stopper' player."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the reward wrapper state for a new game episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieves the current state of the underlying environment."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the underlying environment as per the given state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Custom reward function to enhance the skill of a 'stopper' defense player."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components.setdefault("defensive_bonus", [0.0] * len(reward))
            if 'left_team_roles' in o and 'right_team_roles' in o:
                # Identify stopper players using role indices, could be other depending on definition
                stopper_role_indices = [i for i, role in enumerate(o['left_team_roles']) if role == 2]  # e.g., indices of 'stopper' role
            
                # Calculate distances from ball for identified stopper players and encourage blocking
                for idx in stopper_role_indices:
                    player_pos = o['left_team'][idx]
                    ball_pos = o['ball'][:2]  # Without considering z-coordinate
                    distance = np.linalg.norm(ball_pos - player_pos)
                    
                    # Closer to ball means likely to block or intercept
                    if distance <= 0.1:  # Threshold for 'close enough to act'
                        components["defensive_bonus"][rew_index] += 1
                
                # Normalize reward and add to total reward
                if components["defensive_bonus"][rew_index] > 0:
                    components["defensive_bonus"][rew_index] /= len(stopper_role_indices)

            # Update reward with bonuses
            reward[rew_index] += components["defensive_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """Steps through the environment, collects rewards, and add additional info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
