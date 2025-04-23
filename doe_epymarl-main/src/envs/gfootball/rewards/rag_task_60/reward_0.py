import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive training by encouraging precise stopping and starting movements."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Keep track of player position changes to detect stop-start movements
        self.previous_positions = {}
        self.previous_sticky_actions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward values
        self.stop_start_reward = 0.05  # small reward for effective stop-start actions
        self.move_penalty = -0.01  # small penalty to discourage unnecessary movement

    def reset(self):
        # Reset the tracking dictionaries on environment resets
        self.previous_positions = {}
        self.previous_sticky_actions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the current reward wrapper state to a pickleable dictionary
        to_pickle['CheckpointRewardWrapper'] = {'previous_positions': self.previous_positions,
                                                'previous_sticky_actions': self.previous_sticky_actions}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the state from the pickleable dictionary
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.previous_positions = state_info['previous_positions']
        self.previous_sticky_actions = state_info['previous_sticky_actions']
        return from_pickle

    def reward(self, reward):
        # Obtaining environment observations
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        for rew_index, o in enumerate(observation):
            current_position = np.array(o['left_team'] if o['active'] in o['left_team'] else o['right_team'])
            current_sticky = np.array(o['sticky_actions'])
            previous_position = self.previous_positions.get(rew_index, current_position)
            previous_sticky = self.previous_sticky_actions.get(rew_index, current_sticky)
            
            # Identifying start and stop movement actions
            if np.array_equal(current_sticky, previous_sticky) and not np.array_equal(current_position, previous_position):
                reward[rew_index] += self.move_penalty
                # Negative reward for moving without changing action, discouraging unnecessary movement
            elif not np.array_equal(current_sticky, previous_sticky) and np.array_equal(current_position, previous_position):
                reward[rew_index] += self.stop_start_reward
                # Reward for changing action without changing position, promoting precise control

            # Update the stored positions and actions
            self.previous_positions[rew_index] = current_position
            self.previous_sticky_actions[rew_index] = current_sticky

        return reward, components

    def step(self, action):
        # Wrapper step function to keep track of components
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
