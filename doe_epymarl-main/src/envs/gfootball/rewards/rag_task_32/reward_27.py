import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes wing play for crossing and sprinting especially for wingers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5
        self._wing_crossing_reward = 0.2
        self._checkpoint_positions = np.linspace(-0.4, 0.4, self._num_checkpoints)
        self._winger_position_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "crossing_reward": [0.0, 0.0], "position_reward": [0.0, 0.0]}
        if observation is None:
            return reward, components
        
        # Calculate rewards based on ball position and winger actions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Base score reward
            components['base_score_reward'][rew_index] = reward[rew_index]

            # Check if ball is close to the sidelines for crossing opportunities
            ball_y = o['ball'][1]
            for pos in self._checkpoint_positions:
                if abs(ball_y - pos) < 0.1:  # Ball close to one of the sideline checkpoints
                    components['crossing_reward'][rew_index] += self._wing_crossing_reward
                    
            # Encourage wingers to be positioned wide on the field for crossing
            active_player_y = o['left_team'][o['active']][1] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][1]
            if abs(active_player_y) > 0.3:  # Active player is near sideline areas
                components['position_reward'][rew_index] += self._winger_position_reward

        # Sum the individual components to calculate the final reward for each agent
        final_rewards = [sum(x) for x in zip(*components.values())]
        return final_rewards, components

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
