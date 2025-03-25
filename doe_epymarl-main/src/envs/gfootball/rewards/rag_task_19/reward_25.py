import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for controlling the midfield and enhancing defense strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5
        self._collected_checkpoints = {}
        self.midfield_control_reward = 0.05
        self.defensive_actions_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "defensive_actions_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check ball possession and position
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Assuming the right team is the one we are training
                # Encourage keeping the ball in midfield
                midfield_threshold = [-0.2, 0.2]
                if midfield_threshold[0] <= o['ball'][0] <= midfield_threshold[1]:
                    components['midfield_control_reward'][rew_index] = self.midfield_control_reward
                    reward[rew_index] += components['midfield_control_reward'][rew_index]

                # Reward defensive maneuvers: intercept or block in own half
                if o['ball'][0] < 0:  # The ball is in our half
                    # Identify if active player made a block or an intercept
                    if 'action' in o and o['action'] in ['intercept', 'block']:
                        components['defensive_actions_reward'][rew_index] = self.defensive_actions_reward
                        reward[rew_index] += components['defensive_actions_reward'][rew_index]

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Record reward components to info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        
        # Reset sticky actions counter and populate it from current observations
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
