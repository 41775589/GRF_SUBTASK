import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for offensive maneuvers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._ball_progress_checkpoints = {}
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_progress_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CustomRewardWrapper'] = self._ball_progress_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._ball_progress_checkpoints = from_pickle['CustomRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward,
            "progress_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x = o['ball'][0]
            
            # Reward progression towards opponent's goal
            if o['ball_owned_team'] == 1:  # Assume team 1 is the agent's team
                checkpoints_collected = self._ball_progress_checkpoints.get(rew_index, 0)
                expected_checkpoint = int((ball_x + 1) * self._num_checkpoints / 2)
                while checkpoints_collected < self._num_checkpoints and checkpoints_collected < expected_checkpoint:
                    components["progress_reward"][rew_index] += self._checkpoint_reward
                    checkpoints_collected += 1
                self._ball_progress_checkpoints[rew_index] = checkpoints_collected
            
            reward[rew_index] += components["progress_reward"][rew_index]

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
