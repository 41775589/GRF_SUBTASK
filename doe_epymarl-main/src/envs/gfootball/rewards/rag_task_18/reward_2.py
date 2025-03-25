import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward to encourage central midfielders to focus on transitions
    and controlled pacing, enhancing gameplay synergy.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]  # 2D position

            # Simplification: assume index 2 and 3 are central midfielders
            mid_idx = [2, 3]  
            if o['active'] in mid_idx:
                for other_idx in [i for i in mid_idx if i != o['active']]:
                    distance = np.linalg.norm(o['left_team'][o['active']] - o['left_team'][other_idx])
                    reward_component = 0.0

                    # Reward for moving closer together to transition the ball more effectively
                    if distance < 0.1:
                        reward_component += self._checkpoint_reward * (0.1 - distance)

                    # Check if the controlled pacing is maintained, that is not too fast not too slow
                    speed = np.linalg.norm(o['left_team_direction'][o['active']])
                    if 0.01 < speed < 0.03:
                        reward_component += self._checkpoint_reward
                        
                    components["checkpoint_reward"][rew_index] += reward_component
                    reward[rew_index] += reward_component

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
