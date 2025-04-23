import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for mastering advanced ball control and passing under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self._pass_quality_threshold = 0.2  # Threshold for considering a pass as effective under pressure
        self._ball_control_quality_threshold = 0.1  # Threshold for considering good ball control
        self._pass_control_reward = 0.05
        self._ball_control_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_control_reward": [0.0] * len(reward),
            "ball_control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Consider ball control and passing under pressure
            if o['ball_owned_team'] == 0 and o['designated'] == o['active']:
                # Active player has the ball and is the designated player
                ball_control_quality = np.linalg.norm(o['ball_direction'])
                if ball_control_quality >= self._ball_control_quality_threshold:
                    reward[rew_index] += self._ball_control_reward
                    components['ball_control_reward'][rew_index] = self._ball_control_reward

                # Check pass control by active actions
                if o['sticky_actions'][6] or o['sticky_actions'][7] or o['sticky_actions'][9]:  # High Pass, Long Pass, Short Pass
                    pass_quality = np.linalg.norm(o['ball_direction'] - o['left_team_direction'][o['active']])
                    if pass_quality <= self._pass_quality_threshold:
                        reward[rew_index] += self._pass_control_reward
                        components['pass_control_reward'][rew_index] = self._pass_control_reward

            # Reward for having control over the ball and successfully passing under pressure
            reward[rew_index] += base_reward

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
