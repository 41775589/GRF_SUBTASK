import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on precise high passing skill improvement.
    The focus is on enhancing skills necessary for high passes such as trajectory control,
    power assessment, and situational application in gameplay.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.05  # Reward multiplier for high, well-targeted passes

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
        """
        Custom reward function that adds a reward for successful high passes performed under
        appropriate conditions, accounting for trajectory and power.
        """

        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 1 and o['ball'][2] > 0.1:  # Assuming z > 0.1 means a high pass
                ball_direction = o['ball_direction']

                # Check if pass is moving towards a teammate and adequately lifted
                for teammate_pos in o['right_team']:
                    if np.linalg.norm(ball_direction[:2] - (teammate_pos - o['ball'][:2])) < 0.1:
                        # Reward based on accuracy and appropriateness of the pass
                        components['high_pass_reward'][i] += self.high_pass_reward
                        reward[i] += components['high_pass_reward'][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Monitor sticky actions, which includes 'action_sprint' and 'action_dribble'
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
