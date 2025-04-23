import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful counterattacks via long passes."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)
        self.counterattack_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "counterattack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            current_ball_position = o['ball'][:2]  # ignoring z-coordinate
            ball_moved_significantly = np.linalg.norm(current_ball_position - self.last_ball_position[:2]) > 0.7
            is_own_team = o['ball_owned_team'] == 0

            if ball_moved_significantly and is_own_team:
                # Checking if the long pass has transitioned from our defensive half to offensive half
                if self.last_ball_position[0] < 0 and current_ball_position[0] > 0:
                    reward[index] += self.counterattack_reward
                    components["counterattack_reward"][index] = self.counterattack_reward

            # Update last ball position
            self.last_ball_position = o['ball']
        
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
