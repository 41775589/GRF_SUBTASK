import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful long passes.
    Rewards agents for precise ball passing with varying lengths using controlled dynamics.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_pass_min_distance = 0.3  # Minimum distance for considering a pass as 'long'
        self.pass_accuracy_threshold = 0.1  # Maximum allowed distance from the teammate to consider pass as 'accurate'
        self.long_pass_reward = 1.0  # Reward for successful long pass
        self.last_ball_position = None
        self.pass_initiator = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_position = None
        self.pass_initiator = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        to_pickle['pass_initiator'] = self.pass_initiator
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('last_ball_position', None)
        self.pass_initiator = from_pickle.get('pass_initiator', None)
        return from_pickle

    def reward(self, reward):
        current_observation = self.env.unwrapped.observation()
        reward_components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}
        
        if self.last_ball_position is not None and current_observation['ball_owned_team'] == 1:
            ball_position = np.array(current_observation['ball'][:2])
            ball_transfer_distance = np.linalg.norm(ball_position - self.last_ball_position)
            
            if ball_transfer_distance >= self.long_pass_min_distance:
                for i, player_pos in enumerate(current_observation['right_team']):
                    if np.linalg.norm(player_pos - ball_position) <= self.pass_accuracy_threshold:
                        reward[0] += self.long_pass_reward
                        reward_components['long_pass_reward'][0] = self.long_pass_reward
                        break

        self.last_ball_position = np.array(current_observation['ball'][:2]) if current_observation['ball_owned_team'] != -1 else None
        return reward, reward_components

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
