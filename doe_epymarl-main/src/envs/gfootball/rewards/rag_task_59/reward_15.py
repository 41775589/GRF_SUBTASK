import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for goalkeeper coordination and ball clearing strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.clearing_rewards = {}
        self.goalkeeper_position_threshold = 0.2  # goalkeeper must be close to the goal
        self.clearing_distance_threshold = 0.5  # distance the ball must be cleared towards a midfielder or forward

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_rewards = {}
        self.clearing_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['GoalkeeperRewards'] = self.goalkeeper_rewards
        to_pickle['ClearingRewards'] = self.clearing_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_rewards = from_pickle['GoalkeeperRewards']
        self.clearing_rewards = from_pickle['ClearingRewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        goalkeeper_reward = [0.0] * len(reward)
        clearing_reward = [0.0] * len(reward)
        components = {
            "base_score_reward": base_score_reward,
            "goalkeeper_reward": goalkeeper_reward,
            "clearing_reward": clearing_reward
        }

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            o = observation[idx]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'active' in o:
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball']
                distance_to_goal = abs(player_pos[0] + 1)
                
                if o['left_team_roles'][o['active']] == 0:  # Goalkeeper role
                    if distance_to_goal < self.goalkeeper_position_threshold:
                        goalkeeper_reward[idx] = 0.1  # Reward for being in position
                        self.goalkeeper_rewards[idx] = True

                if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    # Check if the ball is being passed or cleared towards attacking players
                    if np.linalg.norm(ball_pos - player_pos) > self.clearing_distance_threshold:
                        clearing_reward[idx] = 0.05  # Reward for clearing the ball
                        self.clearing_rewards[idx] = True

            # Combine rewards with default match rewards
            reward[idx] += goalkeeper_reward[idx] + clearing_reward[idx]

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
