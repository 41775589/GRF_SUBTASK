import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic transition movements, emphasizing defensive resilience."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.position_change_reward = 0.05
        self.score_diff = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.score_diff = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        components = {
            "base_score_reward": reward.copy(),
            "position_change_reward": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward agents for maintaining strategic positions (i.e., shifting from defensive to offensive positions)
            if self.previous_ball_position is not None:
                ball_distance_moved = np.linalg.norm(np.array(o['ball'][:2]) - np.array(self.previous_ball_position[:2]))

                if ball_distance_moved > 0.05:  # Significant movement
                    components["position_change_reward"][rew_index] = self.position_change_reward
                    reward[rew_index] += components["position_change_reward"][rew_index]

            # Track differential score to motivate defensive actions
            current_score_diff = o['score'][0] - o['score'][1]
            if current_score_diff != self.score_diff:
                reward_increment = (current_score_diff - self.score_diff) * 0.5
                reward[rew_index] += reward_increment
                self.score_diff = current_score_diff

            self.previous_ball_position = o['ball']

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
