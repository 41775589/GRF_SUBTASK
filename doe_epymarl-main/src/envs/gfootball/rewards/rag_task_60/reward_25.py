import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for precise defensive stops."""

    def __init__(self, env):
        super().__init__(env)
        # Sticky actions tracker for all agents (10 actions possible)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for defensive behavior rewards
        self.stop_threshold = 0.1  # Distance within which a stop must occur near an opponent with the ball
        self.immediate_stop_reward = 1.0  # Reward for stopping immediately near a ball-carrying opponent

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        additional_rewards = np.zeros(len(reward))
        components = {"base_score_reward": reward.copy(), "immediate_stop_reward": additional_rewards.tolist()}
        
        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            # Check if opponent has ball control
            if ('ball_owned_team' in o and o['ball_owned_team'] == 1) or ('ball_owned_team' in o and o['ball_owned_team'] == 0 and o['ball_owned_player'] != idx):
                ball_pos = np.array(o['ball'][:2])  # We only consider x, y coordinates
                agent_pos = np.array(o['right_team'][idx] if o['ball_owned_team'] == 0 else o['left_team'][idx])
                # Calculate Euclidean distance between agent and ball-carrier
                distance_to_ball = np.linalg.norm(ball_pos - agent_pos)
                # Check if the agent has stopped
                if np.any(o['sticky_actions'][[0, 1, 2, 3, 4, 5, 6, 7]] == 0):  # Movement related sticky actions
                    # Reward for stopping near the ball-carrier
                    if distance_to_ball <= self.stop_threshold:
                        additional_rewards[idx] = self.immediate_stop_reward + reward[idx]
                        components['immediate_stop_reward'][idx] += self.immediate_stop_reward

        return reward + additional_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_present in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_present
                info[f"sticky_actions_{i}"] = action_present
        return observation, reward, done, info
