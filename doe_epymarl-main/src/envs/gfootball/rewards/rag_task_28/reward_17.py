import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances dribbling skills in 1v1 situations against the goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        # Counter for each agent to avoid rewarding the same behavior too often
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards tuning parameters
        self.dribble_reward = 0.2
        self.change_direction_reward = 0.15
        self.close_control_reward = 0.1
        self.goalkeeper_proximity_threshold = 0.2  # Close enough to the goalkeeper
        self.ball_control_proximity_threshold = 0.1  # Close control of the ball

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribbling_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if the active player has the ball and is close to the goalkeeper
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                distance_to_goalkeeper = np.linalg.norm(o['right_team'][0] - o['ball'][:2])
                if distance_to_goalkeeper < self.goalkeeper_proximity_threshold:
                    components["dribbling_reward"][rew_index] += self.dribble_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

                    # Reward for changing direction rapidly implying feinting/dodging
                    if np.any(np.abs(o['ball_direction'][:2]) > 0.01):
                        components["dribbling_reward"][rew_index] += self.change_direction_reward
                        reward[rew_index] += self.change_direction_reward

                    # Reward for keeping the ball in close control
                    if np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']]) < self.ball_control_proximity_threshold:
                        components["dribbling_reward"][rew_index] += self.close_control_reward
                        reward[rew_index] += self.close_control_reward

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
