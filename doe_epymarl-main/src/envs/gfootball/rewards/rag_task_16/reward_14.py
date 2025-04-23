import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for specific high pass skill enhancements in Google Research Football environment.
    This encourages agents to execute high passes efficiently under the designated circumstances to approach learning
    trajectory control, power assessment, and situational application of high passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.5  # Hypothetical threshold to consider a high pass 'accurate'
        self.high_pass_reward = 0.3  # Reward for executing a precise high pass
        self.last_high_pass_pos = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_high_pass_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_high_pass_pos'] = self.last_high_pass_pos
        return state

    def set_state(self, state):
        self.last_high_pass_pos = state.get('last_high_pass_pos', None)
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 6:  # Assuming mode 6 is a high pass context
                ball_pos = o['ball']
                ball_direction = o['ball_direction']
                
                # Check if a high pass has been executed recently
                if self.last_high_pass_pos is not None:
                    distance_travelled = np.linalg.norm(self.last_high_pass_pos - ball_pos[:2])
                    # Reward if the ball travels a distance greater than a threshold
                    if distance_travelled > self.pass_accuracy_threshold:
                        components['high_pass_reward'][rew_index] = self.high_pass_reward
                        reward[rew_index] += components['high_pass_reward'][rew_index]
                
                # Update the last known high pass ball position
                self.last_high_pass_pos = ball_pos[:2]

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
