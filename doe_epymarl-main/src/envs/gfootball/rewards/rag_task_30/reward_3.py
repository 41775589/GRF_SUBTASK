import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions and transitioning to counterattack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        base_score_reward = np.array(reward.copy())

        # Get current observations from the environment
        observation = self.env.unwrapped.observation()
        transition_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {"base_score_reward": base_score_reward, "transition_reward": transition_reward}

        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Encourage maintaining possession and passing in defensive half
            if obs['ball_owned_team'] == 0 and (obs['ball'][0] <= 0):  # player's own half
                transition_reward[rew_index] += 0.05
            
            # Additional reward for successful pass start from defense going forward towards midfield
            if obs['game_mode'] == 3:  # FreeKick or similar advantageous restart
                transition_reward[rew_index] += 0.1

            reward[rew_index] += transition_reward[rew_index]

        return reward, {"base_score_reward": base_score_reward, "transition_reward": transition_reward}

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
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
