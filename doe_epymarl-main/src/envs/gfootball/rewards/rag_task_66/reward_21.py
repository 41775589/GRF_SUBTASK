import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to encourage effective short passing under defensive pressure,
    focusing on ball possession continuity and supportive plays for counters.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_receivers = {}  # Tracks the position of last pass receiver for each team
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_receivers = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_receivers'] = self.pass_receivers
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_receivers = from_pickle['pass_receivers']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_success_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward) 
        }

        for i in range(len(reward)):
            obs = observation[i]
            if obs['ball_owned_team'] == -1:
                continue

            # Reward for maintaining possession
            if obs['ball_owned_team'] == 0:  # left team possessing the ball
                components['possession_reward'][i] += 0.01

                # Reward for successful pass
                if obs['game_mode'] == 5:  # Throw in indicates a pass happened
                    last_receiver = self.pass_receivers.get('left', -1)
                    if last_receiver != obs['ball_owned_player']:
                        components['pass_success_reward'][i] += 0.05
                        self.pass_receivers['left'] = obs['ball_owned_player']

            elif obs['ball_owned_team'] == 1:  # right team possessing the ball
                components['possession_reward'][i] += 0.01

                # Reward for successful pass
                if obs['game_mode'] == 5:  # Throw in indicates a pass happened
                    last_receiver = self.pass_receivers.get('right', -1)
                    if last_receiver != obs['ball_owned_player']:
                        components['pass_success_reward'][i] += 0.05
                        self.pass_receivers['right'] = obs['ball_owned_player']

            reward[i] += sum(components[k][i] for k in components)

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
            for idx, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[idx] += action
        return observation, reward, done, info
