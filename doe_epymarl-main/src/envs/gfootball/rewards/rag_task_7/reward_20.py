import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles."""
    
    def __init__(self, env):
        super().__init__(env)
        self._tackle_success_count = {}
        self._tackle_attempt_count = {}
        self._tackle_success_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success_count = {}
        self._tackle_attempt_count = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_success_count'] = self._tackle_success_count
        to_pickle['tackle_attempt_count'] = self._tackle_attempt_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._tackle_success_count = from_pickle.get('tackle_success_count', {})
        self._tackle_attempt_count = from_pickle.get('tackle_attempt_count', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            tackle_success = o.get('sticky_actions', [0] * 10)[9]  # assuming index 9 is for tackling
            ball_owned = o.get('ball_owned_team', -1)
            game_mode = o.get('game_mode', 0)

            # Counting tackle attempts and successes
            if tackle_success and game_mode == 0:  # Only count tackles during normal game play
                self._tackle_attempt_count[rew_index] = self._tackle_attempt_count.get(rew_index, 0) + 1
                if ball_owned == 1:  # Assuming '1' indicates ownership by opponent
                    self._tackle_success_count[rew_index] = self._tackle_success_count.get(rew_index, 0) + 1
                    reward[rew_index] += self._tackle_success_reward  # Reward per successful tackle
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rewards, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, rewards, done, info
