import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for effective defensive maneuvers, 
    particularly focusing on the timing and precision of sliding tackles.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success_reward = 1.0
        self._tackle_attempt_penalty = -0.1
        self._close_call_penalty = -0.5

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": np.array(reward).copy(),
            "tackle_success_reward": np.zeros(len(reward), dtype=float),
            "tackle_attempt_penalty": np.zeros(len(reward), dtype=float),
            "close_call_penalty": np.zeros(len(reward), dtype=float)
        }
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if 'sticky_actions' in o:
                # Check if sliding tackle action is active
                if o['sticky_actions'][9]:  # Assuming sliding tackle is action index 9
                    if 'game_mode' in o and o['game_mode'] == 3:  # game_mode 3 for free kick (indicating foul/tackle)
                        components["tackle_success_reward"][i] = self._tackle_success_reward
                    else:
                        components["tackle_attempt_penalty"][i] = self._tackle_attempt_penalty
                # Check for ball possession status
                if o['ball_owned_team'] != 0 and np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2]) < 0.1:
                    components["close_call_penalty"][i] = self._close_call_penalty

        for key, value in components.items():
            reward += value
            
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
