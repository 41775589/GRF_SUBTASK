import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for precise 'Stop-Dribble' actions under pressure,
    facilitating enhanced ball control in defensive tactics.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_reward = 0.3

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
        # The debug statement returns the observations and the reward components for tracing the modifications done here.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components.setdefault("stop_dribble_reward", [0.0] * len(reward))

            if 'sticky_actions' in o:
                # Reward applying the 'stop dribble' precisely under high-pressure scenarios
                stop_action = o['sticky_actions'][9]  # Assuming 'stop dribble' is encoded at index 9
                opponent_close = np.any([np.linalg.norm(np.array(o['left_team'][i]) - np.array(o['right_team'][0])) < 0.1 for i in range(len(o['left_team']))])
                
                if stop_action and opponent_close:
                    components["stop_dribble_reward"][rew_index] = self.stop_dribble_reward
                    reward[rew_index] += components["stop_dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
