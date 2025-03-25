import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense, skill-based reward for offensive football skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_components = {
            "shot": 0.5,
            "long_pass": 0.3,
            "short_pass": 0.2,
            "dribble": 0.1,
            "sprint": 0.05
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.sticky_actions_counter = from_picle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        for key in self.reward_components.keys():
            components[key + "_reward"] = [0.0] * len(reward)

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            active_actions = o.get('sticky_actions', [])
            for idx, action_active in enumerate(active_actions):
                if action_active == 1 and idx < len(self.sticky_actions_counter):
                    action_name = self._action_name_from_index(idx)
                    if action_name in self.reward_components:
                        components[action_name + "_reward"][rew_index] += self.reward_components[action_name]

                        # Increase base reward using the skill's specific component reward
                        reward[rew_index] += components[action_name + "_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info

    def _action_name_from_index(self, index):
        # Maps sticky action indices to their respective names based on FootballEnv action settings
        action_map = {
            0: "left",
            1: "top_left",
            2: "top",
            3: "top_right",
            4: "right",
            5: "bottom_right",
            6: "bottom",
            7: "bottom_left",
            8: "sprint",
            9: "dribble"
        }
        return action_map.get(index, "")
