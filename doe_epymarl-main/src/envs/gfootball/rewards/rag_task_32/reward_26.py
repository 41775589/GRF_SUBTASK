import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for wingers focusing on crossing and sprinting."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 10  # Number of zones on the wing for crossing
        self._crossing_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_crosses = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_crosses
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_crosses = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Preparing the output according to the desired observation and reward structure
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if "right_team_roles" in o:
                winger_indices = np.where(o["right_team_roles"] == 7)[0]  # Wingers are usually role '7' for RM/LM
            else:
                continue

            for winger_index in winger_indices:
                position = o["right_team"][winger_index]
                # Check if a winger is close to the sides for crossing
                is_near_wing = position[1] > 0.3 or position[1] < -0.3
                is_in_good_position = position[0] > 0.5
                if is_near_wing and is_in_good_position and o.get("ball_owned_player") == winger_index:
                    components["crossing_reward"][rew_index] += self._crossing_reward
                    reward[rew_index] += components["crossing_reward"][rew_index] * self._num_zones

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions directly from observations if available
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
