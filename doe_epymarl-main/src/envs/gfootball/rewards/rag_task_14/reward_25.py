import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes the sweeper performance in defensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_clearences = {}
        self._tackle_successful = {}
        self._recoveries = {}
        self.clearence_reward = 0.2
        self.tackle_reward = 0.3
        self.recovery_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_clearences = {}
        self._tackle_successful = {}
        self._recoveries = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'clearences': self._successful_clearences,
            'tackles': self._tackle_successful,
            'recoveries': self._recoveries
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_dict = from_pickle['CheckpointRewardWrapper']
        self._successful_clearences = state_dict['clearences']
        self._tackle_successful = state_dict['tackles']
        self._recoveries = state_dict['recoveries']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearences": [0.0] * len(reward),
            "tackles": [0.0] * len(reward),
            "recoveries": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Determine if the action of this agent was a successful clearence
            if o['ball_owned_team'] == 0 and o['game_mode'] in [2, 4]:  # Considering game modes like goal_kick, corner
                if rew_index not in self._successful_clearences:
                    self._successful_clearences[rew_index] = True
                    components["clearences"][rew_index] = self.clearence_reward
                    reward[rew_index] += components["clearences"][rew_index]

            # Checking for successful tackles
            if o['ball_owned_team'] == 0 and 'tackle' in o['sticky_actions']:
                if rew_index not in self._tackle_successful:
                    self._tackle_successful[rew_index] = True
                    components["tackles"][rew_index] = self.tackle_reward
                    reward[rew_index] += components["tackles"][rew_index]

            # Checking for positional recovery
            # Assuming position recovery can be seen from fast repositioning in defensive mode
            if o['ball_owned_team'] != 0 and np.linalg.norm(o['left_team_direction'][o['active']]) > 0.01:  # high speed
                if rew_index not in self._recoveries:
                    self._recoveries[rew_index] = True
                    components["recoveries"][rew_index] = self.recovery_reward
                    reward[rew_index] += components["recoveries"][rew_index]

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
