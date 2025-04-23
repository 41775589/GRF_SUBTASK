import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides dense rewards based on defensive play and counterattacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_zones_covered = 8
        self._defense_reward = 0.1
        self._counterattack_reward = 0.2
        self._zones_reward_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._zones_reward_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._zones_reward_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._zones_reward_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward),
                      "counterattack_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive reward calculation
            if o['game_mode'] in [2, 3, 4, 6]:  # FreeKick, GoalKick, Corner, Penalty
                zone_idx = self.zone_index(o['ball'])
                if zone_idx not in self._zones_reward_collected:
                    components["defense_reward"][rew_index] += self._defense_reward
                    reward[rew_index] += components["defense_reward"][rew_index]
                    self._zones_reward_collected[zone_idx] = True

            # Positioning for counterattack
            if (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']) or \
                    (o['game_mode'] == 0 and o['ball'][0] > 0.5):  # Ball in opponent's half
                if 'countering' not in self._zones_reward_collected:
                    components["counterattack_reward"][rew_index] += self._counterattack_reward
                    reward[rew_index] += components["counterattack_reward"][rew_index]
                    self._zones_reward_collected['countering'] = True

        return reward, components

    def zone_index(self, ball_position):
        """ Calculate the zone index based on ball position """
        x, y, z = ball_position
        column = int((x + 1) / 0.25)  # 8 vertical partitions of the field, from -1 to 1
        row = int((y + 0.42) / 0.105)  # 8 horizontal partitions of the field, from -0.42 to 0.42
        return row * 4 + column

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active_action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active_action
        
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
            
        return observation, reward, done, info
