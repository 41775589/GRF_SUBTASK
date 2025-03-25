import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This wrapper augments the original reward with rewards that encourage players to develop
    advanced dribbling techniques with Sprint use, aimed at breaking through tight defensive lines.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._num_sprint_zones = 5
        self._sprint_reward = 0.2
        self._dribble_zones_count = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._dribble_zones_count = {}
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._dribble_zones_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._dribble_zones_count = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            zone = int((o['ball'][0] + 1) * self._num_sprint_zones / 2)  # Normalize and scale position
            
            # Check if the active player is controlling the ball and sprinting
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and o['sticky_actions'][8]:
                # Reward for sprinting with the ball, more reward if closer to opponent's goal
                if zone not in self._dribble_zones_count:
                    self._dribble_zones_count[zone] = 1
                    reward_multiplier = (self._num_sprint_zones - zone) / self._num_sprint_zones
                    components["sprint_dribble_reward"][rew_index] = self._sprint_reward * reward_multiplier
                    reward[rew_index] += 1.5 * components["sprint_dribble_reward"][rew_index]

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
