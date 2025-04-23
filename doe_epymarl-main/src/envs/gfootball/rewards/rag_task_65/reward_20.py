import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances training on scenario-based shooting and passing skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Number of regions for forward progress towards the opponent's goal
        self._num_zones = 5  
        self._zone_rewards = [0.2, 0.4, 0.6, 0.8, 1.0]  # Reward values for each zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zones_collected = np.zeros((3, self._num_zones), dtype=bool)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zones_collected.fill(False)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.zones_collected.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zones_collected = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "zone_rewards": [[0.0] * self._num_zones for _ in range(len(reward))]}

        for i, rew in enumerate(reward):
            o = observation[i]
            x_ball = o['ball'][0]
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:  # Right team has the ball
                position_index = int((x_ball + 1) / 0.4)  # Calculate the zone index from the x position
                # Only reward if zone is not yet collected and within valid bounds
                if 0 <= position_index < self._num_zones and not self.zones_collected[i, position_index]:
                    reward_increment = self._zone_rewards[position_index]
                    components["zone_rewards"][i][position_index] = reward_increment
                    reward[i] += reward_increment
                    self.zones_collected[i, position_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum([sum(sub) for sub in value])
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
