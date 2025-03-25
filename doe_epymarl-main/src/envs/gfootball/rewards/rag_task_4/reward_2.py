import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for advanced dribbling and sprint usage in tight defense scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_usage = 0.1  # Reward increment for using sprint in offensive position
        self.dribble_proximity_reward = 0.2  # Extra reward for dribbling close to opponents
        self.control_zone_threshold = 0.3  # Distance threshold to consider 'close' to an opponent

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward),
                      "dribble_proximity_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for using sprint during offensive plays
            if o['sticky_actions'][8]:  # Index 8 is the action_sprint
                components['sprint_reward'][rew_index] = self.sprint_usage
                reward[rew_index] += components['sprint_reward'][rew_index]
                self.sticky_actions_counter[8] += 1

            # Calculate proximity to any opponent
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            opponent_team = 'right_team' if o['ball_owned_team'] == 0 else 'left_team'
            for opponent_pos in o[opponent_team]:
                distance = np.linalg.norm(player_pos - opponent_pos)
                if distance < self.control_zone_threshold:
                    components["dribble_proximity_reward"][rew_index] = self.dribble_proximity_reward
                    reward[rew_index] += components["dribble_proximity_reward"][rew_index]
    
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
