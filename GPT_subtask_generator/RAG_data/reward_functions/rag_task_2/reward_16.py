import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on defensive teamwork and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_formation_checkpoints = {}
        self.reward_for_strategic_positioning = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_formation_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.player_formation_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.player_formation_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
 
            # Calculate strategic positioning reward for controlled player
            if o['active'] > -1 and o['left_team_active'][o['active']]:
                # Define defensive zones on the left side
                player_x = o['left_team'][o['active']][0]
                player_y = o['left_team'][o['active']][1]

                # Reward for players strategically located in defensive zones
                if player_x < 0 and -0.25 < player_y < 0.25:
                    zone_key = (o['active'], 'defense_center')
                    components['defense_positioning_reward'][rew_index] += (1 if 
                    self.player_formation_checkpoints.get(zone_key) is None else 0) * self.reward_for_strategic_positioning
                    self.player_formation_checkpoints[zone_key] = True

            # Incorporate new rewards into existing reward structure
            reward[rew_index] += components['defense_positioning_reward'][rew_index]
        
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
