import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards refined passing and possession skills under pressure
    in tight game situations. It specifically rewards maintaining ball control,
    successful short, high, and long passes under defensive pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self._num_checkpoints = 5  # Number of zones to monitor tight game situations
        self._checkpoint_reward = 0.1  # Reward increment for skillful ball management
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_possession_zones = {}  # Tracks ball possession in high pressure zones

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_possession_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._ball_possession_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._ball_possession_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "skillful_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            player_role = o['left_team_roles'][o['active']] if o['ball_owned_team'] == 0 else o['right_team_roles'][o['active']]
            ball = o['ball']

            # Analyze the current situation of ball control, and attempt passes under pressure
            if (o['ball_owned_team'] == o['active'] and player_role in [2, 5, 8]):  # Midfielders and attackers
                distance_from_goal = np.linalg.norm(ball[:2] - np.array([1, 0])) if player_pos[0] > 0 else np.linalg.norm(ball[:2] - np.array([-1, 0]))
                zone_idx = int((distance_from_goal / 2) * self._num_checkpoints)

                if zone_idx < self._num_checkpoints:
                    collected = self._ball_possession_zones.get(rew_index, [])
                    if zone_idx not in collected:
                        # Reward for maintaining possession or making a successful pass in a tight zone
                        components['skillful_pass_reward'][rew_index] = self._checkpoint_reward
                        reward[rew_index] += components['skillful_pass_reward'][rew_index]
                        collected.append(zone_idx)
                        self._ball_possession_zones[rew_index] = collected

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
