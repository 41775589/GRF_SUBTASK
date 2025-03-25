import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards based on defensive positioning and quick transitions for counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defensive checkpoint setup
        self.defense_zones = 5
        self.defense_rewards = np.zeros(self.defense_zones, dtype=float)
        self.defense_reward_multiplier = 0.2
        # Transition rewards setup
        self.ball_possession_reward = 0.1
        self.last_ball_position = None
        self.transition_distances = []
        self.transition_reward_multiplier = 0.3

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defense_rewards.fill(0)
        self.last_ball_position = None
        self.transition_distances.clear()
        return observation

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'defense_rewards': self.defense_rewards,
            'last_ball_position': self.last_ball_position,
            'transition_distances': self.transition_distances
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = saved_state['sticky_actions_counter']
        self.defense_rewards = saved_state['defense_rewards']
        self.last_ball_position = saved_state['last_ball_position']
        self.transition_distances = saved_state['transition_distances']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        for player_obs in observation:
            # Defensive reward based on distance to own goal
            own_goal_position = -1 if player_obs['ball_owned_team'] == 1 else 1
            current_ball_position = player_obs['ball'][0]
            distance_to_goal = own_goal_position - current_ball_position
            zone_index = int((distance_to_goal + 1) * (self.defense_zones - 1) / 2)
            if self.defense_rewards[zone_index] == 0:
                self.defense_rewards[zone_index] = self.defense_reward_multiplier
                reward += self.defense_reward_multiplier

            # Reward for quick transition focusing on ball movement speed
            if self.last_ball_position is not None:
                distance = np.linalg.norm(player_obs['ball'] - self.last_ball_position)
                self.transition_distances.append(distance)
            if len(self.transition_distances) >= 2:
                speed_increase = self.transition_distances[-1] - self.transition_distances[-2]
                if speed_increase > 0:
                    transition_reward = self.ball_possession_reward * speed_increase
                    reward += transition_reward
            self.last_ball_position = player_obs['ball']

        components["defense_reward"] = self.defense_rewards.tolist()
        components["transition_distance"] = self.transition_distances[-1] if self.transition_distances else 0
        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mod_reward, components = self.reward(reward)
        info["final_reward"] = mod_reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, mod_reward, done, info
