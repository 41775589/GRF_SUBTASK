import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances defensive training focused on goalkeeper and defenders.

    The reward increases upon successful tackling, ball retention by defenders,
    shot stopping, and goalie's play initiation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.tackle_reward = 0.1
        self.retention_reward = 0.1
        self.goalie_stop_reward = 0.2
        self.play_initiation_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle['CheckpointRewardWrapper']

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, {}

        for rew_index, o in enumerate(observation):
            # Specific rewards to roles: Defenders (2=RB, 3=LB, 4=CB), Goalkeepers (0)
            player_role = o['left_team_roles'][o['active']] if o['ball_owned_team'] == 0 else \
                          o['right_team_roles'][o['active']]

            ball_owned = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']
            
            # Rewarding defenders for ball retention
            if player_role in [2, 3, 4] and ball_owned:
                components['defensive_reward'][rew_index] += self.retention_reward

            # Rewarding goalkeeper for shot stops and play initiations
            if player_role == 0:
                if self.is_goalkeeper_action_successful(o):
                    components['defensive_reward'][rew_index] += self.goalie_stop_reward

            # Evaluating reward components
            total_defensive_reward = components['defensive_reward'][rew_index]
            # Combine rewards with a multiplier for defensive importance
            reward[rew_index] += 2 * total_defensive_reward

        return reward, components

    def is_goalkeeper_action_successful(self, observation):
        # Placeholder logic for determining successful goalkeeper actions
        # This should ideally check for actions like shot stopping or clearances
        return np.random.random() > 0.5  # Randomly simulate success for demo

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
