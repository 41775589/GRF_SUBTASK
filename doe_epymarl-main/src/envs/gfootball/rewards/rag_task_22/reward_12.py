import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a sprint-focused defensive positioning reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define parameters for the sprint reward strategy
        self._sprint_reward_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            active_player_index = o['active']
            
            # Check if active player is sprinting and contributing to defensive positioning
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Index 8 is 'action_sprint'
                # Check if player is moving to a defensive position
                player_pos = o['left_team'][active_player_index] if o['side'] == 0 else o['right_team'][active_player_index]
                own_goal_pos = -1 if o['side'] == 0 else 1  # 1 is the right goal, -1 is the left goal
                
                # Reward for moving towards own goal for defense
                if (player_pos[0] * own_goal_pos < 0):  # Moving towards own goal
                    components["sprint_reward"][idx] = self._sprint_reward_factor
                    reward[idx] += components["sprint_reward"][idx]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
