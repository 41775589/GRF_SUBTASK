import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage midfielders in space creation and collaboration with strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielders_positions = {}
        self.strikers_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielders_positions = {}
        self.strikers_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_midfielders'] = self.midfielders_positions
        to_pickle['CheckpointRewardWrapper_strikers'] = self.strikers_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfielders_positions = from_pickle['CheckpointRewardWrapper_midfielders']
        self.strikers_positions = from_pickle['CheckpointRewardWrapper_strikers']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "coordinated_play_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Iterate through each agent's observation
        for idx in range(len(reward)):
            obs = observation[idx]
            player_pos = obs['left_team' if obs['active'] in obs['left_team'] else 'right_team'][obs['active']]
            player_role = obs['left_team_roles' if obs['active'] in obs['left_team'] else 'right_team_roles'][obs['active']]

            if player_role in [4, 6, 7]:  # Midfield roles: DM, LM, RM
                self.midfielders_positions[idx] = player_pos

            if player_role == 9:  # Striker roles: CF
                self.strikers_positions[idx] = player_pos

            # Calculating reward for coordinated midfield-striker operations
            for mid_idx, mid_pos in self.midfielders_positions.items():
                for str_idx, str_pos in self.strikers_positions.items():
                    distance = np.linalg.norm(np.array(mid_pos) - np.array(str_pos))
                    # Reward for keeping midfielders and strikers close for potential plays
                    if distance < 0.2:
                        components['coordinated_play_reward'][mid_idx] += 0.1
                        components['coordinated_play_reward'][str_idx] += 0.1

            reward[idx] += components['coordinated_play_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()  # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
