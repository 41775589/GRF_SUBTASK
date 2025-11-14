import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes defensive collaboration and and tactical positioning to improve goal prevention abilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Adds penalty zone defense collaboration rewards."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Reward agents for defending collaboratively in the penalty zone
        for idx, obs in enumerate(observation):
            player_pos = obs['left_team'][obs['active']]
            opponent_positions = obs['right_team']

            # Check if the controlled player is in own penalty area
            in_penalty_area = player_pos[0] < -0.6
            distance_to_closest_opponent = np.min(np.linalg.norm(opponent_positions - player_pos, axis=1))

            if in_penalty_area:
                # Encourage proximity to the opponents to disrupt their plays
                if distance_to_closest_opponent < 0.1:
                    components["defensive_positioning_reward"][idx] = 0.2 * (0.1 - distance_to_closest_opponent)
                    reward[idx] += components["defensive_positioning_reward"][idx]
                    
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
