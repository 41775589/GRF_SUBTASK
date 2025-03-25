import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that implements a reward function focused on precise defense and strategic midfield management."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.controlled_midfield = np.zeros(10, dtype=int)  # Count midfield control events
        self.successful_defenses = np.zeros(10, dtype=int)  # Count successful defenses

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.controlled_midfield.fill(0)
        self.successful_defenses.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['controlled_midfield'] = self.controlled_midfield
        to_pickle['successful_defenses'] = self.successful_defenses
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.controlled_midfield = from_pickle['controlled_midfield']
        self.successful_defenses = from_pickle['successful_defenses']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control": [0.0] * len(reward),
                      "defense_success": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            ball_position_x = o['ball'][0]
            is_midfield = -0.2 < ball_position_x < 0.2

            if is_midfield and o['ball_owned_team'] == o['left_team']:
                self.controlled_midfield[i] += 1
                components["midfield_control"][i] = 0.5  # This should be tuned

            if o['game_mode'] in [2, 3] and o['ball_owned_team'] == 0:  # Considering own free-kick as a successful defense
                self.successful_defenses[i] += 1
                components["defense_success"][i] = 0.3  # This should be tuned
            
            reward[i] += components["midfield_control"][i] + components["defense_success"][i]

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
