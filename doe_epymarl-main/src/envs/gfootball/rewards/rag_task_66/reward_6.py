import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage successful short passes under defensive pressure in a soccer game."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define regions on the field relevant to rewarding short passes
        self.passing_region_thresholds = [
            (-1, -0.3), (-0.5, 0), (0, 0.3), (0.5, 1)  # representative passing zones from defense to attack
        ]
        self.pass_completion_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "short_pass_reward": [0.0] * len(reward)}

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]

            # Check if the controlled player has completed a pass
            active_player_position = o['left_team'][o['designated']] if o['ball_owned_team'] == 0 else o['right_team'][o['designated']]
            if o['game_mode'] == 5:  # Assume mode 5 is a successful pass completion
                # Apply different rewards based on where the pass was completed
                for idx, (start_x, end_x) in enumerate(self.passing_region_thresholds):
                    if start_x <= active_player_position[0] <= end_x: 
                        components["short_pass_reward"][rew_index] = self.pass_completion_reward * (idx + 1)
                        break
            # Update rewards
            reward[rew_index] += components["short_pass_reward"][rew_index]

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

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
