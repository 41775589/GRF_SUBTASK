import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on quick attacking maneuvers and game phase adaptation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the reward for advancing quickly towards the opponent's goal and game phase changes
        self.advance_reward = 0.2
        self.quick_attack_bonus = 0.3
        self.game_phase_change_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": [reward[0]], "advance_reward": [0.0], "quick_attack_bonus": [0.0], "game_phase_change_bonus": [0.0]}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            # Check if the ball is moving quickly towards opponents' goal
            if o['ball'][0] > 0 and o['ball_owned_team'] == 0:  # Assuming 0 is the team of agent
                components["advance_reward"][i] = self.advance_reward
                reward[i] += components["advance_reward"][i]

            # Detect quick attacks (large movement of the ball towards the opponent's goal)
            if o['ball_direction'][0] > 0.1:  # Fast movement towards the opposing goal
                components["quick_attack_bonus"][i] = self.quick_attack_bonus
                reward[i] += components["quick_attack_bonus"][i]

            # Detect changes in the game phase and reward strategic adaptations
            if o['game_mode'] != 0 and self.env.previous_game_mode == 0:  # From normal game to any other state
                components["game_phase_change_bonus"][i] = self.game_phase_change_bonus
                reward[i] += components["game_phase_change_bonus"][i]
        
        self.env.previous_game_mode = o['game_mode']        

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
