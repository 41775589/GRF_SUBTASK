import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward based on strategic positioning and transition in defensive game scenarios."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the entity of strategic positions on the field
        self.strategic_positions = {
            "defensive_zone": [-1.0, -0.5],  # Left half close to the goal
            "transition_zone": [-0.5, 0.0],  # Middle left
            "attacking_transition": [0.0, 0.5]  # Middle right
        }
        self.defensive_reward = 0.3  # Increased reward for staying in defensive zone
        self.transition_reward = 0.2  # Reward for operating in transition zone
        self.attack_transition_reward = 0.15  # Reward for initiating attacks from transition

        # State for managing reward zones to encourage defensive resilience
        self.is_in_defensive_zone = False
        self.is_in_transition_zone = False
        self.is_in_attack_transition_zone = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.is_in_defensive_zone = False
        self.is_in_transition_zone = False
        self.is_in_attack_transition_zone = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore state here if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_x_pos = o['left_team'][o['active']][0]  # active player's x position

            # Check and update position flags
            if self.strategic_positions['defensive_zone'][0] <= player_x_pos <= self.strategic_positions['defensive_zone'][1]:
                if not self.is_in_defensive_zone:
                    reward[rew_index] += self.defensive_reward
                    self.is_in_defensive_zone = True
                components["defensive_positioning"] = self.defensive_reward
            else:
                self.is_in_defensive_zone = False
            
            if self.strategic_positions['transition_zone'][0] <= player_x_pos <= self.strategic_positions['transition_zone'][1]:
                if not self.is_in_transition_zone:
                    reward[rew_index] += self.transition_reward
                    self.is_in_transition_zone = True
                components["transition_play"] = self.transition_reward
            else:
                self.is_in_transition_zone = False

            if self.strategic_positions['attacking_transition'][0] <= player_x_pos <= self.strategic_positions['attacking_transition'][1]:
                if not self.is_in_attack_transition_zone:
                    reward[rew_index] += self.attack_transition_reward
                    self.is_in_attack_transition_zone = True
                components["attack_initiation"] = self.attack_transition_reward
            else:
                self.is_in_attack_transition_zone = False
        
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
