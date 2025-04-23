import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic positioning, transitioning from defense to attack,
    and enhancing team defensive resilience through player movements."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_transition_reward = 0.1
        self.defensive_positioning_reward = 0.05
        self.reset_rewards()

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.reset_rewards()
        return self.env.reset()

    def reset_rewards(self):
        self.backline_positions_reached = [False] * 2  # Track backline defense
        self.midfield_transitions_achieved = [False] * 2  # Midfield transition
        self.strikers_advanced_positions = [False] * 2  # Forward push

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "backline": self.backline_positions_reached,
            "midfield": self.midfield_transitions_achieved,
            "strikers": self.strikers_advanced_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_data = from_pickle['CheckpointRewardWrapper']
        self.backline_positions_reached = wrapper_data["backline"]
        self.midfield_transitions_achieved = wrapper_data["midfield"]
        self.strikers_advanced_positions = wrapper_data["strikers"]
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward),
            "attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            # Observations and positioning within the field
            o = observation[rew_index]
            if 'left_team' not in o:
                continue
            
            # Analyze player positions based on field locations
            for idx, pos in enumerate(o['left_team']):
                if pos[0] < -0.7:  # Deep in defensive end
                    if not self.backline_positions_reached[rew_index]:
                        components["defensive_reward"][rew_index] += self.defensive_positioning_reward
                        self.backline_positions_reached[rew_index] = True
                elif -0.3 <= pos[0] <= 0.3:  # Transition through midfield
                    if not self.midfield_transitions_achieved[rew_index]:
                        components["transition_reward"][rew_index] += self.positive_transition_reward
                        self.midfield_transitions_achieved[rew_index] = True
                elif pos[0] > 0.5:  # Advancing towards goal
                    if not self.strikers_advanced_positions[rew_index]:
                        components["attack_reward"][rew_index] += self.positive_transition_reward
                        self.strikers_advanced_positions[rew_index] = True
            
            reward[rew_index] += sum([
                components["defensive_reward"][rew_index],
                components["transition_reward"][rew_index],
                components["attack_reward"][rew_index]
            ])

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
