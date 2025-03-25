import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function to focus on defensive skills and quick transitions for counter-attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define specific checkpoints based on defensive positions and transitions
        self.defensive_checkpoints = [0.6, 0.4, 0.2, 0.0]  # Represents defensive depth on the field
        self.transition_checkpoints = [0.2, 0.4, 0.6, 0.8, 1.0]  # Represents movement towards opponent's goal
        self.defensive_reward = 0.05
        self.transition_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return state

    def set_state(self, state):
        state_from_env = self.env.set_state(state)
        self.sticky_actions_counter = np.array(state_from_env['CheckpointRewardWrapper']['sticky_actions_counter'])
        return state_from_env

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for agent_id, o in enumerate(observation):
            player_pos = o['right_team'][o['active']]
            opponent_goal_pos = 1  # Assuming the right side is towards opponent goal

            # Calculate the defensive reward
            for checkpoint in self.defensive_checkpoints:
                if player_pos[0] <= checkpoint:
                    components["defensive_reward"][agent_id] += self.defensive_reward
                    break  # Only reward the farthest reached checkpoint
            
            # Calculate the transition reward
            for checkpoint in reversed(self.transition_checkpoints):  # Reverse to start checking from closest to goal
                if player_pos[0] >= checkpoint:
                    components["transition_reward"][agent_id] += self.transition_reward
                    break

            # Summing up the rewards with base rewards
            total_reward = 1 * components["base_score_reward"][agent_id] + \
                           components["defensive_reward"][agent_id] + \
                           components["transition_reward"][agent_id]
            reward[agent_id] = total_reward

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
