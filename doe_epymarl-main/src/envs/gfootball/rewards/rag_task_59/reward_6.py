import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for adding sophisticated rewards specialized in goalkeeper coordination strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.goalkeeper_enhancement_reward = 0.5
        self.clearance_success_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['previous_ball_position'] = self.previous_ball_position
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        current_obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_enhancement_reward": np.zeros_like(reward),
            "clearance_success_reward": np.zeros_like(reward)
        }

        for rew_index, obs in enumerate(current_obs):
            if obs['active'] == obs['left_team_roles'][0]:  # assuming 0 is the goalkeeper role
                if obs['game_mode'] == 3:  # free kick defense mode
                    components['goalkeeper_enhancement_reward'][rew_index] += self.goalkeeper_enhancement_reward

            if self.previous_ball_position is not None:
                # Check for successful clearances by comparing distances
                previous_dist = np.linalg.norm(self.previous_ball_position[:2] - [0, 0])
                current_dist = np.linalg.norm(obs['ball'][:2] - [0, 0])
                if current_dist > previous_dist:
                    components['clearance_success_reward'][rew_index] += self.clearance_success_reward

            # Update the reward for this player
            reward[rew_index] += components['goalkeeper_enhancement_reward'][rew_index]
            reward[rew_index] += components['clearance_success_reward'][rew_index]

        self.previous_ball_position = obs['ball']  # Update the ball's previous position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Provide reward breakdown in the info for debugging/monitoring purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky action counters for analysis
        current_obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in current_obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
