import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized midfield dynamics reward focusing on role-specific contributions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_interactions_rewards = 0.05
        self.defensive_rewards = 0.1
        self.forward_movement_rewards = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_checkpoint_wrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_interaction_reward": [0.0] * len(reward),
                      "defensive_reward": [0.0] * len(reward),
                      "forward_movement_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            midfield_roles = [4, 5, 6]   # DM, CM, LM position roles indices
            midfield_activity = False
            for role_index, role in enumerate(o['left_team_roles']):
                if role in midfield_roles and o['left_team_active'][role_index] and np.linalg.norm(o['left_team_direction'][role_index]) > 0:
                    midfield_activity = True
                    break
            
            if midfield_activity:
                reward[rew_index] += self.midfield_interactions_rewards
                components["midfield_interaction_reward"][rew_index] += self.midfield_interactions_rewards

            if o['ball_owned_team'] == 0:  # If left team (controlled) owns the ball
                if o['ball_owned_player'] >= 0 and o['left_team_roles'][o['ball_owned_player']] in midfield_roles:
                    reward[rew_index] += self.forward_movement_rewards
                    components["forward_movement_reward"][rew_index] += self.forward_movement_rewards

            if o['ball'][0] > 0 and o['ball'][1] > 0:  # Ball in forward quadrant
                reward[rew_index] += self.defensive_rewards
                components["defensive_reward"][rew_index] += self.defensive_rewards

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
