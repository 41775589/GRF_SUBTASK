import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defense specialization and counterattack positioning reward in a football task."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Update reward based on defense specialization and counterattack capabilities
        for rew_index, obs in enumerate(observation):
            norm_position_diff = np.linalg.norm(obs['right_team'] - obs['left_team'], axis=1)
            near_opponent = np.any(norm_position_diff < 0.1)  # Close proximity to one opponent
            control_ball = (obs['ball_owned_team'] == 1 and 
                            obs['ball_owned_player'] in obs['right_team'][:, 0])
            effective_defense = near_opponent and not control_ball
            counterattack_prepared = near_opponent and control_ball

            if effective_defense:
                reward[rew_index] += 0.5
                if 'defensive_skill' not in components:
                    components['defensive_skill'] = [0.0] * len(reward)
                components['defensive_skill'][rew_index] = 0.5

            if counterattack_prepared:
                reward[rew_index] += 0.5
                if 'counterattack_positioning' not in components:
                    components['counterattack_positioning'] = [0.0] * len(reward)
                components['counterattack_positioning'][rew_index] = 0.5
        
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
