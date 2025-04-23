import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards effectively performing standing tackles and regaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_reward = 0.5
        self.penalty_reward = -0.2  # Penalty for failed tackle or foul

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_success_reward": [0.0] * len(reward),
                      "penalty_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            game_mode_normal = o['game_mode'] == 0
            is_tackling = any(o['sticky_actions'][3:8])  # assumes tackle actions are 3-7
            tackle_successful = False

            # Reward successful tackles during normal play
            if game_mode_normal and is_tackling:
                if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                    # Reward for successfully regaining possession
                    tackle_successful = True
                    components["tackle_success_reward"][rew_index] = self.tackle_success_reward
                elif o['ball_owned_team'] != 0:
                    # Penalize for tackles that result in fouls or losing possession
                    components["penalty_reward"][rew_index] = self.penalty_reward
            
            # Combine reward components
            total_reward = components["base_score_reward"][rew_index] + \
                           components["tackle_success_reward"][rew_index] + \
                           components["penalty_reward"][rew_index]
            reward[rew_index] = total_reward
                    
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
