import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds rewards focused on defensive strategies and transitions. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.teammate_proximity_reward = 0.05
        self.ball_recovery_reward = 0.1
        self.transition_efficiency_reward = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "teammate_proximity_bonus": [0.0] * len(reward),
                      "ball_recovery_bonus": [0.0] * len(reward),
                      "transition_efficiency_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Improve defensive coordination by rewarding close proximity among defenders
            if o['ball_owned_team'] == 1:
                closest_teammate_distance = np.min(np.linalg.norm(
                    o['right_team'] - o['right_team'][o['active']], axis=1))
                # Reward for being close to at least one teammate but not too close
                if 0.05 < closest_teammate_distance < 0.2:
                    components["teammate_proximity_bonus"][rew_index] = self.teammate_proximity_reward
                    reward[rew_index] += components["teammate_proximity_bonus"][rew_index]

            # Encourage rapid recovery of ball control in defense
            if o['game_mode'] in [3, 4, 5]:  # FreeKick, Corner, ThrowIn for opponent
                if o['ball_owned_team'] == 1:
                    components["ball_recovery_bonus"][rew_index] = self.ball_recovery_reward
                    reward[rew_index] += components["ball_recovery_bonus"][rew_index]

            # Encourage efficient transition by rewarding quick ball advances from defensive third to midfield
            if o['right_team'][o['active']][0] < -0.33 and o['ball'][0] > 0:
                if self.sticky_actions_counter[4] or self.sticky_actions_counter[3]:  # Right or TopRight actions
                    components["transition_efficiency_bonus"][rew_index] = self.transition_efficiency_reward
                    reward[rew_index] += components["transition_efficiency_bonus"][rew_index]

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
