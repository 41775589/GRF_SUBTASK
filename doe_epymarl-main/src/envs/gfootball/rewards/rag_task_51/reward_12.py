import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on specialized goalkeeper training rewards. Rewards include shot-stopping,
    quick reflexes, and initiating counter-attacks with accurate passes. The task's objective is to
    enhance the goalkeeper's capabilities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stopped_counter = 0
        self.passes_initiated = 0
        self.reflexes_reward_multiplier = 0.2
        self.shot_stopping_reward = 1.0
        self.counter_attack_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stopped_counter = 0
        self.passes_initiated = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shot_stopped_counter': self.shot_stopped_counter,
            'passes_initiated': self.passes_initiated
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_stopped_counter = from_pickle['CheckpointRewardWrapper']['shot_stopped_counter']
        self.passes_initiated = from_pickle['CheckpointRewardWrapper']['passes_initiated']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "reflex_reward": [0.0] * len(reward),
                      "shot_stopping_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 6:  # Penalty mode
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['designated']:
                    self.shot_stopped_counter += 1
                    components["shot_stopping_reward"][rew_index] = self.shot_stopping_reward

            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Defensive kicks as reflexes or good clearances.
                if np.linalg.norm(o['ball_direction'][:2]) > 0.02:
                    components["reflex_reward"][rew_index] += self.reflexes_reward_multiplier

                # Reward for initiating counter-attacks with accurate passes.
                if o['game_mode'] in [2, 5]:  # From goal kick or throw-in modes
                    self.passes_initiated += 1
                    components["counter_attack_reward"][rew_index] = self.counter_attack_reward

        # Calculate new reward summing components
        reward = [components["base_score_reward"][i] +
                  components["reflex_reward"][i] +
                  components["shot_stopping_reward"][i] +
                  components["counter_attack_reward"][i] for i in range(len(reward))]

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
