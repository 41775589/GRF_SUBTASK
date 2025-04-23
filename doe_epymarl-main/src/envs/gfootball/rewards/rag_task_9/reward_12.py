import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for teaching offensive skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.1
        self.shot_reward = 0.2
        self.dribble_reward = 0.05
        self.sprint_reward = 0.03

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
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check if the ball is owned by the active team.
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Check for pass action activation
                if o['sticky_actions'][0] or o['sticky_actions'][1]:  # Short Pass or Long Pass
                    reward[rew_index] += self.pass_reward
                    components["pass_reward"][rew_index] = self.pass_reward

                # Check for shot action
                if o['sticky_actions'][9]:  # Shot
                    reward[rew_index] += self.shot_reward
                    components["shot_reward"][rew_index] = self.shot_reward

                # Check for dribble action activation
                if o['sticky_actions'][8]:  # Dribble
                    reward[rew_index] += self.dribble_reward
                    components["dribble_reward"][rew_index] = self.dribble_reward

                # Check for sprint action activation
                if o['sticky_actions'][7]:  # Sprint
                    reward[rew_index] += self.sprint_reward
                    components["sprint_reward"][rew_index] = self.sprint_reward

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
