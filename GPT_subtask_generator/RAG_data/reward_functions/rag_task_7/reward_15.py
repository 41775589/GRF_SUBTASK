import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering defensive maneuvers, specifically sliding tackles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_threshold = 0.7
        self._ball_owner_penalty = -0.1
        self._successful_tackle_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check for sticky actions related to defensive actions (here index 8 is hypothetical for sliding tackle)
            if o['sticky_actions'][8] == 1:
                # Check ball ownership status
                distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']]) if o['ball_owned_team'] == 1 \
                                    else np.linalg.norm(o['ball'][:2] - o['right_team'][o['active']])
                if distance_to_ball < self._tackle_threshold:

                    # Reward for successful tackle: close to ball and tackle action initiated
                    components["tackle_reward"][rew_index] = self._successful_tackle_reward
                    reward[rew_index] += components["tackle_reward"][rew_index]
                else:
                    # Penalty if tackle action is made far from the ball
                    components["tackle_reward"][rew_index] = self._ball_owner_penalty
                    reward[rew_index] += components["tackle_reward"][rew_index]

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
