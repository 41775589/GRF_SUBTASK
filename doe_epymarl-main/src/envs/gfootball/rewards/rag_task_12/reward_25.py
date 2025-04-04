import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances an agent's rewards based on its effectiveness in midfield and defense roles,
    focusing on high and long passes, dribbling under pressure, and quick transitions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward = 0.1
        self.dribble_maintain_possession_reward = 0.2
        self.transition_effectiveness_reward = 0.15

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
        components = {
            "base_score_reward": reward.copy(),
            "pass_quality_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            # Encourage high and long passes
            if o['sticky_actions'][8] == 1 and o['ball_owned_team'] == 0:
                components["pass_quality_reward"][idx] = self.pass_quality_reward

            # Reward maintaining possession under pressure
            if o['ball_owned_team'] == 0 and np.any(o['sticky_actions'][5:7]):  # Consider dribble and sprint actions
                components["dribble_reward"][idx] = self.dribble_maintain_possession_reward
            
            # Effective defensive and offensive transitions
            if o['game_mode'] in [2, 3, 4, 5] and o['ball_owned_team'] == 0:  # Transition modes
                components["transition_reward"][idx] = self.transition_effectiveness_reward

            # Aggregate the rewards
            reward[idx] += sum(components[comp][idx] for comp in components if comp != "base_score_reward")

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
