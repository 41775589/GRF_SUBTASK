import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for transitioning skills from defense to attack like Passing and Dribbling."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_success_reward = 0.2
        self.dribble_reward = 0.1

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
        """
        Reward agents for successful passes and control handling under pressure:
        - Adds reward for successful short or long passes which indicates good transitions.
        - Adds reward for dribbling actions maintaining ball possession under defensive pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']
            
            if o['ball_owned_team'] == o['active']:
                # Short or long pass actions (indices 0 and 1 in sticky_actions)
                if sticky_actions[0] == 1 or sticky_actions[1] == 1:
                    components["pass_reward"][rew_index] = self.pass_success_reward
                    reward[rew_index] += components["pass_reward"][rew_index]
                
                # Dribble action (index 9 in sticky_actions)
                if sticky_actions[9] == 1:
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
