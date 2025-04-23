import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add rewards based on advanced ball control and passing under pressure."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_accuracy_threshold = 0.8  # Define a threshold accuracy for good passes
        self.previous_ball_owner = -1
        self.previous_ball_team = -1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_for_pass = 0.2
        self.reward_for_advanced_control = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = -1
        self.previous_ball_team = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_owner': self.previous_ball_owner,
            'previous_ball_team': self.previous_ball_team
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        restored = from_pickle['CheckpointRewardWrapper']
        self.previous_ball_owner = restored['previous_ball_owner']
        self.previous_ball_team = restored['previous_ball_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_owner = o['ball_owned_player']
            current_ball_team = o['ball_owned_team']
            # Reward for maintaining ball under pressure
            if current_ball_team == 0 and self.previous_ball_team == 0:
                if current_ball_owner == o['active'] and self.previous_ball_owner != current_ball_owner:
                    components["control_reward"][rew_index] = self.reward_for_advanced_control
                    reward[rew_index] += components["control_reward"][rew_index]

            # Reward for successful passes
            if current_ball_team == 0 and self.previous_ball_team == 0:
                if current_ball_owner != self.previous_ball_owner and current_ball_owner == o['active']:
                    components["pass_reward"][rew_index] = self.reward_for_pass
                    reward[rew_index] += components["pass_reward"][rew_index]

            self.previous_ball_owner = current_ball_owner
            self.previous_ball_team = current_ball_team

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
