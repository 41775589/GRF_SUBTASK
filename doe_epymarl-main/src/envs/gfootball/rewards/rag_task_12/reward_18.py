import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on the agent's performance as a midfielder/advanced defender."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds and rewards for high pass, long pass, dribble, sprint, and stop actions.
        self.high_pass_reward = 0.1
        self.long_pass_reward = 0.1
        self.dribble_reward = 0.05
        self.sprint_reward = 0.05
        self.stop_sprint_reward = 0.05
        self.possession_change_reward = -0.1

        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "long_pass_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward),
            "stop_sprint_reward": [0.0] * len(reward),
            "possession_change_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o["sticky_actions"]

            if sticky_actions[8] == 1:  # Sprint
                reward[rew_index] += self.sprint_reward
                components["sprint_reward"][rew_index] = self.sprint_reward
                
            if sticky_actions[9] == 1:  # Dribble
                reward[rew_index] += self.dribble_reward
                components["dribble_reward"][rew_index] = self.dribble_reward

            if sticky_actions[8] == 0:  # Stop Sprint
                reward[rew_index] += self.stop_sprint_reward
                components["stop_sprint_reward"][rew_index] = self.stop_sprint_reward

            # Detect passes made by checking ball possession change
            current_ball_owner = (o['ball_owned_team'], o['ball_owned_player'])
            if self.previous_ball_owner and self.previous_ball_owner != current_ball_owner:
                reward[rew_index] += self.possession_change_reward
                components["possession_change_reward"][rew_index] = self.possession_change_reward
            
            self.previous_ball_owner = current_ball_owner

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
