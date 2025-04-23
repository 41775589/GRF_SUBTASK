import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for focusing on offensive strategies in midfield and striker coordination."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Store checkpoints based on certain key positions in the offensive half
        self.midfield_checkpoints = []
        self.offensive_checkpoints = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define new checkpoints for each episode
        self.midfield_checkpoints = [False] * 5  # Simulating midfield checkpoints
        self.offensive_checkpoints = [False] * 5  # Simulating zones near the opponent's goal
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_checkpoints'] = self.midfield_checkpoints
        to_pickle['offensive_checkpoints'] = self.offensive_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_checkpoints = from_pickle['midfield_checkpoints']
        self.offensive_checkpoints = from_pickle['offensive_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "strategy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Encourage ball progression through midfield to strikers
            if o['right_team_active'][o['active']] and o['ball_owned_team'] == 1:
                ball_pos_x = o['ball'][0]
                
                # Midfield progression reward
                if 0.2 < ball_pos_x <= 0.5 and not self.midfield_checkpoints[rew_index]:
                    components["strategy_reward"][rew_index] += 0.5
                    self.midfield_checkpoints[rew_index] = True
                
                # Offensive progression reward
                if 0.5 < ball_pos_x <= 0.7 and not self.offensive_checkpoints[rew_index]:
                    components["strategy_reward"][rew_index] += 0.5
                    self.offensive_checkpoints[rew_index] = True

                # Incremental reward for approaching a goal
                if ball_pos_x > 0.7:
                    components["strategy_reward"][rew_index] += (ball_pos_x - 0.7) * 5

            # Update the reward with added components
            reward[rew_index] += components["strategy_reward"][rew_index]

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
