import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper to focus on enhancing performance in defensive maneuvers, including 
    accurate sliding tackles, timely sprints, and effective positioning to prevent opposition scoring.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "defensive_success_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Increase the reward if the active player successfully tackles by sliding or sprinting into a critical position
            if o['active'] != -1:
                if o['sticky_actions'][8] == 1:  # assuming index 8 represents 'action_sprint'  
                    components["defensive_success_reward"][rew_index] += 0.1  # Small reward for sprinting at the right time
                if o['sticky_actions'][9] == 1:  # assuming index 9 represents 'action_sliding'
                    components["defensive_success_reward"][rew_index] += 0.5  # Larger reward for successful slide
                
                # Positional component: reward for being between the ball and goal in a defensive stance
                player_pos = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
                own_goal_pos = (1, 0) if o['ball_owned_team'] == 1 else (-1, 0)  # goal positions need to be statically defined

                ball_pos = o['ball'][:2]  # Assuming ball position returned as x, y, supposedly
                distance_to_goal_line = np.abs(player_pos[rew_index][0] - own_goal_pos[0])
                if distance_to_goal_line < 0.5:  # arbitrary threshold for "good positioning"
                    components["defensive_success_reward"][rew_index] += 0.3

            reward[rew_index] += components["base_score_reward"][rew_index] + components["defensive_success_reward"][rew_index]

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
