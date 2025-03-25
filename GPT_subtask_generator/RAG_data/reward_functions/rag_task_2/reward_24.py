import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing teamwork and coordination in defensive strategies.
    This includes rewards for effective ball control and intelligent positioning to disrupt the opponent's play.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        self.cumulative_defensive_rewards = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        self.cumulative_defensive_rewards = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['prev_ball_position'] = self.prev_ball_position
        to_pickle['cumulative_defensive_rewards'] = self.cumulative_defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_ball_position = from_pickle['prev_ball_position']
        self.cumulative_defensive_rewards = from_pickle['cumulative_defensive_rewards']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # Enemy controls the ball
                # Calculate the distance from each agent to the ball
                for i, pos in enumerate(o['left_team']):
                    dist_to_ball = np.linalg.norm(o['ball'][:2] - pos)
                    if dist_to_ball < 0.1:
                        components["defensive_positioning_reward"][rew_index] += 0.05
                        reward[rew_index] += components["defensive_positioning_reward"][rew_index]
                        
            # Encourage players to stay in proper positions to have defensive coverage
            if self.prev_ball_position is not None:
                ball_movement_dist = np.linalg.norm(self.prev_ball_position - o['ball'][:2])
                components["defensive_positioning_reward"][rew_index] -= 0.01 * ball_movement_dist
                reward[rew_index] += components["defensive_positioning_reward"][rew_index]

            # Update the previous ball position
            self.prev_ball_position = o['ball'][:2]
            self.cumulative_defensive_rewards += components["defensive_positioning_reward"][rew_index]

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()  # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
