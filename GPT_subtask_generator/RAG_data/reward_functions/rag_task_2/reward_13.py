import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defensive teamwork and coordination reward.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Specific parameters can be adjusted to balance the rewards
        self.teamwork_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "teamwork_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage defensive positioning and teamwork
            if 'left_team' in o and 'right_team' in o and 'ball_owned_team' in o:
                # Calculate the distance of team players to the ball
                if o['ball_owned_team'] == 0:  # The ball is owned by the left team
                    own_team_pos = o['left_team']
                    opponent_team_pos = o['right_team']
                elif o['ball_owned_team'] == 1:
                    own_team_pos = o['right_team']
                    opponent_team_pos = o['left_team']
                else:
                    continue  # Ball is not possessed

                # Calculate distances to ball and between team members
                ball_pos = o['ball'][:2]  # Only x, y positions
                distances_to_ball = np.linalg.norm(own_team_pos - ball_pos, axis=1)
                min_distance_to_ball = np.min(distances_to_ball)
                
                # Reward for close defensive coverage
                if min_distance_to_ball < 0.1:  # Threshold can be tuned
                    components["teamwork_reward"][rew_index] = self.teamwork_reward
                    reward[rew_index] += components["teamwork_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter to maintain internal consistency
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
