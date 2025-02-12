import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A new reward wrapper designed to balance the defensive contributions more effectively,
    increasing alignment with actual gameplay strategies.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Coefficients for each reward component
        self.def_action_coefficient = 1.0  # Defensive actions
        self.pos_control_coefficient = 0.5  # Position control
        self.base_coefficient = 0.1  # The coefficient for the original score

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defense_reward_wrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_reward": [0.0] * len(reward),
                      "position_control_reward": [0.0] * len(reward)}

        for index, o in enumerate(observation):
            # Check if the active player is on defense and has the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Encourage passing safely
                if o['sticky_actions'][1]:  # Assuming action '1' is 'Short Pass'
                    components["defensive_action_reward"][index] += self.def_action_coefficient

                # Position control relative to ball and own goal
                player_pos = np.array(o['left_team'][o['active']])
                goal_pos = np.array([1.0, 0])  # Assuming the goal is on the right side
                ball_pos = np.array(o['ball'][:2])
                
                dist_to_ball = np.linalg.norm(player_pos - ball_pos)
                dist_to_goal = np.linalg.norm(player_pos - goal_pos)

                # Reward closer distance to ball and own goal
                components["position_control_reward"][index] += (self.pos_control_coefficient /
                    (dist_to_ball + 0.1)) + (self.pos_control_coefficient / (dist_to_goal + 0.1))

            # Combine all components into the final reward
            reward[index] = (self.base_coefficient * components["base_score_reward"][index] +
                             components["defensive_action_reward"][index] +
                             components["position_control_reward"][index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Calculate final reward and update info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
