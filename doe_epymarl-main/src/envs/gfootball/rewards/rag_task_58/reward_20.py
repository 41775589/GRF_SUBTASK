import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focusses on mastering defensive coordination,
    including unified responses to various attacking scenarios and efficient
    transition from defense to attack through secure ball distribution.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.ball_possession_change_reward = 0.5
        self.defensive_positioning_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
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
        # Initialize the reward modifications and the reward components dictionary
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_possession_change_reward": [0.0] * len(reward),
                      "defensive_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        previous_owner_team = None

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for successful transition from defense to attack through secure ball distribution
            current_owner_team = o['ball_owned_team']
            if previous_owner_team is not None and previous_owner_team != current_owner_team:
                if current_owner_team == 0:  # assuming '0' represents the agent's team
                    components["ball_possession_change_reward"][rew_index] += self.ball_possession_change_reward

            previous_owner_team = current_owner_team

            # Reward for maintaining good defensive positioning relative to the ball and the goals
            ball_position = np.array(o['ball'][:2])  # considering x, y coords
            player_position = np.array(o['left_team'][o['active']][:2])

            # Distance to the center line (normalized distance to the goal line which is '0' in the environment)
            distance_to_center_line = abs(player_position[0])
            components["defensive_positioning_reward"][rew_index] += self.defensive_positioning_reward / (1 + 10 * distance_to_center_line)

            # Aggregate the rewards
            reward[rew_index] += components["ball_possession_change_reward"][rew_index]
            reward[rew_index] += components["defensive_positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
