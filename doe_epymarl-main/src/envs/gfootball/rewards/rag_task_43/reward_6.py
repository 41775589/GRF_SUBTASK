import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive strategy and counterattack capabilities
       with rewards for good positioning and responsiveness."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "counterattack_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Defensive reward: well-positioning between own goal and ball when other team has possession
            if o['ball_owned_team'] == 1:
                player_pos = o['left_team'][o['active']]
                ball_x, ball_y = o['ball'][:2]
                team_goal = -1  # left team goal position
                distance_to_ball = np.linalg.norm(player_pos - np.array([ball_x, ball_y]))
                distance_to_goal = np.linalg.norm(player_pos - np.array([team_goal, 0]))

                if distance_to_goal < distance_to_ball:
                    components["defensive_positioning_reward"][rew_index] = 0.5
                
            # Counterattack transition reward: Encourage quick ball interception and movement towards opponent's half
            if o['ball_owned_team'] == 0: # left team possession
                player_pos = o['left_team'][o['active']]
                if player_pos[0] > 0: # If in opponents' half after taking over the ball
                    components["counterattack_transition_reward"][rew_index] = 0.5

            reward[rew_index] += (components["defensive_positioning_reward"][rew_index] + 
                                  components["counterattack_transition_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
