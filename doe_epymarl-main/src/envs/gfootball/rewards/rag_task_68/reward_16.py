import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on offensive strategies: dribbling, shooting, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.2
        self.passing_reward = 0.4
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": np.zeros(len(reward)),
                      "dribbling_reward": np.zeros(len(reward)),
                      "passing_reward": np.zeros(len(reward))}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']]
            
            # Check if the agent successfully executed any shots or dribbles while close to opponent's goal
            goal_x = 1
            close_to_goal_threshold = 0.2
            if abs(active_player_pos[0] - goal_x) < close_to_goal_threshold:
                if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    if o['sticky_actions'][9]:  # Assuming index 9 is the shooting action
                        components["shooting_reward"][rew_index] = self.shooting_reward
                    if o['sticky_actions'][8]:  # Assuming index 8 is the dribble action
                        components["dribbling_reward"][rew_index] = self.dribbling_reward

            # Check if the agent successfully executed a long or high pass
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['designated']:
                if o['ball_direction'][0] > 0.5:  # Assuming positive x-direction passing can be significant
                    components["passing_reward"][rew_index] = self.passing_reward
            
            reward[rew_index] += (components["shooting_reward"][rew_index] +
                                  components["dribbling_reward"][rew_index] +
                                  components["passing_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
