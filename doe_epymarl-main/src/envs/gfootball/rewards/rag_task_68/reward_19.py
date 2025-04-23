import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for offensive strategies including shooting, dribbling, 
    and passing in a more strategic manner.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_counter']
        return from_pickle

    def reward(self, reward):
        """
        Enhancing the reward based on offensive actions performed by the agent.
        Reward is given for:
        - shooting (direct goal attempt)
        - dribbling (using dribble action effectively)
        - passing (changing ball possession strategically amongst teammates)
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                
                # Shooting reward: check if the player is near the opponent's goal and the ball is being shot
                if np.hypot(o['ball'][0] - 1, o['ball'][1]) < 0.1 and np.linalg.norm(o['ball_direction']) > 0:
                    components["shooting_reward"][i] = self.shooting_reward
                    if o['game_mode'] in [0, 4]:  # Normal play or at corners
                        reward[i] += components["shooting_reward"][i]

                # Dribbling reward: based on dribble action being correctly used when close to opponents
                if o['sticky_actions'][9] == 1:  # dribble action index
                    close_opponents = np.linalg.norm(o['right_team'] - o['left_team'][o['active']], axis=1) < 0.03
                    if np.any(close_opponents):
                        components["dribbling_reward"][i] = self.dribbling_reward
                        reward[i] += components["dribbling_reward"][i]

                # Passing calculation reward: based on changing ownership when tightly covered
                if o['ball_direction'][2] > 0.1 and o['game_mode'] == 0:  # Indicates a potential pass action in normal play
                    components["passing_reward"][i] = self.passing_reward
                    reward[i] += components["passing_reward"][i]

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
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_val
                info[f"sticky_actions_{i}"] = action_val
        return observation, reward, done, info
