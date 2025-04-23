import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing the effectiveness of collaborative
    plays between shooters and passers. It promotes passing behavior towards teammates
    who are well-positioned to exploit scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_position_reward = 0.05
        self.passing_bonus = 0.1
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
        """
        Modifies the reward based on the positioning and interplay between passers
        and shooters, aiming to improve collaborative play scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage passing by giving a reward if a pass leads to a teammate
            # with a good scoring position
            if o['ball_owned_team'] == 0 and 'ball_owned_player' in o:
                passer = o['active']
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Checking if dribble action is active
                    # Check if player has potential passing targets
                    for idx, (player_x, player_y) in enumerate(o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']):
                        # Reward for passes towards teammate closer to opponent's goal
                        if player_x > o['ball'][0] + 0.1:  # Teammate is ahead towards the goal
                            components["passing_reward"][rew_index] = self.passing_bonus
                            reward[rew_index] += self.passing_bonus

            # Adding shooting position rewards based on proximity to the goal
            ball_x, ball_y = o['ball'][:2]
            if abs(ball_x) > 0.5 and abs(ball_y) < 0.2:  # Player is near the central part of the opponent's goal area
                components["positioning_reward"][rew_index] = self.shooting_position_reward
                reward[rew_index] += self.shooting_position_reward

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

    def __str__(self):
        return "CheckpointRewardWrapper: Encourages effective collaborative plays between passers and shooters."
