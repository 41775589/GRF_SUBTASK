import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specific rewards based on the position and actions related to transitioning from
    defense to offense for left back and right back players.
    """
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
        components = {"base_score_reward": reward.copy(), "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Reward for good positioning and making successful transitions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            own_goal_x = -1 if o['left_team'][0][0] > 0 else 1  # Determine which side is the own goal based on left team player position

            # Encourage moving forward towards the opponent's half when in possession
            if o['ball_owned_team'] == 0:  # Assuming '0' is the team index of our agent team
                ball_x_position = o['ball'][0]
                player_x_position = o['left_team'][rew_index][0]

                if ball_x_position * own_goal_x > player_x_position * own_goal_x:
                    # Reward players moving towards the opponent's half with the ball
                    components['positioning_reward'][rew_index] = 0.1
                    reward[rew_index] += components['positioning_reward'][rew_index]

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
