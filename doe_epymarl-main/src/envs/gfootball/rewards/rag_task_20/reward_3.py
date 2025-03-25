import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward scheme aiming to enhance offensive strategies and coordination."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Including checkpoints in the serialized state might help in restoration
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Get the current observation to compute additional rewards based on the offensive actions
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components['offensive_reward'] = [0.0] * len(reward)

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            # Encourage forwarding the ball towards the opponent's goal zone
            ball_progress = o['ball'][0]  # Assuming x-axis is towards the opponent's goal
            components['offensive_reward'][rew_index] = 0.05 * ball_progress
            reward[rew_index] += components['offensive_reward'][rew_index]

            # Reward coordination: Passing attempts towards teammates closer to goal
            if o['ball_owned_team'] == 1 and o['game_mode'] == 0:
                active_player_position = o['right_team'][o['active']]
                teammates_positions = o['right_team']
                forward_teammates = [team for team in teammates_positions if team[0] > active_player_position[0]]
                reward[rew_index] += 0.03 * len(forward_teammates)  # Bonus for each teammate forward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
