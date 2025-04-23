import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive tactics specifically tackling without committing fouls."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_foul_tackles = 0
        self.reward_for_tackle_without_foul = 0.1
        self.penalty_for_foul = -0.2

    def reset(self):
        """ Reset the sticky actions and other metrics for a new episode. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_foul_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Collect and return the state specific for this wrapper. """
        to_pickle['non_foul_tackles'] = self.non_foul_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state specific for this wrapper. """
        from_pickle = self.env.set_state(state)
        self.non_foul_tackles = from_pickle['non_foul_tackles']
        return from_pickle

    def reward(self, reward):
        """ Modify the reward by adding tackling bonuses and penalizing fouls. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward).copy(),
                      "tackle_reward": np.array([0.0, 0.0])}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] in [3, 6] and o['ball_owned_team'] == 1 and o['right_team_yellow_card'][o['ball_owned_player']]:
                # Applying rewards/penalties only to our team (indexed by '1' for right team here)
                # Check if a foul caused a free kick or penalty
                components['tackle_reward'][rew_index] = self.penalty_for_foul
            elif o['game_mode'] == 0:  # Game mode is normal
                if np.any(o['sticky_actions'][7:9]):  # Tackling actions
                    # Check if ball possession is kept without causing a foul
                    if o['ball_owned_team'] == 0 and not np.any(o['left_team_yellow_card']):
                        components['tackle_reward'][rew_index] = self.reward_for_tackle_without_foul
                        self.non_foul_tackles += 1

            # Aggregate the rewards
            total_reward = reward[rew_index] + components['tackle_reward'][rew_index]
            reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        """ Take a step using the wrapped environment and modify the reward """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
