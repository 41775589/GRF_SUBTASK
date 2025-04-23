import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that adds rewards for offensive strategies involving accurate shooting, dribbling,
    and specific types of passes in the Google Research Football environment.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        extra_state = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle.update(extra_state))

    def set_state(self, state):
        extra_state = self.env.set_state(state)
        self.sticky_actions_counter = extra_state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return extra_state

    def reward(self, reward):
        # Access observeration from the unwrapped env
        observation = self.env.unwrapped.observation()

        # Initialize reward components
        components = {
            "base_score_reward": np.array(reward).copy(),
            "shooting_reward": np.zeros(len(reward)),
            "dribbling_reward": np.zeros(len(reward)),
            "passing_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Define rewards for various scenarios
        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # if ball is owned by the controlled team
                if o['ball_owned_player'] == o['active']:
                    # Reward accurate shots on goal
                    if o['game_mode'] == 6:  # Handling the Penalty mode as a proxy for goal-bound shots
                        components['shooting_reward'][rew_index] = 1.0

                    # Reward long distance passes
                    if np.any(o['sticky_actions'][4:6]):  # Checking if a lob or high pass was executed
                        components['passing_reward'][rew_index] = 0.3
                    
                    # Reward effective dribbling moves
                    if o['sticky_actions'][9]:  # checking if dribble action is active
                        next_pos_ball = o['ball'] + o['ball_direction']
                        opponent_distances = np.linalg.norm(o['right_team'] - next_pos_ball[:2], axis=1)
                        if np.min(opponent_distances) > 0.1:  # Assuming dribbling is effective if the ball is moving away from opponents
                            components['dribbling_reward'][rew_index] = 0.5

            # Calculate total reward considering additional components
            reward[rew_index] += (components['shooting_reward'][rew_index] +
                                  components['dribbling_reward'][rew_index] +
                                  components['passing_reward'][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
