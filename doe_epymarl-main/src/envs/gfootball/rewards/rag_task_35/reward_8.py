import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This wrapper augments the reward with additional positional strategy benefits for maintaining effective positioning and transitions between defense and attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = np.zeros((2,), dtype=float)  # Assume two players for simplicity
        self.offensive_transitions = np.zeros((2,), dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions.fill(0.0)
        self.offensive_transitions.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_positions': self.defensive_positions,
            'offensive_transitions': self.offensive_transitions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.defensive_positions = wrapper_state['defensive_positions']
        self.offensive_transitions = wrapper_state['offensive_transitions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for index, (rew, obs) in enumerate(zip(reward, observation)):
            # Reward defensive positioning
            if obs['game_mode'] in {2, 3, 4, 6}:  # Defensive modes
                if obs['ball_owned_team'] == 1:  # Enemy has the ball
                    distance_to_goal = np.linalg.norm(obs['ball'][0] - (-1))  # X distance to left goal
                    if distance_to_goal < self.defensive_positions[index]:
                        components["positioning_reward"][index] += 0.01
                        self.defensive_positions[index] = distance_to_goal

            # Reward offensive transitions
            if obs['game_mode'] == 0:  # Normal game mode
                if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                    if self.offensive_transitions[index] == 0:
                        components["positioning_reward"][index] += 0.1
                        self.offensive_transitions[index] = 1  # Transition acknowledged
            else:
                self.offensive_transitions[index] = 0  # Reset on game mode change

            # Final reward calculation
            reward[index] += components["positioning_reward"][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
