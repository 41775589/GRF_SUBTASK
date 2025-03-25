import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically rewards defensive actions within the football environment."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Establish checkpoints for defensive positioning and actions
        self._interception_reward = 0.3
        self._tackle_reward = 0.5
        self._defensive_position_reward = 0.2
        self._max_tackle_distance = 0.2  # Max distance to consider a tackle

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
                      "interception_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "defensive_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive positioning reward for being close to an opponent with the ball
            if o['ball_owned_team'] == 1:  # Opponent team has the ball
                for opponent_pos in o['right_team']:
                    for defender_pos in o['left_team']:
                        distance = np.linalg.norm(opponent_pos - defender_pos)
                        if distance < self._max_tackle_distance:
                            components["tackle_reward"][rew_index] += self._tackle_reward
                            components["defensive_position_reward"][rew_index] += self._defensive_position_reward

            # Interception reward for changing the ball possession from the opponent
            if o['ball_owned_team'] == 1 and 'ball_owned_player' in o:
                current_ball_owner = o['ball_owned_player']
                if current_ball_owner in o['left_team']:
                    components["interception_reward"][rew_index] += self._interception_reward

            # Update the reward based on the components
            reward[rew_index] += components["interception_reward"][rew_index]
            reward[rew_index] += components["tackle_reward"][rew_index]
            reward[rew_index] += components["defensive_position_reward"][rew_index]

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
