import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for transition skills like Short Pass, Long Pass, and Dribble."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._max_pass_reward = 0.3
        self._dribble_reward = 0.05
        self._control_bonus_threshold = 0.4  # Circular distance threshold for control while dribbling.

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

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "transition_skills_reward": [0.0] * len(reward)}

        for i, obs in enumerate(observation):
            if obs is None:
                continue

            ball_controlled_by_active_player = (
                obs['ball_owned_team'] == 0 and 
                obs['ball_owned_player'] == obs['active']
            )

            if ball_controlled_by_active_player:
                direction_mag = np.linalg.norm(obs['ball_direction'][:2])
                
                # Check distance moved with the ball to assess dribbling under pressure
                if direction_mag > self._control_bonus_threshold:
                    components["transition_skills_reward"][i] += self._dribble_reward

                # Bonus for discerning type of pass or maintaining ball control
                sticky_actions = obs['sticky_actions']
                if sticky_actions[7] or sticky_actions[9]:  # Assume indices for long pass and short_pass
                    components["transition_skills_reward"][i] += self._max_pass_reward * min(1, direction_mag)

            reward[i] += components["transition_skills_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Update the core rewards and component contributions in the step return
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
