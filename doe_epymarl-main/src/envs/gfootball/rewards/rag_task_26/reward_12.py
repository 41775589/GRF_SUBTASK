import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering midfield dynamics focusing on distinct roles and game transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.role_reward = 0.1
        self.transition_reward = 0.2
        self._last_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._last_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self._last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle.get('last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components["role_reward"] = [0.0] * len(reward)
        components["transition_reward"] = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            prev_ball_position = self._last_ball_position
            current_ball_position = o.get('ball')

            # Handling role-based rewards for midfield players
            ball_owned_by_player = o.get('ball_owned_player')
            if ball_owned_by_player is not None:
                player_role = o['left_team_roles'][ball_owned_by_player] if o['ball_owned_team'] == 0 else o['right_team_roles'][ball_owned_by_player]
                # Central midfield (role 5) or wide midfield roles (6 and 7)
                if player_role in (5, 6, 7):
                    components["role_reward"][rew_index] = self.role_reward
                    reward[rew_index] += self.role_reward

            # Handling game transition rewards
            if prev_ball_position is not None and current_ball_position is not None:
                prev_y = prev_ball_position[1]
                current_y = current_ball_position[1]
                if abs(current_y) > abs(prev_y):  # Assuming positive movement towards opponent's goal
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += self.transition_reward

            # Update last ball position
            self._last_ball_position = current_ball_position

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
