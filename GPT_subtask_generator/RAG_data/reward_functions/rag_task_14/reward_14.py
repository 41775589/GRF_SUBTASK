import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for the 'sweeper' role focusing on clearing the ball, tackling, and recovery."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients
        self.clear_ball_reward = 0.1
        self.last_man_tackle_reward = 0.2
        self.recovery_speed_reward = 0.05

        self.previous_ball_position = None
        self.previous_distance_to_goal = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.previous_distance_to_goal = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_position'] = self.previous_ball_position
        to_pickle['previous_distance_to_goal'] = self.previous_distance_to_goal
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.previous_ball_position = from_pickle['previous_ball_position']
        self.previous_distance_to_goal = from_pickle['previous_distance_to_goal']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clear_ball_reward": [0.0] * len(reward),
            "last_man_tackle_reward": [0.0] * len(reward),
            "recovery_speed_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball' in o:
                current_ball_position = o['ball'][:2]
                distance_to_goal = np.linalg.norm(np.array([1, 0]) - current_ball_position)

                # Reward for moving the ball away from the goal area when the team is under threat
                if self.previous_ball_position is not None and o['ball_owned_team'] == 0:  # Team 0 is the defensive team
                    distance_diff = np.linalg.norm(np.array(current_ball_position) - np.array(self.previous_ball_position))
                    if distance_diff > 0.1:  # Ball was significantly moved
                        components["clear_ball_reward"][rew_index] = self.clear_ball_reward

                # Reward last-man tackles preventing opponent from scoring
                if o['game_mode'] in [3, 6] and o['ball_owned_team'] == 1:  # FreeKick or Penalty
                    components["last_man_tackle_reward"][rew_index] = self.last_man_tackle_reward

                # Recovery speed
                if self.previous_distance_to_goal is not None and distance_to_goal < self.previous_distance_to_goal:
                    components["recovery_speed_reward"][rew_index] = self.recovery_speed_reward

                self.previous_ball_position = current_ball_position
                self.previous_distance_to_goal = distance_to_goal

            # Calculate combined reward
            reward[rew_index] += sum(components[c][rew_index] for c in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                if action_status:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_status

        return observation, reward, done, info
