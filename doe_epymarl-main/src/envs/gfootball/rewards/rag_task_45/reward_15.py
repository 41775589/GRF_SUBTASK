import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards to focus on stopping, sprinting, and quick directional changes defensively."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds where rewards for stopping or sprinting are included
        self.stop_threshold = 0.1  # Threshold for minimal speed to consider stopping
        self.sprint_reward = 0.5  # Reward for sprinting in defensive situations
        self.stop_reward = 1.0  # Reward for stopping rapidly
        self.reaction_time_steps = 5  # Steps to wait before rewarding stop 

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            current_speed = np.linalg.norm(o['right_team_direction'][o['active']])
            previous_speed = np.linalg.norm(o.get('prev_right_team_direction', [0, 0]))

            components.setdefault("stop_reward", []).append(0.0)
            components.setdefault("sprint_reward", []).append(0.0)

            # Reward for stopping quickly
            if previous_speed > self.stop_threshold > current_speed:
                # Check if stopped within a small number of steps to react
                if o.get('steps_since_high_speed', 0) <= self.reaction_time_steps:
                    reward[rew_index] += self.stop_reward
                    components["stop_reward"][-1] = self.stop_reward

            # Reward sprinting in defensive situations
            if o['sticky_actions'][8] == 1:  # Check if sprint action is active
                dist_to_ball = np.linalg.norm(o['right_team'][o['active']] - o['ball'][:2])
                if dist_to_ball < 0.3:  # Considered defensive if close to ball
                    reward[rew_index] += self.sprint_reward
                    components["sprint_reward"][-1] = self.sprint_reward

            # Maintain history for the next step calculation
            o['prev_right_team_direction'] = o['right_team_direction'][o['active']]
            o['steps_since_high_speed'] = 0 if current_speed > self.stop_threshold else o.get('steps_since_high_speed', 0) + 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
