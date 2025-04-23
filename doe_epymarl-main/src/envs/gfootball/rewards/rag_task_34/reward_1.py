import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for close-range assaults, dribbling and shooting precision."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_distance_threshold = 0.1  # proximity to goal to initiate shooting

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_precision_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward for dribbling effectively when near the goalkeeper
        # and for precision in shooting
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            distance_to_goal = abs(o['ball'][0] - 1.0)  # distance to right team goal on x-axis

            # Check dribbling effectiveness and close-range shot precision
            is_controlled_by_player = o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']
            if is_controlled_by_player:
                # Dribbling reward
                if 'action_dribble' in o['sticky_actions']:
                    components["dribble_reward"][rew_index] += 0.05

                # Close range shot with precision
                if distance_to_goal < self.goal_distance_threshold:
                    # Ensuring shot direction towards goal when close enough
                    ball_goal_direction = np.sign(o['ball_direction'][0]) == 1
                    if ball_goal_direction and 'action_shot' in o:
                        components["shot_precision_reward"][rew_index] += 0.1

            # Aggregate rewards
            reward[rew_index] += sum(components.values())

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
