import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on offensive plays such as shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.offensive_bonus = 0.1
        self.shoot_distance_threshold = 0.2
        self.shoot_angle_threshold = 0.1
        self.dribble_effectiveness_threshold = 0.05
        self.pass_success_threshold = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "offensive_play_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["offensive_play_bonus"][rew_index] = 0.0

            if o['ball_owned_team'] == 0:  
                # Dribbling effectiveness
                if any(o['sticky_actions'][6:10]) and np.linalg.norm(o['ball_direction'][:2]) > self.dribble_effectiveness_threshold:
                    components["offensive_play_bonus"][rew_index] += self.offensive_bonus

                # Shooting towards goal
                goal_y_range = np.array([-0.044, 0.044])
                if o['ball'][0] > (1 - self.shoot_distance_threshold) and goal_y_range[0] < o['ball'][1] < goal_y_range[1]:
                    components["offensive_play_bonus"][rew_index] += self.offensive_bonus

                # Successful passes measure
                if any(o['sticky_actions'][0:2]) and np.linalg.norm(o['ball_direction'][:2]) > self.pass_success_threshold:
                    components["offensive_play_bonus"][rew_index] += self.offensive_bonus

            # Sum component rewards to base reward
            reward[rew_index] += components["offensive_play_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active

        return observation, reward, done, info
