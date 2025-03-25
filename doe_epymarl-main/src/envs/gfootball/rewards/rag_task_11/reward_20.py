import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances training for faster maneuvers and precise control in offensive plays.
    Consider ball control, approach speed to goal, and accurate shot making for rewards.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_zones = 5  # Dividing the field into zones for precise play
        self.zone_divider = np.linspace(-0.42, 0.42, self.num_zones+1)  # Vertical field division
        self.control_bonus = 0.05
        self.shot_accuracy_bonus = 0.1
        self.approach_speed_bonus = 0.05
        self.score_bonus = 1.0  # Bonus for scoring

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_bonus": [0.0] * len(reward),
                      "shot_accuracy": [0.0] * len(reward),
                      "approach_speed": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Control bonuses
                control_factor = np.any(o['sticky_actions'][[8, 9]])  # Check for sprint or dribble
                components["control_bonus"][i] = control_factor * self.control_bonus
                reward[i] += components["control_bonus"][i]

                # Determine the zone of approach
                y_pos = o['ball'][1]
                for j in range(self.num_zones):
                    if self.zone_divider[j] <= y_pos < self.zone_divider[j+1]:
                        components["approach_speed"][i] = self.approach_speed_bonus
                        reward[i] += components["approach_speed"][i]
                        break

                # Shot accuracy when in last zone towards goal
                if y_pos >= self.zone_divider[-2]:
                    direction_to_goal = (1 - o['ball'][0])**2 + (0 - o['ball'][1])**2  # Simple Euclidean distance to center goal
                    shot_accuracy = np.clip(1 - np.sqrt(direction_to_goal), 0, 1)  # Normalize and clamp
                    components["shot_accuracy"][i] = shot_accuracy * self.shot_accuracy_bonus
                    reward[i] += components["shot_accuracy"][i]

            # Additional bonus for scoring
            if 1 in o['score']:
                reward[i] += self.score_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
