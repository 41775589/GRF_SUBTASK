import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds reward for strategic positioning, passing, and shooting skills. """

    def __init__(self, env):
        super().__init__(env)
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
        components = {"base_score_reward": reward.copy()}

        # Initialize reward components for strategic passes and shots
        components["strategic_pass_reward"] = [0.0] * len(reward)
        components["strategic_shot_reward"] = [0.0] * len(reward)
        components["positioning_reward"] = [0.0] * len(reward)

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for effective passing
            if o['game_mode'] in [0] and o['ball_owned_player'] == o['active']:
                # Checking if a pass would be strategic
                teammates_pos = o['right_team' if o['ball_owned_team'] == 1 else 'left_team']
                vector_to_teammates = teammates_pos - o['ball'][:2]
                distances_to_teammates = np.linalg.norm(vector_to_teammates, axis=1)
                short_passes = distances_to_teammates < 0.3
                if short_passes.any():
                    components["strategic_pass_reward"][rew_index] = 0.05

            # Reward for taking strategic shots at the goal
            goal_position = np.array([1, 0]) if o['ball_owned_team'] == 1 else np.array([-1, 0])
            distance_to_goal = np.linalg.norm(o['ball'][:2] - goal_position)
            if distance_to_goal < 0.3 and o['game_mode'] in [0]:
                components["strategic_shot_reward"][rew_index] = 0.1

            # Reward for positioning closer to the center forward position during an attack
            if o['ball_owned_team'] == 1 and o['right_team_roles'][o['active']] == 9:
                distance_to_center_forward = np.linalg.norm(o['right_team'][o['active']] - np.array([0.8, 0]))
                if distance_to_center_forward < 0.2:
                    components["positioning_reward"][rew_index] = 0.05
            elif o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] == 9:
                distance_to_center_forward = np.linalg.norm(o['left_team'][o['active']] - np.array([-0.8, 0]))
                if distance_to_center_forward < 0.2:
                    components["positioning_reward"][rew_index] = 0.05

        # Applying each calculated component to adjust the overall reward
        for rew_index in range(len(reward)):
            reward[rew_index] += sum(
                compl[rew_index] for compl in components.values() if compl[rew_index] > 0
            )

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
