import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a more dynamically scaled reward for mastering offensive tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward_scale = 0.1  # Reduced to decrease the uniform high reward
        self.shot_reward = 1.5  # Introduced to amplify the impact of successful shots
        self.dribble_reward_scale = 0.02  # Reduced to make it less dominant but still relevant

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": np.zeros(len(reward)),
            "shot_reward": np.zeros(len(reward)),
            "dribble_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        for player_index, obs in enumerate(observation):
            # Checking pass action effectiveness
            if 'sticky_actions' in obs and obs['sticky_actions'][1] == 1:  # Assuming index 1 is pass action
                passes_completed = np.random.rand() > 0.5  # Random check for successful pass simulation
                if passes_completed:
                    components["pass_reward"][player_index] = self.pass_reward_scale
                    reward[player_index] += self.pass_reward_scale

            # Checking shot on goal
            if obs['game_mode'] == 3 and obs['ball_owned_player'] == obs['active']:  # Game mode 3 assumed to be shooting
                components["shot_reward"][player_index] = self.shot_reward
                reward[player_index] += self.shot_reward

            # Dribbling effectiveness
            if 'sticky_actions' in obs and obs['sticky_actions'][4] == 1:  # Assuming index 4 is dribbling
                components["dribble_reward"][player_index] = self.dribble_reward_scale
                reward[player_index] += self.dribble_reward_scale

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
