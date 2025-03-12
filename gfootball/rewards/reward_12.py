import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the learning of offensive football skills."""

    def __init__(self, env):
        super().__init__(env)
        self._checkpoint_rewards = [0, 0, 0, 0, 0]

    def reset(self):
        self._checkpoint_rewards = [0, 0, 0, 0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._checkpoint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._checkpoint_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "shot_accuracy_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            base_reward, dribble_reward, pass_reward, shot_accuracy_reward = self.calculate_rewards(o, idx)
            components["dribble_reward"][idx] = dribble_reward
            components["pass_reward"][idx] = pass_reward
            components["shot_accuracy_reward"][idx] = shot_accuracy_reward
            reward[idx] += dribble_reward + pass_reward + shot_accuracy_reward

        return reward, components

    def calculate_rewards(self, observation, index):
        base_reward = 0
        dribble_reward = 0
        pass_reward = 0
        shot_accuracy_reward = 0
        if observation['game_mode'] == 0:  # Normal play mode
            if observation['ball_owned_team'] == 1 and observation['ball_owned_player'] == observation['active']:
                # Check dribbling effectiveness
                dribble_effectiveness = np.clip(np.linalg.norm(observation['ball_direction'][:2]), 0, 1)
                dribble_reward = 0.05 * dribble_effectiveness
                # Reward for successful passes
                if 'action' in observation['sticky_actions']:
                    if observation['sticky_actions'][9]:  # Action for a high pass
                        pass_reward = 0.1
                    elif observation['sticky_actions'][8]:  # Action for a low pass
                        pass_reward = 0.07
                # Evaluation of shot accuracy
                goal_distance = np.linalg.norm([observation['ball'][0] - 1, observation['ball'][1]])
                if goal_distance < 0.2:  # Close to opponent's goal
                    shot_accuracy_reward = 0.3
        return base_reward, dribble_reward, pass_reward, shot_accuracy_reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
