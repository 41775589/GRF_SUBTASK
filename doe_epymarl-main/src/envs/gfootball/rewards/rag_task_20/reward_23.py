import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for enhancing offensive strategies in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_attempts = 0
        self.successful_passes = 0
        self.shoot_attempts = 0
        self.placement_quality = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_attempts = 0
        self.successful_passes = 0
        self.shoot_attempts = 0
        self.placement_quality = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "pass_attempts": self.pass_attempts,
            "successful_passes": self.successful_passes,
            "shoot_attempts": self.shoot_attempts,
            "placement_quality": self.placement_quality
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_attempts = from_pickle['CheckpointRewardWrapper']['pass_attempts']
        self.successful_passes = from_pickle['CheckpointRewardWrapper']['successful_passes']
        self.shoot_attempts = from_pickle['CheckpointRewardWrapper']['shoot_attempts']
        self.placement_quality = from_pickle['CheckpointRewardWrapper']['placement_quality']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward),
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for successful passes, checks for pass action and change in ball possession
            if 'ball_owned_player' in o and o['ball_owned_team'] == 1:  # Assuming 1 is the team index
                self.pass_attempts += 1
                if o['action'] == 'pass' and o['ball_owned_player'] != o['old_ball_owned_player']:
                    self.successful_passes += 1
                    components["pass_reward"][rew_index] = 0.2
                    reward[rew_index] += components["pass_reward"][rew_index]

            # Reward for shooting attempts towards the goal
            if 'action' in o and o['action'] == 'shoot':
                self.shoot_attempts += 1
                components["shoot_reward"][rew_index] = 0.1
                reward[rew_index] += components["shoot_reward"][rew_index]

            # Position improvement reward considering proximity to opponent's goal
            ball_position = np.array(o['ball'][:2])  # assuming ball observation includes only x, y
            distance_to_goal = np.linalg.norm(ball_position - np.array([1, 0]))  # opponent goal at x=1
            if 'old_ball_position' in o:  # Need to store the old ball position in observation after action
                old_distance = np.linalg.norm(np.array(o['old_ball_position'][:2]) - np.array([1, 0]))
                improvement = old_distance - distance_to_goal
                if improvement > 0:
                    self.placement_quality += improvement
                    reward[rew_index] += improvement * 0.05  # Scale the improvement

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
