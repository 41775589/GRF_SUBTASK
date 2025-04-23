import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for practicing and mastering long passes in football.
    It aims to encourage the agent to learn accurate passes over various distances under different game conditions.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define several distance thresholds (in normalized coordinates) for long passes
        self.distance_thresholds = np.linspace(0.2, 1.0, num=5)  # thresholds for long pass distances
        self.pass_accuracy_reward = 0.2  # Reward for passing accuracy
        self.pass_completion_bonus = 0.1  # Additional bonus if the pass is completed successfully

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper'].get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_accuracy_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in [1, 2, 3, 4]:  # Only consider passes during active play modes
                if o['ball_owned_team'] == 1 or o['ball_owned_team'] == 0:  # Ensure that the ball is owned by a team
                    pass_distance = np.linalg.norm(np.array(o['ball_direction'])[:2])
                    pass_completed = np.any(o['sticky_actions'][9])  # Assuming action 9 corresponds to successful pass completion
                    
                    # Check the distance and reward increasingly for longer successful passes
                    for distance in self.distance_thresholds:
                        if pass_distance > distance:
                            components['pass_accuracy_reward'][rew_index] += self.pass_accuracy_reward
                            if pass_completed:
                                components['pass_accuracy_reward'][rew_index] += self.pass_completion_bonus

                # Adjust the base reward accordingly
                reward[rew_index] += components['pass_accuracy_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
