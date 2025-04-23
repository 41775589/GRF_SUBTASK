import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on the skill of executing long passes effectively under different match conditions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_distance_threshold = 0.5  # Distance representing a 'long pass'
        self.pass_accuracy_threshold = 0.1       # How close the ball must end to a teammate to be considered accurate

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Adjust the reward based on the execution of long passes.
        Reward is increased if a pass travels beyond a set threshold distance and is accurately received by a teammate.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_direction' in o and 'ball_owned_team' in o:
                ball_travel = np.linalg.norm(o['ball_direction'][:2])
                is_long_pass = ball_travel > self.long_pass_distance_threshold and o['ball_owned_team'] == 0

                if is_long_pass:
                    # Simulate accuracy by checking proximity to closest teammate after pass
                    ball_end_position = o['ball'][:2] + o['ball_direction'][:2]
                    teammate_distances = [
                        np.linalg.norm(ball_end_position - pos) for pos in o['left_team']
                    ] if o['ball_owned_team'] == 0 else [
                        np.linalg.norm(ball_end_position - pos) for pos in o['right_team']
                    ]

                    if min(teammate_distances) < self.pass_accuracy_threshold:
                        components["long_pass_reward"][rew_index] = 0.5  # Reward for successful long pass

            reward[rew_index] += components["long_pass_reward"][rew_index]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
