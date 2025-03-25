import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on advanced dribbling and sprint usage."""

    def __init__(self, env):
        super().__init__(env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 5  # Define the number of zones for dribbling checkpoints
        self._checkpoint_reward = 0.2  # Reward for reaching each dribbling checkpoint
        self._sprint_bonus = 0.05  # Bonus reward for using sprint effectively
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check for ball possession and controlling player
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # Calculate distance from the ball to the opponent's goal (goal at x=1)
                ball_x = o['ball'][0]
                dribbling_progress = min(max((ball_x + 1) / 2, 0), 1)  # Normalized from 0 to 1
                
                # Check and give rewards based on dribbling progress in segments
                segment = int(dribbling_progress * self._num_checkpoints)
                if segment > self._collected_checkpoints.get(rew_index, 0):
                    components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                    reward[rew_index] += self._checkpoint_reward
                    self._collected_checkpoints[rew_index] = segment

                # Check for sprint usage while dribbling
                if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Action 8 is sprint
                    components["sprint_reward"][rew_index] = self._sprint_bonus
                    reward[rew_index] += self._sprint_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
