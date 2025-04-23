import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for controlling the ball, making strategic plays, and exploiting open spaces.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize internal state
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActions']
        return from_pickle

    def reward(self, reward):
        """
        Reward function to incentivize ball control, strategic pass making,
        and utilizing open spaces effectively.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_reward": [0.0] * len(reward),
                      "strategic_play_reward": [0.0] * len(reward),
                      "space_exploitation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Reward for maintaining ball ownership
            if o['ball_owned_team'] == 0:  # assuming controlled team is always "0"
                components["control_reward"][idx] = 0.1
                reward[idx] += components["control_reward"][idx]

            # Reward for moving the ball towards less dense areas (open spaces)
            teammate_positions = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            ball_position = o['ball'][:2]
            density = sum(np.linalg.norm(teammate_positions - ball_position, axis=1) < 0.2)
            if density < 2:  # less than two players are within close range of the ball
                components["space_exploitation_reward"][idx] = 0.2
                reward[idx] += components["space_exploitation_reward"][idx]

            # Reward for passes made toward strategic regions (toward goal, or splitting defense)
            if 'action' in o:  # assuming 'action' key in observation indicates recent action
                if o['action'] == "pass" and np.linalg.norm(ball_position - [1, 0]) < 0.5:  # near opponent's goal
                    components["strategic_play_reward"][idx] = 0.3
                    reward[idx] += components["strategic_play_reward"][idx]

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
