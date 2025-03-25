import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on offensive skills.
    This wrapper focuses on rewarding actions related to passing, shooting, and dribbling,
    as well as progressing towards the opponent's goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for different actions
        self.pass_reward = 0.05
        self.shot_reward = 0.1
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02
        # Define progression rewards towards the goal
        self.progression_rewards = np.linspace(0.01, 0.1, 10)

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

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            original_reward = reward[rew_index]
            midfield_x = 0.0  # midfield x-coordinate

            # Calculating positional reward based on progression towards opponent's goal
            if o['ball_owned_team'] == 1:
                distance_from_midfield = abs(midfield_x - o['ball'][0])
                progress = min(int(distance_from_midfield / 0.1), 9)
                reward[rew_index] += self.progression_rewards[progress]

            # Adding rewards for special actions
            reward[rew_index] += o['sticky_actions'][8] * self.dribble_reward  # dribble
            reward[rew_index] += o['sticky_actions'][9] * self.sprint_reward  # sprint

            if o['game_mode'] == 6:  # shot or penalty mode
                reward[rew_index] += self.shot_reward

            if o['sticky_actions'][0] or o['sticky_actions'][2]:  # short or long pass
                reward[rew_index] += self.pass_reward

            # Update the components dictionary
            components[f"progression_reward_{rew_index}"] = reward[rew_index] - original_reward

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
