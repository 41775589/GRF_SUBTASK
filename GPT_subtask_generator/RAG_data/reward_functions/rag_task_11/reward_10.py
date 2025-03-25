import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on offensive maneuvers and precision passing.
    It encourages fast-paced play and controlled movements near the opponent's goal area.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Sticky actions count for rewarding dribbling and sprinting efficiency.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Initialize reward components to calculate the final rewards for all agents dynamically.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        # Check for None to avoid execution during environment reset.
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Direct goal reward included in base reward (positive if scored).
            # Offensive reward increases for ball advancement towards the opponent's goal.
            if o['ball_owned_team'] == 0: # Owned by left team (agent's team)
                ball_x = o['ball'][0]
                close_to_goal = max(0, (ball_x - 0.5) * 2)  # Reward scaling from mid-point to goal.
                components['offensive_play_reward'][i] = close_to_goal

                # Track dribbling and sprinting efficiency.
                sprint_action = o['sticky_actions'][8] # Index for sprint action
                dribble_action = o['sticky_actions'][9] # Index for dribble action
                if sprint_action or dribble_action:
                    self.sticky_actions_counter[i] += 1
                    components['offensive_play_reward'][i] += 0.01 * self.sticky_actions_counter[i]  # Encourage maintained control

            # Finalize individual rewards incorporating components.
            reward[i] += components['offensive_play_reward'][i]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
