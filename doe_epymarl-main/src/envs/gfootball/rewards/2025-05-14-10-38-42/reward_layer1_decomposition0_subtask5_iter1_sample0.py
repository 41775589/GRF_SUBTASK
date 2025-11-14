import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that reinforces passing skills under pressure from defensive positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_modifier = 0.5  # Reward for successfully executing a pass under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "passing_reward": [0.0]}

        if observation is None:
            return reward, components  # Immediate return if no observation is available

        o = observation[0] # Single agent scenario
        ball_owned_team = o['ball_owned_team']

        if ball_owned_team == 0:  # Ball is controlled by the agent's team
            player_x = o['left_team'][o['active']][0]  # Fetch the x-coordinate of the active player
            if player_x < -0.3:  # Defensive position is generally in the left third of the field
                if o['sticky_actions'][9] == 1:  # Assuming index 9 in sticky actions corresponds to 'High Pass'
                    components['passing_reward'][0] = self.pass_reward_modifier  # Apply modifier for successful high pass
                    reward += components['passing_reward'][0]
                if o['sticky_actions'][10] == 1:  # Assuming index 10 corresponds to 'Short Pass'
                    components['passing_reward'][0] = self.pass_reward_modifier * 0.5  # Less reward for short pass
                    reward += components['passing_reward'][0]

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
                self.sticky_actions_counter[i] += action  # Count sticky actions
        return observation, reward, done, info
