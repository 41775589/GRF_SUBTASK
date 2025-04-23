import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a pass-under-pressure reward for mastering short passes while defended."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_count = 0
        self.reward_for_pass = 0.2
        self.penalty_for_loss = -0.1
        self.last_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_count = 0
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'pass_completion_count': self.pass_completion_count,
            'last_ball_owner': self.last_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_vars = from_pickle['CheckpointRewardWrapper']
        self.pass_completion_count = state_vars['pass_completion_count']
        self.last_ball_owner = state_vars['last_ball_owner']
        return from_pickle

    def reward(self, reward):
        # Initialize reward components for each agent
        components = {'base_score_reward': reward.copy(), 'pass_completion_reward': [0.0] * len(reward)}

        # Get the current observation to analyze the game state
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for rew_index, agent_reward in enumerate(reward):
            o = observation[rew_index]

            # Register a successful pass
            if o['ball_owned_player'] != self.last_ball_owner and self.last_ball_owner is not None:
                if o['ball_owned_team'] == 0:  # Assuming '0' is the team index for our agents
                    components['pass_completion_reward'][rew_index] += self.reward_for_pass
                    reward[rew_index] += components['pass_completion_reward'][rew_index]
                    self.pass_completion_count += 1
                else:
                    # Add penalty if the ball is lost to the opponent after being owned
                    components['pass_completion_reward'][rew_index] += self.penalty_for_loss
                    reward[rew_index] += components['pass_completion_reward'][rew_index]

            self.last_ball_owner = o['ball_owned_player']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Information for debugging and tracking performance
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update info with encountered sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
