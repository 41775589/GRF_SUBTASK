import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing precise high passes in a football game."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_executed = False
        self.initial_conditions_met = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_executed = False
        self.initial_conditions_met = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'high_pass_executed': self.high_pass_executed,
            'initial_conditions_met': self.initial_conditions_met
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_pass_executed = from_pickle['CheckpointRewardWrapper']['high_pass_executed']
        self.initial_conditions_met = from_pickle['CheckpointRewardWrapper']['initial_conditions_met']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Check if ball is owned by the active player to verify the initial conditions for a high pass
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                self.initial_conditions_met = True

            # Verify if a high pass action is taken (assumed 6 is the action code for high pass)
            if self.initial_conditions_met and o['sticky_actions'][6] == 1:
                # Reward points for executing a high pass only if no high passes have been executed before
                if not self.high_pass_executed:
                    reward[i] += 2.0  # Bonus for the high pass
                    self.high_pass_executed = True

            # Returning ball ownership back to unowned (simulating end of pass)
            if self.high_pass_executed and o['ball_owned_team'] != 1:
                self.initial_conditions_met = False  # Reset the initial conditions

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
