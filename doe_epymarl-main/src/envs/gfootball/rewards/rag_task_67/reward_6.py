import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that strengthens skills like passes and dribbles in transitioning phases."""

    def __init__(self, env):
        super().__init__(env)
        # Control how frequently agents attempt important game skills
        self.pass_control_counter = np.zeros(10, dtype=int)  # To monitor successful passes
        self.dribble_control_counter = np.zeros(10, dtype=int)  # To monitor successful dribbles
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions state counter

    def reset(self):
        # Reset counters on new game start
        self.pass_control_counter.fill(0)
        self.dribble_control_counter.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include the current state for rewards processing in saved game states
        to_pickle['checkpoint_reward_wrapper'] = (self.pass_control_counter, self.dribble_control_counter)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_control_counter, self.dribble_control_counter = from_pickle['checkpoint_reward_wrapper']
        return from_pickle

    def reward(self, reward):
        # Extend reward calculation based upon observational data on player control performance
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_control_reward": [0.0] * len(reward),
            "dribble_control_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # Reward for successful passes
                if o['sticky_actions'][7] or o['sticky_actions'][8]:  # Check if 'short' or 'long' pass actions are active
                    components["pass_control_reward"][rew_index] = 0.05  # Reward increment for attempting passes
                    self.pass_control_counter[rew_index] += 1

                # Reward for successful dribbles
                if o['sticky_actions'][9]:  # Check if 'dribble' action is active
                    components["dribble_control_reward"][rew_index] = 0.03  # Reward increment for dribbling
                    self.dribble_control_counter[rew_index] += 1
            
            # Updating the final reward components
            reward[rew_index] += components["pass_control_reward"][rew_index] + components["dribble_control_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Perform an action in the environment
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Append rewards and component details to info for monitoring
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
