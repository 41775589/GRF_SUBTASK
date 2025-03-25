import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on the stopper role, emphasizing man-marking, blocking, and stalling opponents."""
    
    def __init__(self, env):
        super().__init__(env)
        # Adding a counter for sticky actions (movement and tactical decisions)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocks_performed = 0
        self.interceptions = 0

    def reset(self):
        """Reset the environment and wrapper specific variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocks_performed = 0
        self.interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Allow the current state to be pickled with added environment specifics."""
        to_pickle['blocks_performed'] = self.blocks_performed
        to_pickle['interceptions'] = self.interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from a pickled object."""
        from_pickle = self.env.set_state(state)
        self.blocks_performed = from_pickle['blocks_performed']
        self.interceptions = from_pickle['interceptions']
        return from_pickle

    def reward(self, reward):
        """Compute the dense reward considering defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        # Simplified role focus: index '2' of roles might represent the stopper (depends on exact setup)
        stopper_index = 2  # Example index that needs to be defined based on the environment setup
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward interception of the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == stopper_index:
                if self.interceptions < 5:  # Cap the number of rewarded interceptions to prevent exploitation
                    self.interceptions += 1
                    reward[rew_index] += 1.0  # Increment reward for interception

            # Reward blocks (assumes blocks relate to stopping shots or passes, defined by reduction in ball speed)
            current_speed = np.linalg.norm(o['ball_direction'])
            if current_speed < 0.1:  # Example threshold for "stopped" ball
                if self.blocks_performed < 3:  # Cap the number of rewarded blocks
                    self.blocks_performed += 1
                    reward[rew_index] += 2.0  # Increment for effective block
            components["defensive_reward"][rew_index] = (self.interceptions * 1.0) + (self.blocks_performed * 2.0)

        return reward, components

    def step(self, action):
        """Step environment and modify reward and info with defensive metrics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
