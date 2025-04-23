import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages efficient dribbling, quick feints, and skillful ball handling in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = 0
        self.feint_reward = 0.05
        self.dribble_reward = 0.02
        self.controlled_approach_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Get the latest observations from the environment
        observation = self.env.unwrapped.observation()
        # Initialize components dictionary to store additional reward components
        components = {"base_score_reward": reward.copy(),
                      "feint_bonus": [0.0, 0.0],
                      "dribble_bonus": [0.0, 0.0],
                      "controlled_approach": [0.0, 0.0]}

        # Process each agent's observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Increase control count if the agent owns the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                self.ball_control_counter += 1
            
                # Give feint bonus if dribble or sprint action is toggled on
                if o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1:  # dribble and sprint indices
                    reward[rew_index] += self.feint_reward
                    components["feint_bonus"][rew_index] += self.feint_reward
                
                # Reward for continuous control of the ball
                if self.ball_control_counter >= 3:
                    reward[rew_index] += self.controlled_approach_reward
                    components["controlled_approach"][rew_index] += self.controlled_approach_reward

            # If not controlling the ball, reset control counter
            else:
                self.ball_control_counter = 0

            # Bonus for dribble action activation
            if o['sticky_actions'][9] == 1:  # dribble index
                reward[rew_index] += self.dribble_reward
                components["dribble_bonus"][rew_index] += self.dribble_reward

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
