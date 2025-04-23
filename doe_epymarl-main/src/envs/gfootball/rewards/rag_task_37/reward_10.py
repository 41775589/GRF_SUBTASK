import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on ball control and passing skills under pressure in tight game situations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds and relative coefficients for rewarding passing under pressure
        self.pass_pressure_coefficient = 0.05
        self.control_coefficient = 0.1

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the wrapper and its components.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        """
        Restore the state of the wrapper from the saved state.
        """
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        """
        Adjust the reward based on advanced ball control and passing accuracy under pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "pass_quality_reward": [0.0] * len(reward)}
        if not observation:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]

            # Check if the ball is owned by the player's team
            if o['ball_owned_team'] == 0:
                player_index = o['active']
                ball_possession = o['ball_owned_player'] == player_index
                
                # Additional reward for successful passing under pressure
                sticky_actions = o['sticky_actions']
                if ball_possession and any(sticky_actions[1:3]):  # either High Pass or Long Pass active under pressure
                    # Calculate pressure index based on proximity of opposite team players
                    player_pos = o['left_team'][player_index]
                    opponents = o['right_team']
                    distances = np.linalg.norm(opponents - player_pos, axis=1)
                    close_opponents = np.sum(distances < 0.2)  # Count opponents within pressure range
                    if close_opponents > 1:  # More pressure
                        components["pass_quality_reward"][i] = self.pass_pressure_coefficient * close_opponents

                # Reward for maintaining control of the ball under pressure
                if ball_possession and sticky_actions[9]:  # dribbling under pressure
                    components["pass_quality_reward"][i] += self.control_coefficient

            reward[i] += components["pass_quality_reward"][i]

        return reward, components

    def step(self, action):
        """
        Execute an action in the environment and modify the reported rewards and other information.
        """
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
