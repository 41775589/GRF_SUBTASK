import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on advanced dribbling techniques, evasion, ball control,
    and the use of 'Sprint' to enhance agents' effectiveness in offensive positions during tight defense.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        # Reward configurations
        self.sprint_reward = 0.1
        self.evasion_reward = 0.2
        self.control_reward = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0] * len(reward),
            "evasion_reward": [0.0] * len(reward),
            "control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward for sprinting
            if o['sticky_actions'][8]:  # Sprint action index is 8
                components["sprint_reward"][i] = self.sprint_reward
                reward[i] += components["sprint_reward"][i]
            
            # Control and evasion rewards
            if o['ball_owned_team'] == 0:  # Assuming 0 is the team of the agent
                current_ball_position = np.array(o['ball'][:2])
                if self.last_ball_position is not None:
                    # Calculate movement
                    movement = np.linalg.norm(current_ball_position - self.last_ball_position)
                    # Control reward
                    components["control_reward"][i] = self.control_reward * movement
                    reward[i] += components["control_reward"][i]
                    
                    # Evasion reward if ball is moved significantly but without opponent intercepting
                    if movement > 0.05 and o['game_mode'] == 0:  # Normal mode, not intercepted
                        components["evasion_reward"][i] = self.evasion_reward
                        reward[i] += components["evasion_reward"][i]
                
                self.last_ball_position = current_ball_position
        
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
