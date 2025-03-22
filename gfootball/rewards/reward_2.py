import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on offensive strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Track the positions and states of interest
        self.ball_control_checkpoints = None

    def reset(self):
        # Reset the checkpoints state
        self.ball_control_checkpoints = [False] * 5  # Assuming 5 offensive checkpoints
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_checkpoints'] = self.ball_control_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_checkpoints = from_pickle['ball_control_checkpoints']
        return from_pickle

    def reward(self, reward):
        # Fetch the last observation from the environment
        obs = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        # Process each agent's observation
        for agent_index, agent_obs in enumerate(obs):
            # Focus on key offensive capabilities
            if agent_obs['ball_owned_team'] == 1:  # Assuming team 1 is the agent's team
                # Calculate distance to the goal line (x=1)
                ball_pos = agent_obs['ball'][0]
                goal_distance = 1 - ball_pos
                
                # Check dribbling and passing effectiveness
                dribble_action_active = agent_obs['sticky_actions'][9]  # Assuming index 9 is dribbling
                long_pass_effectiveness = (np.abs(agent_obs['ball_direction'][1]) > 0.5 and
                                           agent_obs['ball_owned_player'] == agent_obs['active'])
                
                # Define checkpoint successes
                if dribble_action_active and goal_distance < 0.5 and not self.ball_control_checkpoints[0]:
                    components['offensive_play_reward'][agent_index] += 1.0
                    self.ball_control_checkpoints[0] = True

                if long_pass_effectiveness and goal_distance < 0.3 and not self.ball_control_checkpoints[1]:
                    components['offensive_play_reward'][agent_index] += 1.0
                    self.ball_control_checkpoints[1] = True

            # Calculate the final reward
            reward[agent_index] = components["base_score_reward"][agent_index] + components["offensive_play_reward"][agent_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Aggregate and record the components of the reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
