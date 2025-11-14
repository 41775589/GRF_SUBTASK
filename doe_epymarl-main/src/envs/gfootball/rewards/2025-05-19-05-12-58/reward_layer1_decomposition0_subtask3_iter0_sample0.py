import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance learning for strategic passing, utilizing Short Pass and High Pass.
    This helps in creating scoring opportunities by manipulating defensive setups.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for debug purposes.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset the counter on environment reset.
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Placeholder for any state specifics from rewards.
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)  # Load state from the environment's previous state.
        # Load specific state alterations from CheckpointRewardWrapper if any were saved.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Get current observation
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),  # Baseline reward from the environment.
            "pass_quality_reward": [0.0] * len(reward)  # Reward for strategic passing.
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = np.array(o['ball'])  # Ball position
            ball_owned_team = o['ball_owned_team']  # Who owns the ball

            # Calculate reward only when the action by the agent is a pass and their team owns the ball
            if ball_owned_team == 1 and 'active action' in o['sticky_actions'] and (o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1):
                # Calculate the effect of the pass based on the change in position of the ball
                # towards opponents' goal and players' arrangement
                goal_direction = np.array([1, 0])  # Direction towards the opponents' goal
                ball_direction = np.array(o['ball_direction'])
                
                # Normalize vectors to compute dot product
                ball_direction_norm = ball_direction / np.linalg.norm(ball_direction)
                goal_direction_norm = goal_direction / np.linalg.norm(goal_direction)

                alignment = np.dot(ball_direction_norm, goal_direction_norm)
                components["pass_quality_reward"][rew_index] = alignment * 0.1  # Scaling factor for reward

                reward[rew_index] += components["pass_quality_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add additional details about rewards and actions to info for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
