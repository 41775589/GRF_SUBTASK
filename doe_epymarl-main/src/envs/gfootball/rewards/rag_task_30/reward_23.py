import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on strategic defensive positioning and transition efficacy."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards = {}
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards = {}
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['transition_rewards'] = self.transition_rewards
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transition_rewards = from_pickle.get('transition_rewards', {})
        self.previous_ball_position = from_pickle.get('previous_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components["transition_reward"] = [0.0] * len(reward)  # Initialize the transition reward component

        for rew_index, o in enumerate(observation):
            # Compute transition rewards based on the ball movement and player positioning
            current_ball_position = np.array(o['ball'][:2])  # We use only x, y coordinates

            if self.previous_ball_position is not None:
                movement_vector = current_ball_position - self.previous_ball_position
                # Reward for quick backward to forward transitions
                if movement_vector[0] > 0.02:
                    components["transition_reward"][rew_index] = 0.1
                    reward[rew_index] += components["transition_reward"][rew_index]
                # Reward for significant lateral movements that disrupt opponent's play
                elif abs(movement_vector[1]) > 0.02:
                    components["transition_reward"][rew_index] = 0.05
                    reward[rew_index] += components["transition_reward"][rew_index]

            # Reward maintaining good defense positioning when not in possession
            if o['ball_owned_team'] != 0:
                player_pos = o['left_team'][o['active']]
                distance_to_ball = np.linalg.norm(player_pos - current_ball_position)
                if distance_to_ball < 0.1:  # close to ball defensively
                    components["transition_reward"][rew_index] += 0.03
                    reward[rew_index] += components["transition_reward"][rew_index]

            self.previous_ball_position = current_ball_position

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
