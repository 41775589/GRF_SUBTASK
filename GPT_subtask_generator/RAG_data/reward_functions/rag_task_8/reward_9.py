import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic reward for quick counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Nothing to load specifically for this wrapper
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Check if we just regained possession
                if not o['right_team_active'][o['active']]:  # I assume right_team is our team
                    # Calculate distance to opponent's goal
                    xpos_ball = o['ball'][0]
                    xpos_goal = -1  # Assuming that player is in the right half and goal is to the left
                    distance_to_goal = np.abs(xpos_goal - xpos_ball)
                    # Counter-attack opportunity: incentivize moving toward the goal
                    # This reward is scaled based on proximity to the goal
                    reward_counter_attack = max(0, 1 - distance_to_goal) * 0.1  # Scale by 0.1 for balance
                    components[f"counter_attack_reward_{rew_index}"] = reward_counter_attack
                    reward[rew_index] += reward_counter_attack
                
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
