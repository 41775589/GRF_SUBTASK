import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on enhancing defending strategies, including tackling proficiency,
    efficient movement control, and pressured passing tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter"""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the wrapper along with the environment state"""
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper along with the environment state"""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Reward function focused on defense strategies"""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Initialize reward components
            if 'defensive_reward' not in components:
                components['defensive_reward'] = [0.0] * len(reward)

            # Encourage successful tackling
            if o['ball_owned_team'] == 0:  # ball owned by left team, assume left team is defending
                components['defensive_reward'][rew_index] += 0.1  # reward for having the ball
            
            # Enhance stopping abilities
            if o['sticky_actions'][0] == 1:  # assume index 0 in sticky actions corresponds to "action_stop"
                components['defensive_reward'][rew_index] += 0.05  # reward for executing stop action

            # Reward efficient, pressured passing when under opponent pressure
            if o['game_mode'] in {3, 4, 5}:  # assume these modes involve interaction with opponents (e.g., FreeKick, Corner)
                if o['sticky_actions'][9] == 1:  # assume index 9 in sticky actions corresponds to "action_pass"
                    components['defensive_reward'][rew_index] += 0.1  # additional reward for passing under pressure

            # Combine the rewards
            reward[rew_index] += components['defensive_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Overriding the step function to include reward enhancements"""
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
