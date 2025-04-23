import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically for transitioning from defense to attack, focusing on Short Pass, Long Pass, and Dribble skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = "state_information"
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # State restoration specific to this wrapper could be placed here
        return from_pickle

    def reward(self, reward):
        """
        Augments the existing reward signal based on effective control transitions.
        Focused on actions considered as effective transition skills.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        pass_reward = 0.2  # Reward for successful passes under pressure
        dribble_reward = 0.3  # Reward for successful dribbles

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if ('ball_owned_team' in o and 
                 o['ball_owned_team'] == 0):
                # Assuming team 0 is our team, which should match observation setup

                # Get sticky actions for dribbling and identify pass actions
                sticky_actions = o['sticky_actions']

                if sticky_actions[8] == 1:  # action_sprint
                    components['transition_reward'][rew_index] += dribble_reward

                # Check for pass actions; assuming 4: 'action_right' represents a pass
                if (sticky_actions[4] == 1 or sticky_actions[7] == 1):  # action_right, action_left
                    components['transition_reward'][rew_index] += pass_reward

            # Update the reward for this player
            reward[rew_index] += components['transition_reward'][rew_index]

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
