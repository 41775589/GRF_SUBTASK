import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This reward wrapper is designed to promote effective dribbling and sprint usage,
    crucial for breaking through defensive lines and maintaining control under pressure."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for dribbles and sprints in some predefined zones
        self.dribble_rewards = np.zeros(2)
        self.past_actions = np.zeros((2, 10), dtype=np.int)  # Track the last 10 actions to judge dribbling consistency
        self.sprint_rewards = np.zeros(2)

    def reset(self):
        self.dribble_rewards.fill(0)
        self.past_actions.fill(0)
        self.sprint_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_rewards'] = self.dribble_rewards
        to_pickle['past_actions'] = self.past_actions
        to_pickle['sprint_rewards'] = self.sprint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_rewards = from_pickle['dribble_rewards']
        self.past_actions = from_pickle['past_actions']
        self.sprint_rewards = from_pickle['sprint_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_reward": [0.0] * len(reward), "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for dribbling should check for consistent dribble actions over several steps.
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'sticky_actions' in o:
                dribble_action_index = 9  # Assuming 9 is the index for "dribble"
                sprint_action_index = 8   # Assuming 8 is the index for "sprint"
                
                # Check for dribbling
                dribble_count = self.past_actions[rew_index].tolist().count(dribble_action_index)
                if o['sticky_actions'][dribble_action_index] == 1:
                    if dribble_count >= 5:  # Reward dribbling if maintained for more than 5 steps
                        components["dribble_reward"][rew_index] = 0.2  
                        
                # Update past actions for dribbling
                self.past_actions[rew_index, :-1] = self.past_actions[rew_index, 1:]
                self.past_actions[rew_index, -1] = dribble_action_index if o['sticky_actions'][dribble_action_index] == 1 else -1

                reward[rew_index] += components["dribble_reward"][rew_index]

                # Check for sprinting while dribbling to overcome defensive pressure
                if o['sticky_actions'][sprint_action_index] == 1 and o['sticky_actions'][dribble_action_index] == 1:
                    components["sprint_reward"][rew_index] = 0.2  # Encouraging sprinting while dribbling
                
                reward[rew_index] += components["sprint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Populate info with updated sticky actions status
        self.past_actions.fill(0)   # Reset past actions at every step
        for agent_idx, agent_obs in enumerate(observation):
            self.past_actions[agent_idx] = agent_obs['sticky_actions']

        return observation, reward, done, info
