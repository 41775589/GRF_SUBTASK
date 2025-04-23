import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards good passing, dribbling, and transitioning from defense to attack.
    This assumes that actions related to passing and dribbling are captured by specific sticky action indices.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_completion_bonus = 0.05
        self.pass_completion_bonus = 0.05
        self.transition_bonus = 0.1  # Bonus for transitioning from defense to offense

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_completion_reward": [0.0] * len(reward),
            "dribble_completion_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_sticky_actions = o['sticky_actions']
            
            # Assuming index 5 and 7 correspond to "pass" actions
            if any(current_sticky_actions[i] and not self.sticky_actions_counter[i] 
                   for i in [5, 7]):  # Change this according to actual indices
                components["pass_completion_reward"][rew_index] += self.pass_completion_bonus
                reward[rew_index] += self.pass_completion_bonus
            
            # Assuming index 9 corresponds to "dribble" action
            if current_sticky_actions[9] and not self.sticky_actions_counter[9]:
                components["dribble_completion_reward"][rew_index] += self.dribble_completion_bonus
                reward[rew_index] += self.dribble_completion_bonus
                
            # Transition from defense to offense: assuming team 0 is on the left and the scenario shifts flawlessly
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and self.sticky_actions_counter[0]:
                components["transition_reward"][rew_index] += self.transition_bonus
                reward[rew_index] += self.transition_bonus 

            self.sticky_actions_counter = current_sticky_actions.copy()

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act

        return observation, reward, done, info
