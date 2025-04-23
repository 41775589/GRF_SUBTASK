import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic play and ball control reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = {}
        self.ball_control_checkpoint = 0.1 # reward given for maintaining ball control

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_counter'] = self.ball_control_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_counter = from_pickle.get('ball_control_counter', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward maintaining ball under controlled team 
            if o['ball_owned_team'] == 1 and o['right_team_roles'][o['active']] not in [0, 4]: # excluding goalkeeper and defense midfield
                self.ball_control_counter[rew_index] = self.ball_control_counter.get(rew_index, 0) + 1
                if self.ball_control_counter[rew_index] % 10 == 0: # Every 10 steps of control give a reward
                    components["ball_control_reward"][rew_index] = self.ball_control_checkpoint
                    reward[rew_index] += components["ball_control_reward"][rew_index]
            else:
                self.ball_control_counter[rew_index] = 0 # Reset on loss of control

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Keep track of sticky actions
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
