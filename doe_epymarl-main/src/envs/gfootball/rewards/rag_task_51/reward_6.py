import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the environment's rewards to foster specialized goalkeeper training.
    Rewards are based on shot-stopping, quick reflexes, and initiating counter-attacks.
    """

    def __init__(self, env):
        super().__init__(env)
        # Arrays to keep track of sticky actions for reflections on actions taken
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Track the number of saves by the goalkeeper
        self.num_saves = 0
        # Track successful long passes by the goalkeeper
        self.successful_long_passes = 0
        # Define reward multipliers
        self.save_reward = 2.0
        self.reflex_reward = 0.5
        self.counter_attack_reward = 1.5

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.num_saves = 0
        self.successful_long_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['saved_state'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'num_saves': self.num_saves,
            'successful_long_passes': self.successful_long_passes
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['saved_state']
        self.sticky_actions_counter = saved_state['sticky_actions_counter']
        self.num_saves = saved_state['num_saves']
        self.successful_long_passes = saved_state['successful_long_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0] * len(reward),
                      "reflex_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalkeeper_role = 0  # Assuming '0' is the role index for a goalkeeper
            goalie_index = np.where(o['left_team_roles'] == goalkeeper_role)[0][0]

            if o['active'] == goalie_index:
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == goalie_index:
                    # if the goalie stops a fast-moving ball, increment save counter
                    if np.linalg.norm(o['ball_direction'][:2]) > 0.05:
                        components["save_reward"][rew_index] = self.save_reward
                        self.num_saves += 1
                    
                    # if the goalie makes a quick action (changing sticky actions quickly)
                    components["reflex_reward"][rew_index] = self.reflex_reward * np.sum(o['sticky_actions'][0:3] > 0)
                    
                # if the goalie has just passed the ball long accurately
                if np.linalg.norm(o['ball'][:2] - o['left_team'][goalie_index][:2]) > 0.5 and o['ball_owned_team'] == -1:
                    components["counter_attack_reward"][rew_index] = self.counter_attack_reward
                    self.successful_long_passes += 1

            # Update rewards with additional components
            reward[rew_index] += components["save_reward"][rew_index]
            reward[rew_index] += components["reflex_reward"][rew_index]
            reward[rew_index] += components["counter_attack_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
