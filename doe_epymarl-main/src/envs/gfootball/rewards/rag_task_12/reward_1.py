import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checkpoint_collected = 0
        self.dribble_checkpoint_collected = 0
        self.sprint_checkpoint_collected = 0

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.pass_checkpoint_collected = 0
        self.dribble_checkpoint_collected = 0
        self.sprint_checkpoint_collected = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle.update({
            "pass_checkpoint_collected": self.pass_checkpoint_collected,
            "dribble_checkpoint_collected": self.dribble_checkpoint_collected,
            "sprint_checkpoint_collected": self.sprint_checkpoint_collected
        })
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_checkpoint_collected = from_pickle["pass_checkpoint_collected"]
        self.dribble_checkpoint_collected = from_pickle["dribble_checkpoint_collected"]
        self.sprint_checkpoint_collected = from_pickle["sprint_checkpoint_collected"]
        return from_pickle

    def reward(self, reward):
        reward, components = super().reward(reward)
        observation = self.env.unwrapped.observation() 
        components["base_score_reward"] = reward.copy() if isinstance(reward, list) else [reward]
        components["pass_reward"] = [0.0]
        components["dribble_reward"] = [0.0]
        components["sprint_reward"] = [0.0]

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            actions = o['sticky_actions']
            # Pass Rewards: actions_index 0 is short pass, 1 is long pass according to typical settings
            if actions[0] or actions[1]:
                if self.pass_checkpoint_collected < 1:
                    components["pass_reward"][rew_index] = 0.1
                    reward[rew_index] += components["pass_reward"][rew_index]
                    self.pass_checkpoint_collected += 1

            # Dribble Rewards
            if actions[9]:  # Index 9 typically corresponds to dribble
                if self.dribble_checkpoint_collected < 1:
                    components["dribble_reward"][rew_index] = 0.1
                    reward[rew_index] += components["dribble_reward"][rew_index]
                    self.dribble_checkpoint_collected += 1

            # Sprint and Stop Sprint Rewards
            if actions[8]:  # Sprint action
                if self.sprint_checkpoint_collected < 1:
                    components["sprint_reward"][rew_index] = 0.05
                    reward[rew_index] += components["sprint_reward"][rew_index]
                    self.sprint_checkpoint_collected += 1
            else:
                self.sprint_checkpoint_collected = 0  # Reset if not sprinting

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
