import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds additional rewards for successfully completed long passes."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_targets = []
        self.pass_count = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.pass_targets = []
        self.pass_count = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        # Calculate reward for successful long passes based on ball possession and trajectory
        for i in range(len(observation)):
            o = observation[i]
            if o["ball_owned_team"] == 1 and o["ball_owned_player"] == o["active"]:
                ball_distance = np.linalg.norm(o["ball"][:2])
                # Check if the distance is long enough for a long pass reward
                if ball_distance > 0.5:
                    self.pass_count += 1
                    components["long_pass_reward"][i] = 0.5  # reward for successful long pass

        # Update the reward list
        reward = [r + components["long_pass_reward"][idx] for idx, r in enumerate(reward)]
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "pass_targets": self.pass_targets,
            "pass_count": self.pass_count
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.pass_targets = saved_state["pass_targets"]
        self.pass_count = saved_state["pass_count"]
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        for i, active in enumerate(self.env.unwrapped.observation()['sticky_actions']):
            self.sticky_actions_counter[i] = active
        return observation, reward, done, info
