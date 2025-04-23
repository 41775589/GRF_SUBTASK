import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward for goalkeeper training, focusing on shot
    stopping, quick reflexes, and initiating counter-attacks with accurate passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation() 
        components = {"base_score_reward": reward.copy(), "goalkeeper_effectiveness": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the player is a goalkeeper (role index 0).
            if (o.get('active') is not None and
                    o['left_team_roles'][o['active']] == 0 and
                    o['ball_owned_team'] == 1):

                # Calculate distance of the ball from the goal (x = -1 is our goal line)
                ball_dist = 1 + o['ball'][0]
                
                # Reward for shot stopping based on proximity of the ball to the goal
                if ball_dist < 0.2:
                    components["goalkeeper_effectiveness"][rew_index] = 1.0

                # Determine if the goalkeeper possesses the ball and checks for kick out (ball_owned_team should change or ball y position should increase)
                if (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and
                        np.abs(o['ball_direction'][0]) > 0.1):
                    
                    # Reward for initiating counter-attacks with an effective pass
                    components["goalkeeper_effectiveness"][rew_index] = 1.0 + 0.5  # Bonus for effective passing

            # Update main reward with new components
            reward[rew_index] += components["goalkeeper_effectiveness"][rew_index]

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
