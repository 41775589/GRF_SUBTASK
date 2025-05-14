from . import *
def build_scenario(builder):
    builder.config().game_duration = 500  # Short duration for quick iterations
    builder.config().deterministic = False  # Non-deterministic to mimic real-life variability
    builder.config().offsides = False  # No offsides to focus purely on defending skills
    builder.config().end_episode_on_score = True  # End episode when a goal is scored for rapid feedback
    builder.config().end_episode_on_out_of_play = True  # End on ball out of play to reset quickly
    builder.config().end_episode_on_possession_change = False  # Keep running even if possession changes to increase pressure on defense

    # Set initial ball position near the defending goal to simulate defensive pressure
    builder.SetBallPosition(-0.7, 0.0)

    # Define the left team (training team) with the agent focusing on defensive actions
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper, not controllable
    builder.AddPlayer(-0.8, 0.0, e_PlayerRole_CB, controllable=True)  # Only controlling Center Back to focus on defensive training

    # Define the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper, not controllable
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_CF)  # Centre Forward to challenge the defender directly
    builder.AddPlayer(-0.6, 0.2, e_PlayerRole_CM)  # Midfielder near the side to increase the angle of attack
    builder.AddPlayer(-0.6, -0.2, e_PlayerRole_CM)  # Another midfielder to provide passes and crosses
