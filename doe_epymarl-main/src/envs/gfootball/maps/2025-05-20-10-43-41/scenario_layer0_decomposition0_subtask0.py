from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.8, 0.0)  # Setting up a scenario where the ball is close to the offensive zone

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.45, -0.35, e_PlayerRole_LB, controllable=True)  # Left Back as a controllable agent
    builder.AddPlayer(0.45, 0.35, e_PlayerRole_RB, controllable=True)  # Right Back as a controllable agent
    
    # Adding center backs outside of the main training focus for structure
    builder.AddPlayer(0.0, -0.2, e_PlayerRole_CB)
    builder.AddPlayer(0.0, 0.2, e_PlayerRole_CB)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's Goalkeeper
    builder.AddPlayer(0.75, -0.35, e_PlayerRole_LB)  # Opponent's Left Back to interact with our Right Back
    builder.AddPlayer(0.75, 0.35, e_PlayerRole_RB)  # Opponent's Right Back to interact with our Left Back
    builder.AddPlayer(0.3, 0.0, e_PlayerRole_CF)  # Opponent's Center Forward to provide pressure
