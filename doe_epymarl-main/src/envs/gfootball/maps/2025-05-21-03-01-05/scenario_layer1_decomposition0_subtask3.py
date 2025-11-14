from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    builder.SetBallPosition(-0.4, 0.0)  # Ball is set in a central defensive location.

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Left team goalkeeper.
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)  # Centre-back with control over the ball, focusing on passing.

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Right team goalkeeper.
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CB)  # Opposing centre-back to provide a bit of pressure.
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CB) # Another opposing centre-back to simulate game-like pressure.
