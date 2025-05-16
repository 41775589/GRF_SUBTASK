from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    # Set the ball near the defensive zone to simulate ball clearing under pressure
    builder.SetBallPosition(-0.5, 0.0)

    # Configure the left team (team of the trained agent)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.9, 0.0, e_PlayerRole_CB, controllable=True)  # The main player as a CB to focus on defensive passes

    # Configure the right team (opponent team) to apply pressure
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.6, 0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.6, -0.15, e_PlayerRole_CF)
