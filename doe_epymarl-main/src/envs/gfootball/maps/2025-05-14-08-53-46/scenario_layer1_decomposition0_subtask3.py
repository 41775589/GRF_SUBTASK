from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = False
    builder.config().end_episode_on_possession_change = False

    builder.SetBallPosition(0.1, 0.0)  # Start closer to midfield for better movement creation

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    
    # Our trained player (controllable) positioned strategically to exploit open spaces
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CM)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    
    # Adding dummy players as obstacles that don't engage actively
    builder.AddPlayer(-0.7, 0.3, e_PlayerRole_CB, lazy=True)
    builder.AddPlayer(-0.7, -0.3, e_PlayerRole_CB, lazy=True)
    builder.AddPlayer(-0.6, 0.15, e_PlayerRole_CM, lazy=True)
    builder.AddPlayer(-0.6, -0.15, e_PlayerRole_CM, lazy=True)
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_LM, lazy=True)
