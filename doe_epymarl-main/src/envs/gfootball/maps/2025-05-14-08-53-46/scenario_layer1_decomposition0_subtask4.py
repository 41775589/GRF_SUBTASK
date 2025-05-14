from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = True
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    # Position the ball near the mid-field
    builder.SetBallPosition(0.0, 0.0)
    
    # Configuring the left team (our training side) with specific players to practice defensive maneuvers
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CB)  # Centre Back 1
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CB) # Centre Back 2
    
    # Configuring the right team (opponents)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.2, 0.15, e_PlayerRole_CF) # Center Forward
    builder.AddPlayer(-0.2, -0.15, e_PlayerRole_CF) # Center Forward
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_AM)  # Attacking Midfielder
