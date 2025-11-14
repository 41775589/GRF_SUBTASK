from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting starting position of the ball near the defensive zone
    builder.SetBallPosition(-0.6, 0.0)

    # Set up the Left Team (controlled team with agents)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.8, 0.0, e_PlayerRole_CB)  # Central back controlling the ball
    builder.AddPlayer(-0.8, 0.15, e_PlayerRole_CB)  # Another central back
    
    # Set up the Right Team (opposing team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing goalkeeper
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CF)  # Attacking player one from the opposing team
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CF)  # Attacking player two from the opposing team

    # Set two attacking players close to the controlled central backs to simulate pressure and need for effective dispossessing and safe passing.
