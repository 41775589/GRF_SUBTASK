from . import *
def build_scenario(builder):
    # Set basic configuration for the game
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Position the ball in a neutral area of the field
    builder.SetBallPosition(0.0, 0.0)

    # Configuring the left team (controlled team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CM)  # Center Midfielder, main agent

    # Configuring the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.7, 0.25, e_PlayerRole_CM)  # Center Midfielder
    builder.AddPlayer(-0.7, -0.25, e_PlayerRole_CM) # Another Center Midfielder

    # Scenario designed to train the agent in holding position and intercepting passes
    # with minimal opposition disruption.
