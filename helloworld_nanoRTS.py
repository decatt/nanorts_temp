from nanorts.game_env import GameEnv
from ais.nano_rts_ai import RuleBasedAI
from nanorts.render import Render

if __name__ == "__main__":
    num_envs = 1
    map_paths = ['maps\\16x16\\basesWorkers16x16.xml' for _ in range(num_envs)]
    max_steps=5000
    env = GameEnv(map_paths, max_steps)
    
    width = 16
    height = 16

    ai0 = RuleBasedAI(0, "Random", width, height)

    ai1 = RuleBasedAI(1, "Random", width, height)

    r = Render(width, height)

    for _ in range(100000):
        r.draw(env.games[0])
        game = env.games[0]
        action0 = ai0.get_action(game)
        action1 = ai1.get_action(game)
        action_lists = [[action0, action1]]
        states, rewards, dones, winners = env.step(action_lists)