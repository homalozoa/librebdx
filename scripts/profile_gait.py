import cProfile, pstats
from disneyrepro.gait import sim
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait
setup, style = dino(), dino_gait()
cProfile.run("sim.run(setup, style, 0.2, lambda t: (0,0,0))", "/tmp/gait.prof")
pstats.Stats("/tmp/gait.prof").sort_stats("cumtime").print_stats(25)
