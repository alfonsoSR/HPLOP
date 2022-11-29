from simconfig import Grail
from hplop.core.main import main
from hplop.utils.plot_utils import error, periapsis
from hplop.utils.utils import grail_orbit
from time import perf_counter as timer

case = Grail

prop = main(case)

t0 = timer()

t, s = prop.propagate_orbit()

print(f"Execution time: {timer() - t0}")

periapsis(t, s)


# grail = grail_orbit(t, case)

# error(t, s, grail)
