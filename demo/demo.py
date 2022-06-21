from hplop.core.main import main
from hplop.utils.utils import grail_orbit
from hplop.utils.plot_utils import error
from simconfig import Grail

case = Grail

prop = main(case)

t, s = prop.propagate_orbit()

grail = grail_orbit(t, case)

error(t, s, grail)
