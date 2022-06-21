from hplop.utils.utils import Case
import numpy as np

grail_kernels = [
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/lsk/naif0010.tls"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/pck/pck00009.tpc"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/spk/de421.bsp"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/fk/moon_080317.tf"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/pck/moon_pa_de421_1900_2050.bpc"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0"
     "/grlsp_1000/data/spk/grail_120301_120529_sci_v02.bsp")
]

harmonics_db = ("https://pds-geosciences.wustl.edu/grail/"
                "grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_0900c_sha.tab")

Grail = Case(
    root="grail",
    kernels=grail_kernels,
    initial_epoch="2012-03-06 00:00:00",
    initial_state=np.array(
      [-794.3370097053423, 1127.7972952406371, 1200.2327410752166,
         0.28899406627597957, -1.0622541922252995, 1.1900559677817348]
    ),
    tspan=1.,
    db_name="grgm900c",
    harmonics_db=harmonics_db,
    harmonics_deg=400,
    db_path="databases",
    cartesian=True,
    nu=0.,
    days=True,
    rtol=5e-9,
    atol=5e-9
)
