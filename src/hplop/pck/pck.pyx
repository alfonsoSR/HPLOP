# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

moon = Body("moon", 301, 4902.79996708864, 1738.0)
sun = Body("sun", 10, 132712440041.279419, 0.)
mercury = Body("mercury", 199, 22031.868551, 0.)
venus = Body("venus", 299, 324858.592000, 0.)
earth = Body("earth", 399, 398600.435507, 0.)
mars = Body("mars", 4, 42828.375816, 0.)
jupiter = Body("jupiter", 5, 126712764.100000, 0.)
saturn = Body("saturn", 6, 37940584.841800, 0.)
uranus = Body("uranus", 7, 5794556.400000, 0.)
neptune = Body("neptune", 8, 6836527.100580, 0.)
pluto = Body("pluto", 9, 975.500000, 0.)

def moon_data():
  
  return moon.mu, moon.R
