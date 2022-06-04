from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
import numpy as np
import matplotlib.pyplot as plt

from nelson_siegel_svensson.calibrate import calibrate_ns_ols

t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
y = np.array([0.0, 0.011, 0, 0.016, 0.019, 0.021, 0.026, 0.03, 0.035, 0.037, 0.038, 0.04])

# curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau
# assert status.success
# print(curve)
# plt.plot(t, y, "bo")
# plt.plot(t, curve(t), "r")


from nelson_siegel_svensson.calibrate import calibrate_nss_ols

t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
y = np.array([0.0, 0.011, 0, 0.016, 0.019, 0.021, 0.026, 0.03, 0.035, 0.037, 0.038, 0.04])

curve, status = calibrate_nss_ols(t, y)  # starting value of 1.0 for the optimization of tau
assert status.success
print(curve)
T= np.arange(0,30,0.1)
plt.plot(t, y, "bo")
plt.plot(T, curve(T), "r")


from scipy import interpolate
spl = interpolate.CubicSpline(t, y)
plt.plot(T, spl(T), "r")