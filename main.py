import mathstat as ms
import matplotlib.pyplot as plt


def cdf_uniform(x):
    if x <= 0:
        return 0
    elif 0 < x <= 1:
        return x
    else:
        return 1


s = ms.Sample('uniform', {'a': 0, 'b': 1})
s.generate(10000)
ed = ms.EmpiricalDistribution(s.x)
ed.plot()
print(ed.calc_stat_kolm_smir(func))
