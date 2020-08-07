import mathstat as ms
import matplotlib.pyplot as plt

s = ms.Sample('uniform', {'a': 0, 'b': 1})
s.generate(5)
ed = ms.EmpiricalDistribution(s.x)
ed.plot()
