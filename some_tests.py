import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
import pandas as pd

speaker = sc.default_speaker()
vec = np.random.rand(10000,1)
plt.plot(vec,label="data",linestyle="--")
rms = np.sqrt(np.square(vec))
plt.plot(rms,label="rms",linestyle="-.")
# plt.show()

# dbfs
dbfs = 20*np.log10(rms)
# plt.plot(dbfs,label="dbfs")
new_dbfs = np.where(dbfs> -10,rms,0.5*rms)
plt.plot(new_dbfs,label="new dbfs",linestyle=":")



plt.legend()
# plt.show()


# speaker.play(new_dbfs,2000)

d = {"file":["hallo"],"label":[2]}
df = pd.DataFrame(d)

new_row = {"filename":"path","label":2}

df = pd.concat([df,pd.DataFrame(new_row)],axis=0)
print(df)