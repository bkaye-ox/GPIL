# %%
import pandas as pd
# %%
import gpil

# %%
df = pd.read_feather('dp_df.feather')
df.head()

# %%
Y = df['o2pp'].iloc[list(range(1000))]


gpil.init_latent_state(Y=Y, M_dim=4)


