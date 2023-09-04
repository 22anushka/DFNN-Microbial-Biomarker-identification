# taxa mapping

import pandas as pd
import numpy as np

#IBD
df1 = pd.read_csv("/content/gg_13_5_taxonomy.txt", names=['OTU_ID'])  
df1[['OTU_ID','Taxa']] = df1.OTU_ID.str.split("\t",expand=True)

# feature list of subset
df2 = pd.read_csv("/content/features_10%.csv", header=None)  
df = pd.DataFrame({'OTU_ID': [0], 'Taxa': ['0']})
indices = df2.to_numpy()

for i in indices:
  if(df1.isin([str(i[0])]).any().any()):
    
      df_temp = df1.loc[df1['OTU_ID'] == str(i[0])]
      df = df.append(df_temp)
  # elif(df2.isin([i[0]]).any().any()):
  #   df_temp = df2.loc[df2['OTU ID'] == i[0]]
  else:
    print(i[0])

df.to_csv("/content/taxa_mapping_ibd(10%).csv")

