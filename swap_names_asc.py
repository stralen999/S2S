import os
from datetime import datetime

directory_in_str = '/media/prsstorage/INTAKE-Baselines/data/smock_afd2_asc/'



# list dir
directory = os.fsencode(directory_in_str)
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".asc"): 
         # print(os.path.join(directory, filename))
         # change from 2020_09_01_medianmodel.asc
         # to medianmodel_07-Aug-2020.asc
         date_str = filename[:10] # 2020_09_01
         datetime = datetime.strptime(date_str, '%Y_%m_%d')

         new_date_str = datetime.strftime("%d-%b-%Y") # 07-Aug-2020
         new_filename = 'medianmodel_'+new_date_str+'.asc' # medianmodel_07-Aug-2020.asc
         print(filename,new_filename)
         os.rename(directory_in_str+filename, directory_in_str+new_filename)
         continue
     else:
         continue

