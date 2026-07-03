import numpy as np
import pandas as pd
import csv

Eliment_names = ['Sb','Cr',  'V','Ti','K', 'F','Li', 'Be', 'B', 'C', 'N', 'Na', 'Mg', 'Al','Si', 'P', 'S','Cl','Ca', 'Sc', # 'O', #'H' 
     'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge','As',
     'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 
     'Rh', 'Pd', 'Ag', 'Cd','In', 'Sn','Te', 'I', 
     'Cs', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Gd', 'Dy', 'Er',
     'Yb', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb', 'Bi', 'Th', 'U']

for eliname in Eliment_names:
        File_name ="Element_localities_"+ eliname + ".csv"
        #columns=["mindat_ID", "Locality_Name", "Minerals_count", "Minerals_list"])
        csvFile = open(File_name, "r",encoding='utf-8')
        df = pd.read_csv(csvFile,header = 0)
        df.info()
        
        species_set = {}
        for row in range(len(df)):
           str_species = df.iloc[row, -1]
           cur_specie = str_species.split(',') 
           if  len(species_set) == 0:
                species_set = set(cur_specie)
           else:
                species_set.update(set(cur_specie))       
        print(species_set)
        l_species = len(species_set)
        print(l_species)
        list_species = list(species_set)   
        #generate a zero matrix
        
        admatrix = np.zeros((l_species,l_species)) 
        for irow in range(len(df)):
           str_species = df.iloc[irow, -1]
           cur_specie = str_species.split(',')      
           if len(cur_specie)!=1:
        #       cur_count = int(df.iloc[irow, -2])
        #       print(cur_count)
        #       if len(cur_specie)== 2:
               for rrow in cur_specie:
                   for jrow in cur_specie:
                       if jrow != rrow:
                              c_idx =  list_species.index(rrow)
                              r_idx =  list_species.index(jrow)
                              #update the matrix 
                              admatrix[c_idx,r_idx] += 1
        list_from = []
        list_to = []
        list_weihgt = []
        
        for ff in range(admatrix.shape[0]):
            for tt in range(admatrix.shape[1]):
                if (tt >= ff) & (admatrix[ff,tt] !=0):
                        list_from.append(list_species[ff])
                        list_to.append(list_species[tt])
                        list_weihgt.append(admatrix[ff,tt])
                
        temp_list_to =  list_to[:] 
        temp_list_from =  list_from[:]  
           
        list_from.extend(temp_list_to)
        list_to.extend(temp_list_from)
        list_weihgt.extend(list_weihgt)
        #output dataframe
        df_output_edges = pd.DataFrame({"from":list_from,"to":list_to,"weight":list_weihgt})
        out_edges_filename  ="loc_edges_"+ eliname +".csv"
        df_output_edges.to_csv(out_edges_filename,index=False) 
        
        array_species = np.array(list_species).reshape((-1,1))
        df_output_nodes = pd.DataFrame(array_species, columns = ["Names"] )
        out_node_filename  ="loc_nodes_"+ eliname +".csv"
        df_output_nodes.to_csv(out_node_filename,index=False) 
               
           
       

    

#1.seperate the Minerals_list and get all the species (print out)
#2.calculate the bi-mineral species counts at the same localities
#3.output the matrix record the count that co-exists.
