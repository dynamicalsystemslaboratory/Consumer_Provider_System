
#Import Data structure libraries
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import csv
from ast import literal_eval

#Import libraries for controlling crawling rate
from time import sleep, time
import random
from random import randint

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import Series, DataFrame
import numpy as np




import geopandas as gpd
from matplotlib.patches import Polygon
from shapely.geometry import Polygon
from shapely.geometry import Point, Polygon
import seaborn as sns


from shapely.geometry import Point, Polygon
import seaborn as sns

from scipy import stats
import heapq

# from pysal.explore import esda
from esda.moran import Moran
from splot.esda import moran_scatterplot, lisa_cluster, plot_local_autocorrelation
from libpysal.weights.spatial_lag import lag_spatial
from spreg import OLS
from pysal.explore import esda  # Exploratory Spatial analytics
from pysal.lib import weights  # Spatial weights

# import pysal 
import libpysal as lps
import shapely.geometry as shg

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable


def access_individual(Aij,Cij): ## accessible index of l -> i
    if Aij > 0 and Aij <= Cij:
        aij = 1
    else:
        aij = 0
    if Aij>Cij or Aij<0:
        print(Aij,Cij)
        print("wrong") 
    return aij

def _access(Aij,Cij): ## accessible index of j -> i
    if Aij>0 and Aij<=Cij:
        aij = 1
    else:
        aij = 0
    if Aij>Cij or Aij<0:
        print(Aij,Cij)
        print("wrong")   
    return aij

def update_rate(R_pos,R_neg,In,Out,alpha,beta):
    update_R_pos = R_pos+alpha*In
    update_R_neg = R_neg+beta*Out
    rate = (update_R_pos-update_R_neg)/(update_R_pos+update_R_neg)
    return update_R_pos,update_R_neg,rate

def sig_trust(x):
    return 1/(1+np.exp(-0.01*(x-500)))

def trustworthiness(Ti,Si):
    if Si < 1/2: #bad rating
        tau = 1
    else: #good rating
        tau = sig_trust(Ti) #(0,1)
    return tau

def sig(x,Fii):
    return Fii/(1+np.exp(-0.08*(x-50)))
    # return 0.5/(1+np.exp(-0.08*(x-50)))*np.random.uniform(0, 0.5)

def prob_choose(Ti,Si,Tj,Sj): ### probability that consumers who have access to P_i and P_j select P_i from both 
    rho_i = trustworthiness(Ti,Si) #Ti: number of textual reviews; Si: satisfaction rate 
    rho_j = trustworthiness(Tj,Sj)
    choose_i = np.exp(rho_i*Si)/(np.exp(rho_i*Si)+np.exp(rho_j*Sj))
    return choose_i 

def _prob_choose(rho, R, all):

    choose_i = np.exp(rho*R+0.5*np.random.normal(0, 0.2))/all
    return choose_i

def transition(M,dis,a,prob_choose):
    decide_to_change = sig(M)+dis
    if decide_to_change >= 1:
        decide_to_change = 1
    decide_to_choose = prob_choose
    # print(decide_to_change,decide_to_choose,a)
    tran_prob = a*decide_to_change*decide_to_choose
    # print(tran_prob)
    return tran_prob

def Intersection(A,B):
    ##A, B: list
    return list(set(A).intersection(B))

def distance(lng_A,lat_A,lng_B,lat_B):
    R = 6371.004
    pi = 3.141592654

    Mlng_A = lng_A
    Mlat_A = 90 - lat_A

    Mlng_B = lng_B
    Mlat_B = 90 - lat_B

    C = math.sin(Mlat_A*pi/180)*math.sin(Mlat_B*pi/180)*math.cos((Mlng_A-Mlng_B)*pi/180)+math.cos(Mlat_A*pi/180)*math.cos(Mlat_B*pi/180)
    Distance = R * math.acos(C)

    return Distance

def haversine(coord1, coord2):
    import math
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters)
    km = round(km, 3)
    # print(f"Distance: {meters} m")
    # print(f"Distance: {km} km")
    return meters

def haversine_km(coord1, coord2):
    import math
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters)
    km = round(km, 3)
    # print(f"Distance: {meters} m")
    # print(f"Distance: {km} km")
    return km


def nsmallest(m):
    min_number = heapq.nsmallest(5, m) 
    min_index = []
    for t in min_number:
        index = m.index(t)
        min_index.append(index)
        m[index] = 0
    # print(min_number)
    # print(min_index)
    return min_index
        
def within(A,B):
    checklist = np.zeros(len(B),dtype=float)
    for a in A:
        if a in B:
            index = np.where(B==a)
            checklist[index] = 1
    return list(checklist) 


def EuclideanDistance(x,y):
    import math

    if type(x) is str:
        x = literal_eval(x)
    if type(y) is str:
        y = literal_eval(y)
    distance = math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    return distance

## version 3, each patient can only move 2 times
def version_3(G, M):
  random.seed(100)
  # pa_doc = {x: [] for x in range(M)}
  for i in range(len(pa_doc)):
    start_node = random.sample(list(G.nodes), 1)[0]
    pa_doc[i] = [start_node]
    move_times = 0

    for j in G.neighbors(start_node):
      if (random.uniform(0, 1) > 0.8):
        all_true = True
        v = pa_doc[i][len(pa_doc[i])-1]
        if j in G.neighbors(v):
          pa_doc[i].append(j)
          move_times += 1
          if move_times == 1:
            break

  # doc_pa = {x: [] for x in G.nodes}
  for key, values in pa_doc.items():
    for v in values:
      doc_pa[v].append(key)

rate = [1,2,3,4,5]
PorN = 4 ### above 3: positive; below and equal to 3: negative
V = 3000

def find_element_range(rv,V):
    P = 0
    for i in range(len(rv)):
        if rv[i] >= V:
            P = P+1
    return P

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

def radius_patient_doctor(new_df_zipcode_review,patient_loc,threshold):

    pa_doc_radius = {}
    doc_pa_radius = {}
    for d in range(len(new_df_zipcode_review)):
        doc_pa_radius[d] = []
    for p in range(len(patient_loc)):
        doctor = []
        for d in range(len(new_df_zipcode_review)):
            distance = haversine(literal_eval(new_df_zipcode_review['Geolocation'][d]),patient_loc[p]) 
            if distance <= threshold: # accessiblity
                # doctor.append(doc_manhattan['doctor name'][d])
                doctor.append(d)
            pa_doc_radius.update({p: doctor})
        if len(doctor) >= 5:
            for dd in doctor:
                doc_pa_radius[dd].append(p)
        # print(p,doctor)
        else:
        # if doctor == []:
            all_distance = [haversine(literal_eval(new_df_zipcode_review['Geolocation'][d]),patient_loc[p]) for d in range(len(new_df_zipcode_review))]
            doctor = nsmallest(all_distance)
            pa_doc_radius.update({p:doctor})
            for dd in doctor:
                doc_pa_radius[dd].append(p)

    for d in range(len(new_df_zipcode_review)):
        if len(doc_pa_radius[d]) < 5:
            all_distance = [haversine(patient_loc[p],literal_eval(new_df_zipcode_review['Geolocation'][d])) for p in range(len(patient_loc))]
            patient = nsmallest(all_distance)
            doc_pa_radius.update({d:patient})
            for pp in patient:
                pa_doc_radius[pp].append(d)
                
    return pa_doc_radius


gdfLocations = pd.read_pickle('/Users/tg2426/Documents/Python/FW/Doctor_Network/gdfLocations.csv')
new_df_zipcode_review = pd.read_csv('/Users/tg2426/Documents/Python/FW/Doctor_Network/new_df_zipcode_review.csv')

geo_nyc_merge = pd.read_pickle("/Users/tg2426/Documents/Python/FW/Doctor_Network/geo_nyc_merge.csv")
geoloc_nyc = pd.read_pickle("/Users/tg2426/Documents/Python/FW/Doctor_Network/geoloc_nyc.csv")

################### Doctors in Manhattan ##########################
patient_bounded = []
patient_loc = []
zipcode_population = {}

for i in range(len(geo_nyc_merge)):
    poly = geo_nyc_merge['geometry'][i]
    bound = []
    # colorp = color[int((i + 5)%8)]
    if poly.type == 'MultiPolygon':
        
        for geom in poly.geoms:
            x,y = geom.exterior.xy
            # plt.plot(x,y,alpha=0.5,color = 'grey')
        while True: ##generate patients and distribute uniformly
            g = random.randrange(len(poly.geoms))
            geom_p = poly.geoms[g]
            x1,y1,x2,y2 = geom_p.bounds
            lat = random.uniform(x1, x2)
            lng = random.uniform(y1, y2)
            p = Point(lat,lng)
            if p.within(geom_p):
                # plt.plot(lat,lng,'^',color='green',markersize=10,label='Patients')
                bound.append((lat,lng))
                patient_loc.append((lat,lng))
            population = geo_nyc_merge["POPULATION"][i]
            if len(bound) >= population/500:
                patient_bounded.append(bound)
                break
    elif poly.type == 'Polygon':
        # print('polygon')
        x,y = poly.exterior.xy
        # plt.plot(x,y,alpha=0.5,color='grey')
        x1,y1,x2,y2 = poly.bounds
        while True:
            lat = random.uniform(x1, x2)
            lng = random.uniform(y1, y2)
            p = Point(lat,lng)
            if p.within(poly):
                # ax1 = plt.plot(lat,lng,'^',color='green',markersize=10,label='Patients')
                bound.append((lat,lng))
                patient_loc.append((lat,lng))
            population = geo_nyc_merge["POPULATION"][i]

            if len(bound) >= population/500:
                patient_bounded.append(bound)
                break
for j in range(len(new_df_zipcode_review)):
    loc = literal_eval(new_df_zipcode_review["Geolocation"][j])

K_well_all = {}
K_improved_all = {}
K_improved_keywords = []
K_well_keywords = []
for d in range(len(new_df_zipcode_review)):
    K_improved = {}
    K_well = {}
    for i in range(len(literal_eval(new_df_zipcode_review['What could be improved'][d]))):
        if literal_eval(new_df_zipcode_review['What could be improved'][d])[i] != "N/A":
            text = literal_eval(new_df_zipcode_review['What could be improved'][d])[i].split(" (")[0]
            num = int(literal_eval(new_df_zipcode_review['What could be improved'][d])[i].split(" (")[1].split(")")[0])
            if text == 'Office environment':
                text = 'Office environment was bad'
            K_improved.update({text: num})
            K_improved_keywords.append(text)
        else:
            pass
    K_improved_all.update({d:K_improved})
    for j in range(len(literal_eval(new_df_zipcode_review['What went well'][d]))):
        if literal_eval(new_df_zipcode_review['What went well'][d])[j] != "N/A":
            text = literal_eval(new_df_zipcode_review['What went well'][d])[j].split(" (")[0]
            num = int(literal_eval(new_df_zipcode_review['What went well'][d])[j].split(" (")[1].split(")")[0])
            if text == 'Office environment':
                text = 'Office environment was good'
            K_well.update({text: num})
            K_well_keywords.append(text)
        else:
            pass
    K_well_all.update({d:K_well})
K_improved_keywords = np.unique(K_improved_keywords)
K_well_keywords = np.unique(K_well_keywords)
keywords = []
keywords.extend(K_improved_keywords)
keywords.extend(K_well_keywords)

# print(K_improved_all)

improved_doc = np.zeros((len(new_df_zipcode_review),len(K_improved_keywords)), dtype=float)
for d in range(len(new_df_zipcode_review)):
    for k in range(len(K_improved_keywords)):
        for a in range(len(list(K_improved_all[d].keys()))):
            if list(K_improved_all[d].keys())[a] == K_improved_keywords[k]:
                # print(K_improved_keywords[k])
                improved_doc[d,k] = list(K_improved_all[d].values())[a]
improved_doc[0]

well_doc = np.zeros((len(new_df_zipcode_review),len(K_well_keywords)), dtype=float)
for d in range(len(new_df_zipcode_review)):
    for k in range(len(K_well_keywords)):
        for a in range(len(list(K_well_all[d].keys()))):
            if list(K_well_all[d].keys())[a] == K_well_keywords[k]:
                # print(K_improved_keywords[k])
                well_doc[d,k] = list(K_well_all[d].values())[a]
review_doc = np.zeros((len(new_df_zipcode_review),len(keywords)), dtype=float)
for d in range(len(new_df_zipcode_review)):
    for k1 in range(len(K_improved_keywords)):
        for a in range(len(list(K_improved_all[d].keys()))):
            if list(K_improved_all[d].keys())[a] == K_improved_keywords[k1]:
                # print(K_improved_keywords[k])
                review_doc[d,k1] = list(K_improved_all[d].values())[a]
    for k2 in range(len(K_well_keywords)):
        for b in range(len(list(K_well_all[d].keys()))):
            if list(K_well_all[d].keys())[b] == K_well_keywords[k2]:
                review_doc[d,k1+k2+1] = list(K_well_all[d].values())[b]

L_inverse = [sum(improved_doc[d])/sum(review_doc[d]) for d in range(len(new_df_zipcode_review))]


T = 5000
M = len(patient_loc) ##Population of patients
N = len(new_df_zipcode_review)
Degree_Nodes = np.zeros((10,T+1,N),dtype=float)
M = len(patient_loc) ##Population of patients
N = len(new_df_zipcode_review)
import networkx as nx
for v in [100,200,300,400,500,600,700,800,900]:
    
    pa_doc_radius = radius_patient_doctor(new_df_zipcode_review,patient_loc,v)
    pa_doc = {}
    doc_pa = {}
    for d in range(len(new_df_zipcode_review)):
        doc_pa[d] = []
    # for p in range(len(patient_loc)):
    #     pa_doc_radius[p] = []
    for p in range(len(patient_loc)):
        if np.random.uniform(0,1) <0.5:
            n = 1
        else:
            n = 2
        radius_doctor = random.choices(pa_doc_radius[p], k=n)
        pa_doc[p] = radius_doctor
        for dd in radius_doctor:
            doc_pa[dd].append(p)
    d = 0
    while d<len(new_df_zipcode_review):
        if len(doc_pa[d])<1:
            print(d,doc_pa[d])
            pa_doc = {}
            doc_pa = {}
            for d in range(len(new_df_zipcode_review)):
                doc_pa[d] = []
            # for p in range(len(patient_loc)):
            #     pa_doc_radius[p] = []
            for p in range(len(patient_loc)):
                if np.random.uniform(0,1) < 0.5:
                    n = 1
                else:
                    n = 2
                radius_doctor = random.sample(pa_doc_radius[p], k=n)
                print(radius_doctor)
                pa_doc[p] = radius_doctor
                for dd in radius_doctor:
                    doc_pa[dd].append(p)
            d = 0
        else:
            d = d + 1
        if d == len(new_df_zipcode_review):
            print(v,"pa_doc create!")
    

    Inter_Access = np.empty(N,dtype=object)
    Inter_Access_Num = np.zeros((N,N),dtype=object) ## MFij
    Inter_Access_Fraction = np.zeros((N,N),dtype=object) ## Fij
    a_Rate = np.zeros(N,dtype=object)

    for ia in range(N):
        Inter_Access[ia] = []
    #     Inter_Access_Num[ia] = []
    #     Inter_Access_Fraction[ia] = []
    #     a_Rate[ia] = []
    for n1 in range(N):
        for n2 in range(N):
            if n1 != n2:
                inter = Intersection(doc_pa[n1],doc_pa[n2])
                Inter_Access[n1].append(inter)
                # Inter_Access_Num[n1].append(len(inter))
                # Inter_Access_Fraction[n1].append(len(inter)/M)
                Inter_Access_Num[n1,n2] = len(inter)
                Inter_Access_Fraction[n1,n2] = len(inter)/M
            else:
                Inter_Access[n1].append([])
                Inter_Access_Num[n1,n2] = 0
                Inter_Access_Fraction[n1,n2] = 0
    Fii = np.array([len(vd) for vd in doc_pa.values()])
    Fii_hat = Fii-np.sum(Inter_Access_Num,axis=0)
    np.save("/Users/tg2426/Documents/Python/FW/V{}/Doc_N{}_M{}_Inter_Access_Num_E50.npy".format(str(v),str(N),str(M)),Inter_Access_Num)
    np.save("/Users/tg2426/Documents/Python/FW/V{}/Doc_N{}_M{}_doc_pa.npy".format(str(v),str(N),str(M)),doc_pa)
    np.save("/Users/tg2426/Documents/Python/FW/V{}/Doc_N{}_M{}_pa_doc.npy".format(str(v),str(N),str(M)),pa_doc)
    np.save("/Users/tg2426/Documents/Python/FW/V{}/Doc_N{}_M{}_Fii.npy".format(str(v),str(N),str(M)),Fii)
    np.save("/Users/tg2426/Documents/Python/FW/V{}/Doc_N{}_M{}_Fii_hat.npy".format(str(v),str(N),str(M)),Fii_hat)

    for n in range(1):
        ####### initial condition: Ti, Si ########

        Text_Num = np.zeros((T+1,N),dtype=object)
    # for d in range(len(doc_nyc)):
    #     Text_Num[0,d] = len(literal_eval(doc_nyc['review details'][d]))
        overall_rate = np.empty((T+1,N), dtype=object)
        positive_review = np.empty((T+1,N),dtype=object)
        total_review = np.empty((T+1,N),dtype=object)
        # trust = np.empty((T+1,N),dtype=object) 
        
        for r in range(N):
            Rtot = float(new_df_zipcode_review['number of total review'][r])
            Rp = new_df_zipcode_review['positive review'][r]
            R0 = Rp/Rtot
            overall_rate[0,r] = R0 ## overall rate
            positive_review[0,r] = Rp
            total_review[0,r] = Rtot
            
        Patient_Num = np.zeros((T+1,N),dtype=float)
        Quality_Service = np.zeros((T+1,N),dtype=float)
        In_Degree = np.zeros((T+1,N),dtype=float)
        Out_Degree = np.zeros((T+1,N),dtype=float)
        Rij = np.zeros((N,N),dtype=float)
        Rij_initial = np.zeros((N,N),dtype=float)
        initial_pa_doc = {}
        
        for d in range(N):
            initial_pa_doc[d] = []

        for p in range(M):
            ind_doc = []
            choose_doctor = []
            for p_d in range(len(pa_doc[p])): ## p_d: number of accessible doctors (1 or 2) for patient (p)
                ind = pa_doc[p][p_d]
                ind_doc.append(ind)
                
            if len(ind_doc) == 1 :
                choice_doc = ind_doc[0] ##  only choice of one doctor
                from_doc = 0
                for p_list in Inter_Access[choice_doc]:
                    if p in p_list:
                        Rij_initial[choice_doc,from_doc] = Rij_initial[choice_doc,from_doc] + 1
                    from_doc = from_doc + 1
                Patient_Num[0,choice_doc] = Patient_Num[0,choice_doc] + 1 ## update number of patients of the chosen doctor
                initial_pa_doc[choice_doc].append(p)
            else:
                Ti,Si = Text_Num[0,ind_doc[0]],overall_rate[0,ind_doc[0]]
                Tj,Sj = Text_Num[0,ind_doc[1]],overall_rate[0,ind_doc[1]]
                choose_Pi = prob_choose(Ti,Si,Tj,Sj) 
                choose_Pj = prob_choose(Tj,Sj,Ti,Si)
                # print(choose_Pi+choose_Pj)
                choose_doctor.append(choose_Pi)
                choose_doctor.append(choose_Pj)
                ####### Inverse Transform Sampling Method ##########
                cdf = np.cumsum(choose_doctor)
                a_rnd = np.random.uniform(size=1)[0]
                c = 0
                # print(choose_doctor)
                while c < len(choose_doctor):
                    if a_rnd > cdf[c]:
                        c=c+1
                    else:
                        choice = ind_doc[c] ## which to select / assign a doctor to patient
                        break
                from_doc = 0
                for p_list in Inter_Access[choice]:
                    if p in p_list:
                        Rij_initial[choice,from_doc] = Rij_initial[choice,from_doc] + 1
                        # Rij_initial[from_doc,choice] = Rij_initial[from_doc,choice] + 1 ## update the initial Rij
                    from_doc = from_doc + 1
                Patient_Num[0,choice] = Patient_Num[0,choice] + 1 ## update number of patients of the chosen doctor
                initial_pa_doc[choice].append(p)

            
        ### Vectorize

        t = 0
        a1 = 0.2
        a2 = 0.05
        beta = 0.5
        # a1 = 0.2 ## stay
        # a2 = 0.05 ## leave
        # beta = 0.5

        dis = np.vectorize(sig)
        trust = np.vectorize(trustworthiness)
        choose = np.vectorize(_prob_choose)
        access = np.vectorize(_access)


        Rij = np.zeros((N,N),dtype=float) ## residual capacity from i to j
        Aij = np.zeros((N,N),dtype=int) ## edge from i to j

        dissatisfy = np.zeros((T,N),dtype=float)
        leaving_i = np.zeros(N,dtype=float)
        tau = np.zeros((T,N),dtype=float)

        choose_doc = np.zeros(N,dtype=float)
        num_choose_doc = np.zeros((N,N),dtype=float)
        num_leave_doc = np.zeros((N,N),dtype=float)
        for i in range(N):
            for j in range(N):
                Rij[i][j] = Rij_initial[i][j]

        while t < T:
            print(v,t)
            dissatisfy[t] = dis(Patient_Num[t],Fii/M) + np.array(L_inverse) ### probability of dissatisfaction
            dissatisfy[dissatisfy>1] = 1
            Quality_Service[t] = 1 - dissatisfy[t] ## quality of service Q

            # leaving_i = (Patient_Num[t]-Fii_hat)*dissatisfy[t]
            leaving_i = np.array([Rij[i]*dissatisfy[t,i] for i in range(N)]) ### leaving_i[i][j] dissatisfied patients with i can move from i to j

            tau[t,:] = trust(Text_Num[t,:],overall_rate[t,:])
            V = np.exp(list(tau[t,:]*overall_rate[t,:]))

            sigma = np.array([np.array([V[j]/(V[i]+V[j]) for j in range(N)]) for i in range(N)])  ### sigma[i][j]: prob. select i given that can access to i and j
            H = np.zeros((N,N),dtype=float)
            for i in range(N):
                for j in range(N):
                    if i != j: ### specially, H[i][i] means patients who are not dissatisfied with i but still stay with i
                        H[i][j] = leaving_i[i][j]*sigma[j][i] 
                        # if H[i][j]==0:
                        #     print(leaving_i[i][j],sigma[j][i] )
                        
            num_leave_doc = np.array([sum(H[i]) for i in range(N)]) ### num_leave_doc[i]: number of patients leave from i 

            ################# update Rij: residual capacity ######################    
            flux_ij = np.array([leaving_i[i]*sigma[:,i] for i in range(N)]) ## flux_ij[i][j]: flux out from i to different j: number of patients move from i to j
            flux_ji = np.array([leaving_i[:,i]*sigma[i,:] for i in range(N)])  ## flux_ji[i][j] flux in from different j to i

            leaving = np.array([sum(flux_ij[i]) for i in range(N)])
            incoming = np.array([sum(flux_ji[i]) for i in range(N)])
            
            In_Degree[t] = incoming
            Out_Degree[t] = leaving
            
            for p in range(N):
                if sum(Rij[p]) < leaving[p]:
                    print(sum(Rij[p]),leaving[p])
            Rij = Rij - flux_ij + flux_ji

            ############### update number of patients, positive reviews, total reviews, textual reviews, satisfaction rates   ###################

            Patient_Num[t+1] = Patient_Num[t] - leaving + incoming
            not_stay = dis(Patient_Num[t+1],Fii/M) + L_inverse
            not_stay[not_stay>1] = 1
            Q = 1-not_stay
            positive_review[t+1] = positive_review[t] + a1*incoming*Q
            total_review[t+1] = total_review[t] + a1*incoming*Q + a2*leaving
            Text_Num[t+1] = Text_Num[t] + beta*(a1*incoming*Q + a2*leaving)
            overall_rate[t+1] = positive_review[t+1]/total_review[t+1]
            t += 1

        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Patient_Num_{}_E50.npy".format(str(v),str(n)),Patient_Num)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Patient_Satisfaction_{}_E50.npy".format(str(v),str(n)),overall_rate)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Initial_DISS_{}_E50.npy".format(str(v),str(n)),L_inverse)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Initial_Rij_{}_E50.npy".format(str(v),str(n)),Rij_initial)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Quality_Service_{}_E50.npy".format(str(v),str(n)),Quality_Service)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_In_Degree_{}_E50.npy".format(str(v),str(n)),In_Degree)
        np.save("/Users/tg2426/Documents/Python/FW/V{}/Doctor_Network_Out_Degree_{}_E50.npy".format(str(v),str(n)),Out_Degree)


            # #####