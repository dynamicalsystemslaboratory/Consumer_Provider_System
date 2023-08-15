import networkx as nx
import random
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# import csv
from time import sleep, time
from random import randint
# import re
import numpy as np
import random
import heapq

import matplotlib.pyplot as plt
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
    """sigmoid function of the probability that consumers is dissatisfied with the current provider"""
    return Fii/(1+np.exp(-0.08*(x-50)))
    # return 0.5/(1+np.exp(-0.08*(x-50)))*np.random.uniform(0, 0.5)

def prob_choose(Ti,Si,Tj,Sj): 
    """ probability that consumers who have access to P_i and P_j select P_i from both
    Parameters
    ----------
    Ti : number of textual reviews of P_i
      
    Si: online satisfaction rate of P_i

    Returns
    -------
    choose_i: probability to choose P_i by comparing P_i and P_j
    
    """ 
    rho_i = trustworthiness(Ti,Si) 
    rho_j = trustworthiness(Tj,Sj)
    rS_i = rho_i*Si+random.uniform(-0.5,0.5)
    rS_j = rho_j*Sj+random.uniform(-0.5,0.5)
    choose_i = np.exp(rS_i)/(np.exp(rS_i)+np.exp(rS_j))
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
    """find the closest five locations
    Parameters
    ----------
    m : a list of geographic coordinates
    """
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

rate = [1,2,3,4,5]
PorN = 4 ### above 3: positive; below and equal to 3: negative
V = 3000
f = 1
T = 2000
N = 100 
M = 5000
#DD = [1.98, 3.92, 5.82, 7.68, 9.5, 11.28, 13.02, 14.72, 16.38, 18.0]
DD = range(1,11)
# DD = [5]
S = np.zeros((len(DD),N),dtype =float)
X = np.zeros((len(DD),N),dtype =float)
X_frac = np.zeros((len(DD),N),dtype=float)
Text = np.zeros((len(DD),N),dtype =float)
cluster = np.zeros(len(DD),dtype=float)
assort = np.zeros(len(DD),dtype=float)
amplitude_prevalence = np.zeros(len(DD),dtype=float)
Causal_fidelity = np.zeros((len(DD),T),dtype=float)
DISS = np.zeros((len(DD),N),dtype =float) 
Patient_Num_ALL = np.zeros((len(DD),T+1,N),dtype=float)
Patient_Satisfaction_ALL = np.zeros((len(DD),T+1,N),dtype=float)
Quality_Service_ALL = np.zeros((len(DD),T+1,N),dtype=float)  ## quality of service
Degree_Nodes_ALL = np.zeros((len(DD),T+1,N),dtype=float)
Degrees_Graph=np.zeros((len(DD),T+1,N),dtype=float)
Cluster_Coeff_Nodes_ALL = np.zeros((len(DD),T+1,N),dtype=float)
In_Degree_ALL = np.zeros((len(DD),T+1,N),dtype=float)
Out_Degree_ALL = np.zeros((len(DD),T+1,N),dtype=float)
Centrality_In_Graph = np.zeros((len(DD),T+1,N),dtype=float)
Centrality_Out_Graph = np.zeros((len(DD),T+1,N),dtype=float)

# random.seed(100)

# DD = np.linspace(0,1,100)
for n in range(len(DD)):
    dd = DD[n]
    Patient_Num = np.zeros((T+1,N),dtype=float)
    Rij = np.zeros((N,N),dtype=float)
    Rij_initial = np.zeros((N,N),dtype=float)
    # Weighted_Adjacency_Matrix = np.zeros((T+1,N,N),dtype=float)
        
    G = nx.barabasi_albert_graph(N,dd)
    # G = nx.erdos_renyi_graph(N,dd/N)
    doc_pa = {x: [] for x in G.nodes}
    pa_doc = {x: [] for x in range(M)}    
    
    # random.seed(100)
    for i in range(len(pa_doc)):
        start_node = random.sample(list(G.nodes), 1)[0]
        pa_doc[i] = [start_node]
        move_times = 0

        for j in G.neighbors(start_node):
            if (random.uniform(0, 1) > 0.5):
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
            
    # random.seed(50)       
    L_inverse = np.zeros(N,dtype=float) ## L_inverse: inverse of performance of provider
    for d in range(N):
        L_inverse[d] = random.uniform(0,1)
    DISS[n] = L_inverse 
    
    Rtot = random.randint(1,100)
    tn = random.randint(0,Rtot)
    Rp = random.randint(0,Rtot)
    Rating = Rp/Rtot

    Inter_Access = np.empty(N,dtype=object)
    Inter_Access_Num = np.zeros((N,N),dtype=float) ## MFij
    Inter_Access_Fraction = np.zeros((N,N),dtype=float) ## Fij
    a_Rate = np.zeros(N,dtype=object)

    for ia in range(N):
        Inter_Access[ia] = []
    for n1 in range(N):
        for n2 in range(N):
            if n1 != n2:
                inter = Intersection(doc_pa[n1],doc_pa[n2])
                Inter_Access[n1].append(inter)
                Inter_Access_Num[n1,n2] = len(inter)
                Inter_Access_Fraction[n1,n2] = len(inter)/M
            else:
                Inter_Access[n1].append([])
                Inter_Access_Num[n1,n2] = 0
                Inter_Access_Fraction[n1,n2] = 0

    np.save("BA_N{}_M{}_Inter_Access_Num.npy".format(str(N),str(M)),Inter_Access_Num)
    np.save("BA_N{}_M{}_doc_pa.npy".format(str(N),str(M)),doc_pa)
    np.save("BA_N{}_M{}_pa_doc.npy".format(str(N),str(M)),pa_doc)
    Fii = np.array([len(v) for v in doc_pa.values()])
    Fii_hat = Fii-np.sum(Inter_Access_Num,axis=0)
    np.save("BA_N{}_M{}_Fii.npy".format(str(N),str(M)),Fii)
    np.save("BA_N{}_M{}_Fii_hat.npy".format(str(N),str(M)),Fii_hat)
    
    Text_Num = np.empty((T+1,N),dtype=float)
    overall_rate = np.empty((T+1,N), dtype=float)
    positive_review = np.empty((T+1,N),dtype=float)
    total_review = np.empty((T+1,N),dtype=float)
    trust = np.empty((T+1,N),dtype=float) 
    for d in range(N):
        Rtot = random.randint(1,100)
        tn = random.randint(0,Rtot)
        Rp = random.randint(0,Rtot)
        # Rating = (2*Rp-Rtot)/Rtot
        Rating = Rp/Rtot
        total_review[0,d] = Rtot
        Text_Num[0,d] = tn
        positive_review[0,d] = Rp
        overall_rate[0,d] = Rating
        trust[0,d] = trustworthiness(Text_Num[0,d],Rating)



    ####### initial condition: Rij and Xi ########

    initial_pa_doc = {}
    for d in range(N):
        initial_pa_doc[d] = []

    for p in range(M):
        ind_doc = []
        # T_all = []
        # S_all = []
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
                    # Rij_initial[from_doc,choice] = Rij_initial[from_doc,choice] + 1 ## update the initial Rij
                from_doc = from_doc + 1
            Patient_Num[0,choice_doc] = Patient_Num[0,choice_doc] + 1 ## update number of patients of the chosen doctor
            initial_pa_doc[choice_doc].append(p)
        else:
            Ti,Si = Text_Num[0,ind_doc[0]],overall_rate[0,ind_doc[0]]
            Tj,Sj = Text_Num[0,ind_doc[1]],overall_rate[0,ind_doc[1]]
            choose_Pi = prob_choose(Ti,Si,Tj,Sj) 
            choose_Pj = prob_choose(Tj,Sj,Ti,Si)

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

                
    Patient_Num_ALL[n,0,:] = Patient_Num[0,:]
    Patient_Satisfaction_ALL[n,0,:] = overall_rate[0,:]
    
    t = 0 ## time initialize 
    a1 = 0.2 ## stay
    a2 = 0.05 ## leave
    beta = 0.5

    dis = np.vectorize(sig)
    trust = np.vectorize(trustworthiness)
    choose = np.vectorize(_prob_choose)
    access = np.vectorize(_access)


    Rij = np.zeros((N,N),dtype=float) ## residual capacity from i to j
    Aij = np.zeros((N,N),dtype=int) ## edge from i to j

    dissatisfy = np.zeros((T,N),dtype=float)
    leaving_i = np.zeros(N,dtype=float)
    tau = np.zeros((T,N),dtype=float)  ## trustworthiness

    choose_doc = np.zeros(N,dtype=float)
    num_choose_doc = np.zeros((N,N),dtype=float)
    num_leave_doc = np.zeros((N,N),dtype=float)
    for i in range(N):
        for j in range(N):
            Rij[i][j] = Rij_initial[i][j]

    while t < T:
        print(n,t)

        dissatisfy[t] = dis(Patient_Num[t],Fii/M) + np.array(L_inverse) ### probability of dissatisfaction
        dissatisfy[dissatisfy>1] = 1
        Quality_Service_ALL[n,t] = 1 - dissatisfy[t] ## quality of service Q
        leaving_i = np.array([Rij[i]*dissatisfy[t,i] for i in range(N)]) ### leaving_i[i][j] dissatisfied patients with i can move from i to j
        tau[t,:] = trust(Text_Num[t,:],overall_rate[t,:])
        V = np.exp(list(tau[t,:]*overall_rate[t,:]))

        sigma = np.array([np.array([V[j]/(V[i]+V[j]) for j in range(N)]) for i in range(N)])  ### sigma[i][j]: prob. select i given that can access to i and j
        H = np.zeros((N,N),dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j: ### specially, H[i][i] means patients who are not dissatisfied with i but still stay with i
                    H[i][j] = leaving_i[i][j]*sigma[j][i] 

        G = nx.DiGraph(H)
        plt.plot(nx.in_degree_centrality(G).values())
        plt.plot(nx.out_degree_centrality(G).values())
        num_leave_doc = np.array([sum(H[i]) for i in range(N)]) ### num_leave_doc[i]: number of patients leave from i 

        ################# update Rij: residual capacity ######################    
    
        flux_ij = np.array([leaving_i[i]*sigma[:,i] for i in range(N)]) ## flux_ij[i][j]: flux out from i to different j: number of patients move from i to j
        flux_ji = np.array([leaving_i[:,i]*sigma[i,:] for i in range(N)])  ## flux_ji[i][j] flux in from different j to i
        Rij = Rij - flux_ij + flux_ji
        # Rij = Rij - H + H.T

        ############### update number of patients, positive reviews, total reviews, textual reviews, satisfaction rates   ###################
        leaving = np.array([sum(flux_ij[i]) for i in range(N)])
        incoming = np.array([sum(flux_ji[i]) for i in range(N)])
        
        In_Degree_ALL[n,t] = incoming
        Out_Degree_ALL[n,t] = leaving
        Patient_Num[t+1] = Patient_Num[t] - leaving + incoming

        Patient_Num_ALL[n,t+1] = Patient_Num[t+1]
        not_stay = dis(Patient_Num[t+1],Fii/M) + L_inverse
        not_stay[not_stay>1] = 1
        Q = 1-not_stay
        positive_review[t+1] = positive_review[t] + a1*incoming*Q
        total_review[t+1] = total_review[t] + a1*incoming*Q + a2*leaving
        Text_Num[t+1] = Text_Num[t] + beta*(a1*incoming*Q + a2*leaving)

        overall_rate[t+1] = positive_review[t+1]/total_review[t+1]
        Patient_Satisfaction_ALL[n,t+1] = overall_rate[t+1]

        t += 1
    # np.save("BA_N{}_M{}_m{}_Weighted_Adjacency_Matrix.npy".format(str(N),str(M),str(n)),Weighted_Adjacency_Matrix)
np.save("BA_N{}_M{}_m10_Patient_Num_ALL.npy".format(str(N),str(M)),Patient_Num_ALL)
np.save("BA_N{}_M{}_m10_Patient_Satisfaction_ALL.npy".format(str(N),str(M)),Patient_Satisfaction_ALL)
np.save("BA_N{}_M{}_m10_Initial_DISS.npy".format(str(N),str(M)),DISS)
np.save("BA_N{}_M{}_m10_Degree_Nodes_ALL.npy".format(str(N),str(M)),Degree_Nodes_ALL)
np.save("BA_N{}_M{}_m10_Degrees_Graph.npy".format(str(N),str(M)),Degrees_Graph)
np.save("BA_N{}_M{}_m10_Quality_Service_ALL.npy".format(str(N),str(M)),Quality_Service_ALL)
np.save("BA_N{}_M{}_m10_In_Degree_ALL.npy".format(str(N),str(M)),In_Degree_ALL)
np.save("BA_N{}_M{}_m10_Out_Degree_ALL.npy".format(str(N),str(M)),Out_Degree_ALL)
