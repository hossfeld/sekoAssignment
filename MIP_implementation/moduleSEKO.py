'''
This module implements the optimization problems and the SEKO assignment strategy. It also provides some measures regarding the performance of the assignment.

The assignment of the seminars with respect to the key objectives is formulated by means of integer linear programming (ILP) using the package mip. 

`mip_MaxSeminarUtilization`: The first ILP maximizes the utilization, i.e., the number of assigned places to students, and is referred to as 'MU'. The resulting maximum utilization 
 is then used as constrained in the other ILP problems. 

`mip_AtLeastOneSeminar`: The ILP 'MU+1' provides an assignment which maximizes the ratio of students who are assigned at least one seminar. 
Thereby, the maximum utilization which can be achieved is added as a constraint. 

`mip_MaxFairness`: The third ILP 'MU+F' aims at maximizing fairness while considering the maximum utilization as constraint.
 
 
'''
from mip import Model, xsum, BINARY, maximize, minimize
import numpy as np
import matplotlib.pyplot as plt 

from pandas import read_excel

from functools import reduce 
from operator import iconcat

#%%
def mip_AtLeastOneSeminar(x, numPlaces, semTypes, seed=42, capacity=None, maxSeminarsAssignedPerParticipant=99):
    nUsers, nSeminars = x.shape
    alos = Model() # at least one seminar per participant
    
    y = [[alos.add_var('y({},{})'.format(i, j), var_type=BINARY)
          for j in range(nSeminars)] for i in range(nUsers)]
    
    b = [alos.add_var('b({})'.format(i), var_type=BINARY)
          for i in range(nUsers)]
    
    #% seminar places
    for j in range(nSeminars):
        alos += xsum(y[i][j] for i in range(nUsers)) <= numPlaces[j], 'number seminar places ({})'.format(j)
    
    #% ensure assignment per semType
    for i in range(nUsers):
        for k in semTypes:
            alos += xsum(y[i][j] for j in semTypes[k]) <= 1
    #% assigment fits request
    for i in range(nUsers):
        for j in range(nSeminars):
            alos += y[i][j]<=x[i,j]
    #% signum replacement
    for i in range(nUsers):
        alos += xsum(y[i][j] for j in range(nSeminars)) >= b[i], 'b({}) <= A'.format(j)
        
    # constraint per user: maxSeminarsAssignedPerParticipant
    for i in range(nUsers):
        alos += xsum(y[i][j] for j in range(nSeminars)) <= maxSeminarsAssignedPerParticipant
        
    if capacity is not None:
        alos += xsum(y[i][j] for j in range(nSeminars) for i in range(nUsers)) >= capacity
    
    #%
    alos.objective = maximize(xsum(b[i] for i in range(nUsers)))
    alos.seed = seed
    alos.optimize()
    #%
    res = np.array([[ y[i][j].x
          for j in range(nSeminars)] for i in range(nUsers)], dtype='int')
    return res, alos    

#%%
def mip_MaxSeminarUtilization(x, numPlaces, semTypes, y1=None,  seed=42, maxSeminarsAssignedPerParticipant=99):
    nUsers, nSeminars = x.shape
    alos = Model() # at least one seminar per participant    
    if y1 is None: y1=np.zeros_like(x)
    
    y = [[alos.add_var('y({},{})'.format(i, j), var_type=BINARY)
          for j in range(nSeminars)] for i in range(nUsers)]
    
    #% seminar places
    for j in range(nSeminars):
        alos += xsum(y[i][j] for i in range(nUsers)) <= numPlaces[j], 'number seminar places ({})'.format(j)
    #% assigment fits request
    for i in range(nUsers):
        for j in range(nSeminars):
            alos += y[i][j]<=x[i,j]
    #% ensure assignment per semType
    for i in range(nUsers):
        for k in semTypes:
            alos += xsum(y[i][j] for j in semTypes[k]) <= 1
    
    # constraint per user: maxSeminarsAssignedPerParticipant
    for i in range(nUsers):
        alos += xsum(y[i][j] for j in range(nSeminars)) <= maxSeminarsAssignedPerParticipant
        
    #% ensure assignment from previous step
    nAssignedSeminars = y1.sum(axis=1)
    for i in range(nUsers):
        alos += xsum(y[i][j] for j in range(nSeminars)) >= nAssignedSeminars[i], 'ensure previous assignment user ({})'.format(i)
        #for j in range(nSeminars):
        #    alos += y[i][j]>=y1[i,j]
        
    #%
    alos.objective = maximize(xsum(y[i][j] for i in range(nUsers) for j in range(nSeminars)))
    alos.seed = seed
    alos.optimize()
    #%
    res = np.array([[ y[i][j].x
          for j in range(nSeminars)] for i in range(nUsers)], dtype='int')
    return res, alos
#%%
def mip_MaxFairness(x, numPlaces, semTypes, capacity=None, seed=42, maxSeminarsAssignedPerParticipant=99):
    nUsers, nSeminars = x.shape
    alos = Model() # at least one seminar per participant    
    
    y = [[alos.add_var('y({},{})'.format(i, j), var_type=BINARY)
          for j in range(nSeminars)] for i in range(nUsers)]

    A =  [alos.add_var('A({})'.format(i)) for i in range(nUsers)]    
    #D =  [alos.add_var('D({})'.format(i)) for i in range(nUsers)]    
    T =  [alos.add_var('T({})'.format(i)) for i in range(nUsers)]    
    #% seminar places
    for j in range(nSeminars):
        alos += xsum(y[i][j] for i in range(nUsers)) <= numPlaces[j], 'number seminar places ({})'.format(j)
    #% assigment fits request
    for i in range(nUsers):
        for j in range(nSeminars):
            alos += y[i][j]<=x[i,j]
    #% ensure assignment per semType
    for i in range(nUsers):
        for k in semTypes:
            alos += xsum(y[i][j] for j in semTypes[k]) <= 1
            
    if capacity is not None:
        alos += xsum(y[i][j] for j in range(nSeminars) for i in range(nUsers)) >= capacity
        
    # constraint per user: maxSeminarsAssignedPerParticipant
    for i in range(nUsers):
        alos += xsum(y[i][j] for j in range(nSeminars)) <= maxSeminarsAssignedPerParticipant
        
    #% simplify fairness formulation    
    for i in range(nUsers):
        alos += A[i] == xsum(y[i][j] for j in range(nSeminars)), 'def A({})'.format(i)
        alos += A[i] - capacity/nUsers <= T[i]
        alos += A[i] - capacity/nUsers >= -T[i]
        
    #%
    alos.objective = minimize(xsum(T[i] for i in range(nUsers)) )

    alos.seed = seed
    alos.optimize(max_seconds=10)
    #%
    if alos.num_solutions>0:
        res = np.array([[ y[i][j].x
              for j in range(nSeminars)] for i in range(nUsers)], dtype='int')
    else:
        res = None
    return res, alos  
#%% This function reads the input excel file and stores the information.
# 'file': file name of the input excel file 
# note that the default name is provided by the command line parser object
def readExcelFile(file):    
    WS = read_excel(file, sheet_name='registration') 
    
    # finds the row in the excel file with the corresponding field name and returns the values
    def getValuesForLabel_andRemove(field_names, field):
        s = (WS.iloc[:, 0]).apply(lambda v: v.lower() in field_names)
        res = np.where(s)[0]
        if len(res)==0: # no input line found
            raise ValueError(f'There is no row in the input file for "{field}"!')
        elif len(res)==1: # exactly one input line found
            h = WS.iloc[res[0] ,1:].values
            dropIndex = [res[0]]            
        else: # several input lines found       
            raise ValueError(f'Several rows for "{field}": Row {res+2}!')            
        return h, dropIndex

    # checks if the first field of a row is indicating that this row specifies the seminar capacities
    # accepted names: ['capacity','size']
    seminar_capacity, dropIndex1 = getValuesForLabel_andRemove(field_names=['capacity','size'], field='capacity')      
    
    # checks if the first field of a row is indicating that this row specifies the seminar type
    # accepted names: ['type','seminar_type','content']
    list_seminar_types, dropIndex2 = getValuesForLabel_andRemove(field_names=['type','seminar_type','content'], field='seminar_type')    
    
    # remove both rows from the input data
    w = reduce(iconcat,[dropIndex1, dropIndex2],[])
    WS.drop(WS.index[w], inplace=True)
    
    # student registration for seminars is returned as matrix x    
    matrix = np.array(WS)
    x = matrix[:,1:].astype('int') # binary variable indicating if a student is registered for a seminar
    userid = matrix[:,0] # list of student names or ids as provided in the input file
    seminarNames = WS.columns[1:] # list of seminar names    
    dict_semTypes = hashSemTypes(list_seminar_types)
    
    return x, userid, seminar_capacity, seminarNames, list_seminar_types, dict_semTypes 

#%%
def hashSemTypes(inhaltlich):
    semtypes = {}
    for u in np.unique(inhaltlich):
        semtypes[u] = np.argwhere(inhaltlich == u).flatten()
    return semtypes

#%%
# FIFO assignment
def fifoAssignment(matrix, numPlaces, semTypes, inhaltlich,maxSeminarsAssignedPerParticipant=20):
    x = matrix.copy()
    y = np.zeros_like(x) # assignment
    n, numSeminars = y.shape
    

    
    for i in range(numSeminars):
        registeredUsers = np.squeeze(np.argwhere(x[:,i]>0))
        
        alreadyMaximumAssignedSeminars = np.squeeze(np.argwhere(y.sum(axis=1) >= maxSeminarsAssignedPerParticipant ))
        registeredUsers = np.setdiff1d(registeredUsers, alreadyMaximumAssignedSeminars)[:numPlaces[i]]
        
        y[registeredUsers,i] = 1
        
        for s in semTypes[inhaltlich[i]]: # delete other, content-wise related seminars from user requests
            x[registeredUsers,s] = 0
        
    return y

#%% Assignment of students to requested seminars based on input request matrix
# matrix[user_id, seminar_id] = 1/0 
def assignmentMatrix(matrix, userid, seminarNames, numParticipantsPerSeminar, semTypes, list_seminar_types,                      
                     maxSeminarsAssignedPerParticipant=999, 
                     at_least_one_seminar_prob_factor = 100.0,
                     seed=42, addToLowest=1.0, verbose=False):
    
    # We are changing the input matrix to delete for example if a student is assigned to a seminar of certain seminar type t.
    # Then, the input requests for all other seminars of same type are deleted.
    x = matrix.copy() 
            
    np.random.seed(seed) # initialize the random number generator
    y = np.zeros_like(x) # binary assignment matrix y has same size as binary request matrix x
    n, numSeminars = y.shape # n: number of users; numSeminars: number of seminars
            
    I = list(range(numSeminars)) # list of seminars which need to be assigned
        
    if verbose: probsPerRound = []  # help variable for output in concosle  
        
    while len(I)>0: # iterate over the list of seminars in ascending order of requests: utilization objective
        # determine seminar from remaining with minimum number of requests
        r = x[:,I].sum(axis=0) # number of requests r_i for i in I
        w = np.argmin(r) # determine next (least popular) seminar out of I
        i = I.pop(w) # seminar i is returned and removed from list I
        
        registeredUsers = np.squeeze(np.argwhere(x[:,i]>0)) # which students are registered for this seminar        
        alreadyMaximumAssignedSeminars = np.squeeze(np.argwhere(y.sum(axis=1) >= maxSeminarsAssignedPerParticipant )) # list of students who have been assigned max. #seminars already
        registeredUsers = np.setdiff1d(registeredUsers, alreadyMaximumAssignedSeminars) # now we have a list of eligible students
        
        if len(registeredUsers) <= numParticipantsPerSeminar[i]: # all students can be assigned to that seminar
            y[registeredUsers,i] = 1 # update assignment matrix
            
            for s in semTypes[list_seminar_types[i]]: # delete other, content-wise related seminars from user requests
                x[registeredUsers,s] = 0 # we modify the input request data (copy of the original request matrix)
            if verbose:
                    print(f'seminar {i} ({numParticipantsPerSeminar[i]} places): registrations {len(registeredUsers)}')
                    print(f'  registered students: {registeredUsers}')
                    print('   probability: all students are assigned')
                    probsPerRound.append(([1],['*'],[len(registeredUsers)]))
        else: # there are more requests than seminar places; we have to select students by randomly drawing            
            assigned_places_so_far = y[registeredUsers,:].sum(axis=1) # count the number of assigned seminars for each eligible student
            
            # determine the probabilities per student based on assigned seminars: fairness objective
            curMax = np.max(assigned_places_so_far)            
            p = (curMax+addToLowest-assigned_places_so_far) 
            # take into account ALOS objective
            noseminarsofar = assigned_places_so_far==0
            p[noseminarsofar] *= at_least_one_seminar_prob_factor
            # normalize the numbers to get probabilities
            p = p/p.sum()
            
            # randomly draw students according the 
            select = np.random.choice(registeredUsers, replace=False, size=numParticipantsPerSeminar[i], p=p)
            
            if verbose: # verbose output on console
                bi = np.bincount(assigned_places_so_far)
                probsPerRound.append((np.unique(p)[::-1],np.unique(assigned_places_so_far), bi[bi>0]) )                        
                print(f'seminar {i} ({numParticipantsPerSeminar[i]} places): registrations {len(registeredUsers)}')
                print(f'    registered students: {registeredUsers}')
                for (theuser,theass,thereq,theprob) in zip(registeredUsers, y[registeredUsers,:], matrix[registeredUsers,:],p):
                    print(f'    student {userid[theuser]} has {theass.sum()} seminars assigned: {np.squeeze(np.argwhere(theass))} and requested {np.squeeze(np.argwhere(thereq))}; prob. for next seminar {theprob*100:.2f}%')
                print(f'    Assigned students for seminar {i}: ({len(select)}) students = {np.sort(select)}')
                
            # update the assignment matrix
            y[select,i] = 1
            
            for s in semTypes[list_seminar_types[i]]: # delete other, content-wise related seminars from user requests
                x[select,s] = 0
            
    if verbose: # verbose output on console: summary of assignment
        for k, (probs, assSems, nAss) in enumerate(probsPerRound):
            s = f'Round {k}: Seminar "{seminarNames[k]}" '
            for i,p in enumerate(probs):
                s+=f'p_{assSems[i]}={p*100:.4f}% ({nAss[i]}); '
            print(s)
    return y

#%% Performance of the assignments

# prints the CDF 
def cdfplot(vals, *args,**kwargs):    
    plt.plot(np.sort(vals), (np.arange(1,len(vals)+1))/len(vals), *args, **kwargs )

# computes the utilization of the assignment
def capacityUtilization(yi, numPlaces):
    return yi.sum()/numPlaces.sum()

# computes the ratio of users with at least one seminar assigned
def ratioUserWithAtLeastSeminar(yi):
    return np.sum(yi.sum(axis=1)>0)/yi.shape[0]

# quantifies fairness of the assignment
def fairnessStdAssSems(yi,semTypes):
    z = yi.sum(axis=1).std()    
    return 1-2*z/(len(semTypes))



        