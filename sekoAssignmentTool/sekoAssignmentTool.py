# -*- coding: utf-8 -*-
"""
The SEKO Assignment: Efficient and Fair Assignment of Students to Multiple Seminars

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  randomly initialize the random number generator with that seed
                        Default: current time stamp
  -i INPUT, --input INPUT
                        Name of the input-file which is a standard excel file
                        Default: input.xlsx
  -o OUTPUT, --output OUTPUT
                        Output of the SEKO assignment will be saved in this OUTPUT excel file. 
                        Default: output.xlsx
  -m MAXIMUM, --maximum MAXIMUM
                        Maximum number of seminars which is assigned to a student.
                        Default: 999
  -v [VERBOSE], --verbose [VERBOSE]
                        Prints the assignment steps verbosely while iterating over the seminars.
                        Default: True

@author: double-blinded

"""

import argparse
from datetime import datetime
import hashlib
import numpy as np
import os.path 

from pandas import read_excel, DataFrame, ExcelWriter
from functools import reduce 
from operator import iconcat
#%% Parse Input Arguments
now = datetime.now() # uses current time stamp for random number generation seed

def str2bool(v): # help function which parses command line inputs for boolean variables (true/false, yes/no, 1/0)
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
# Command line arguments and their description
parser = argparse.ArgumentParser(description="The SEKO Assignment: Efficient and Fair Assignment of Students to Seminars")
parser.add_argument("-s", "--seed", 
                    help="randomly initialize the random number generator with that seed.\n Default: current time stamp",
                    default=now.strftime("%m/%d/%Y, %H:%M:%S"))
parser.add_argument("-i", "--input", 
                    help="Name of the input-file which is a standard excel file. Default: input.xlsx",
                    default="input.xlsx")
parser.add_argument("-o", "--output", 
                    help="Output of the SEKO assignment will be saved in this OUTPUT excel file. Default: output.xlsx",
                    default="output.xlsx")
parser.add_argument("-m", "--maximum", 
                    help="Maximum number of seminars which is assigned to a student. Default: 999",
                    default=999)

parser.add_argument("-v", "--verbose",  type=str2bool, nargs='?',
                        const=True, default=False,
                    help="Prints the assignment steps verbosely while iterating over the seminars. Default: True")

args = parser.parse_args()

# Prints the command line arguments passed by the user to the console
print("================================================================")
print("SEKO assignemnt is initialized with the following parameters ...\n")
print(f'Seed for random number generation: "{args.seed}"')
print(f'Input file: {args.input}')
print(f'Output file: {args.output}')
print(f'Maximal number of seminars per student: {args.maximum}')
print(f'Verbose: {args.verbose}\n')

#%% Help function: The list 'input_list' indicates which seminars are content-wise identical. 
# This information is passed in the input excel file by the user in the row 'content'. 
# The function returns a dictionary with the different seminar types. For each entry 
# (i.e. seminar type) of the dictionary, the list of similar seminars is provided.
def hashSemTypes(input_list):
    semtypes = {}
    for u in np.unique(input_list):
        semtypes[u] = np.argwhere(input_list == u).flatten()
    return semtypes

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

#%% Read and parse input excel file
if not os.path.isfile(args.input):
    raise FileNotFoundError(args.input)

# x: binary variable indicating if a student is registered for a seminar    
# userid: list of student names or ids as provided in the input file; used for output file generation
# seminar_capacity: list of number of seminar places
# seminarNames: list of seminar names as provided in the input file; used for output file generation
# list_seminar_types: list of seminar types to indicate which seminars are content-wise identical 
# dict_semTypes: dictionary of seminar types indicating the list of similar seminars
x, userid, seminar_capacity, seminarNames, list_seminar_types, dict_semTypes = readExcelFile(file=args.input)  


#%% Initialize Random Generator
hash = hashlib.sha256(args.seed.encode('utf-8'))
seed = np.sum(np.frombuffer(hash.digest(), dtype='uint32'))
if args.verbose:
    print(f'Random Number Generator initialized with seed: {seed}')

#%% Assignment of students to requested seminars based on input request matrix
# matrix[user_id, seminar_id] = 1/0 
def assignmentMatrix(matrix,numParticipantsPerSeminar, semTypes, list_seminar_types,                      
                     maxSeminarsAssignedPerParticipant=args.maximum, 
                     at_least_one_seminar_prob_factor = 100.0,
                     seed=seed, addToLowest=1.0, verbose=False):
    
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

#%% We additionally provide a list of waiting places, just in case that a student needs to reject a seminar place
# The waiting places will be assigned in FIFO manner in that case. 
# This uses additionally as input the assignment matrix from the previous step: 'preAssignmentMatrix'. 
# The request matrix is modified and 
# Otherwise the implementations are similar, just without verbose output
def waitingPlacesMatrix(matrix,numParticipantsPerSeminar, semTypes, list_seminar_types, preAssignmentMatrix,                     
                     at_least_one_seminar_prob_factor = 100.0,
                     seed=seed, addToLowest=1.0, verbose=False):
        
    numPreAssignedSeminars = preAssignmentMatrix.sum(axis=1, dtype='int') # number of already assigned seminars 
    
    # The input request matrix is modified. Therefore, we use a copy of the request matrix.
    # Assigned seminars and related requests for seminars of same type are removed from the request matrix. 
    wx = matrix.copy() 
    for key, value in dict_semTypes.items():        
        assType = preAssignmentMatrix[:,value].sum(axis=1) # assignedType
        remUser = assType>0 # remove user request
        for k in value:
            wx[remUser,k] = 0
            
    # initialize random number generator and assignemtn matrix y for waiting places        
    np.random.seed(seed)    
    n, numSeminars = x.shape
    
    y = np.zeros_like(x, dtype='float') # assignment of waiting places
    order = {} # ordered list of students per seminar which are on the waiting list
    
    I = list(range(numSeminars)) # list of seminars which need to be assigned
                    
    while len(I)>0:
        # determine seminar from remaining with minimum number of requests
        r = x[:,I].sum(axis=0) # number of requests r_i for i in I
        w = np.argmax(r) # determine next (most popular) seminar out of I
        i = I.pop(w) # seminar i is returned and removed from list I
        
        registeredUsers = np.squeeze(np.argwhere(x[:,i]>0))
        
        if registeredUsers.size == 0:
            order[ seminarNames[i] ] = []
        elif registeredUsers.size == 1:
            order[ seminarNames[i] ] = [registeredUsers]
        else:
            # compute probabilities for determining the order
            assigned_places_so_far = y[registeredUsers,:].sum(axis=1) + numPreAssignedSeminars[registeredUsers] # here we consider the already assigned seminars
            curMax = np.max(assigned_places_so_far)            
            p = (curMax+addToLowest-assigned_places_so_far)                                
            noseminarsofar = assigned_places_so_far==0
            p[noseminarsofar] *= at_least_one_seminar_prob_factor
            p = p/p.sum()
            
            select = np.random.choice(registeredUsers, replace=False, size=registeredUsers.size, p=p)        
            y[select,i] = (registeredUsers.size-np.arange(registeredUsers.size))/registeredUsers.size
            
            order[ seminarNames[i] ] = userid[select]

    return y, order
#%% Let's do the assignment of students to seminars: The binary assignment matrix is returned.
y = assignmentMatrix(x, seminar_capacity, dict_semTypes, list_seminar_types,                      
                     maxSeminarsAssignedPerParticipant=args.maximum,                      
                     seed=seed, verbose=args.verbose)
#%% Let's compile a list of waiting places: The binary assignment matrix of waiting places and the 'order' of students per seminar is returned.
y_wait, order = waitingPlacesMatrix(x, seminar_capacity, dict_semTypes, list_seminar_types, 
                             preAssignmentMatrix=y, seed=seed, verbose=args.verbose)

#%% Print console output: student view 
if args.verbose: print('\n')
for i in range(len(userid)):    
    tmp = np.squeeze(np.argwhere(y[i,:]>0))
    if tmp.size==1:
        s=seminarNames[tmp]
    elif tmp.size>1:
        s = np.array2string(seminarNames[tmp], separator=', ')
    else:
        s = "-- keine --"
    if args.verbose: print(f'Student {userid[i]} is assigned to seminars: {s}')
    
#%% Print console output: seminar view 
if args.verbose: 
    print('\n')
    for i in range(y.shape[1]):
        tmp = np.squeeze(np.argwhere(y[:,i]>0))
        if tmp.size==1:
            s=userid[tmp]
        elif tmp.size>1:
            s = np.array2string(userid[tmp], separator=', ')
        else:
            s = "-- no --"        
        print(f'Seminar {seminarNames[i]}: {len(tmp)} students assigned, while {np.sum(x[:,i])} registered (max. #places {seminar_capacity[i]}):\n     {s}')
    
#%% Store the data in a list for output in excel file
seminar = []
for i in range(len(seminarNames)):
    tmp = np.squeeze(np.argwhere(y[:,i]>0))
    seminar.append(userid[tmp])
    
n = len(userid) # number of users
k = len(seminarNames) # number of seminars
endLetter = chr(ord('A')+k+1) # column name for the seminar in the excel output

#%% Generate proper Pandas data frame to write the information into excel
df = DataFrame(seminar, index=seminarNames, columns=np.arange(1,y.sum(axis=0).max()+1))    
df.insert(0,'capacity',seminar_capacity)
df.insert(1,'assigned_students',y.sum(axis=0))

num_requests = x.astype('int').sum(axis=0)
df.insert(2,'requests', num_requests)
df.insert(3,'seminar_utilization', y.sum(axis=0)/num_requests)
#%% Output the data to Excel sheets: Seminar sheet and Assignment sheet
writer = ExcelWriter(args.output, engine='xlsxwriter')

red_format = writer.book.add_format({'bg_color': '#FFC7CE','font_color': '#9C0006'}) # output styles
green_format = writer.book.add_format({'bg_color': '#C6EFCE','font_color': '#006100'})

df.to_excel(writer,sheet_name='Seminar', index=True, index_label='Seminar')  # summary sheet: Seminar
df = DataFrame(y, index=userid,  columns=seminarNames)
df.to_excel(writer,sheet_name='Assignment', index=True, index_label='Person') # assignment sheet: binary matrix

#%% Output the data to Excel sheets: Difference sheet 
# 1 in difference sheet means that the requested seminar was assigned to the student
# -1 in difference sheet means that the requested seminar was not assigned to the student
# 0 means that the student did not request the seminar
z = 2*y-x
df = DataFrame(z, index=userid, columns=seminarNames)
df.to_excel(writer,sheet_name='Difference', index=True, index_label='Person')

#%% Output the data to Excel sheets: Stats_Person sheet
v = x.sum(axis=1).astype('int')
z = np.stack((v, y.sum(axis=1), y.sum(axis=1)/v ) )
df = DataFrame(z.T, index=userid, columns=['requested','assigned', 'ratio'])
df.to_excel(writer,sheet_name='Stats_Person', index=True, index_label='Person')

#%% Output the data to Excel sheets: Parameters sheet
z = [f'Seed for random number generation: "{args.seed}"', f'Input file: {args.input}', 
     f'Output file: {args.output}', f'Maximum #Seminare: {args.maximum}', f'Verbose: {args.verbose}'
    ]
df = DataFrame(z, columns=['Parameter'])
df.to_excel(writer,sheet_name='Parameters', index=False)

#%% waiting places
reordered_dict = {k: order[k] for k in seminarNames}
df =  DataFrame.from_dict(reordered_dict, orient='index')
df.to_excel(writer,sheet_name='Waiting_places', index=True)

#%% Let's make the excel sheet nicer with some conditional formatting
writer.sheets['Assignment'].conditional_format(f'A1:{endLetter}{n+2}', {'type':     'cell',
                                    'criteria': 'equal to',
                                    'value':    1,
                                    'format':   green_format})

writer.sheets['Difference'].conditional_format(f'A1:{endLetter}{n+2}', {'type':     'cell',
                                    'criteria': 'equal to',
                                    'value':    -1,
                                    'format':   red_format})
writer.sheets['Difference'].conditional_format(f'A1:{endLetter}{n+2}', {'type':     'cell',
                                    'criteria': 'equal to',
                                    'value':    +1,
                                    'format':   green_format})


writer.sheets['Stats_Person'].conditional_format(f'D1:D{n+2}', {'type': '3_color_scale',
                                         'min_color': "#FF0000",
                                         'mid_color': "#FFFF00",
                                         'max_color': "#00FF00"})

writer.sheets['Stats_Person'].conditional_format(f'C1:C{n+2}', {'type': '3_color_scale',
                                         'min_color': "#FF0000",
                                         'mid_color': "#FFFF00",
                                         'max_color': "#00FF00"})

format_percent = writer.book.add_format({'num_format': '0.0%'})
writer.sheets['Stats_Person'].set_column('D:D', 10, format_percent)


writer.sheets['Waiting_places'].set_column('A:A', 20)
writer.sheets['Seminar'].set_column('A:A', 20)
writer.sheets['Seminar'].set_column('B:B', 10)
writer.sheets['Seminar'].set_column('C:C', 12)
writer.sheets['Seminar'].set_column('E:E', 18, format_percent)
#%% save the output and write it to the file
writer.save()
    

