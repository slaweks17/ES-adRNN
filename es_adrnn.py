"""
Created in Sep, 2021, this github version in Jan 2022.
@author: Slawek Smyl
internal version no: 72, for the new test set (with nulls)

The program is meant to save forecasts to a database. Table creation, query and export scripts are listed at the end of this file.
If using ODBC, you also need to create DSN=slawek
The program can also save to a text file, when USE_DB==False, so each of the workers will output a file, 
but you need to create script to average the forecasts and calculate errors. See the query at the end for ideas.

Also, it is meant to be run concurrently, e.g. with 5 workers, e.g. by executing in Windows following script (run5.cmd) 
start /low /min python %1  1
start /low /min python %1  2
start /low /min python %1  3
start /low /min python %1  4
python %1  5
Please note passing worker number on the command line - it is the only parameter needed. 
Example script for Linux (run5) also provided  
"""

USE_DB=False

if USE_DB:
  #an example, in my environments: SQL Server on Windows, MySQL on Mac, and Postgress on Linux
  import platform
  if platform.system()=='Darwin':
    USE_ODBC=True
    USE_POSTGRESS=False
    USE_MySQL=True
  elif platform.system()=='Windows':
    USE_ODBC=True #SQL Server
    USE_POSTGRESS=False
    USE_MySQL=False
  else:
    USE_ODBC=False 
    USE_POSTGRESS=True
  USE_DB=USE_ODBC or USE_POSTGRESS
  USE_ANSI_DRIVER=USE_MySQL
  
  if USE_ODBC:
    import pyodbc
    if USE_MySQL:
      dbConn = pyodbc.connect(r'DSN=slaweka') #has to be ansi driver, slaweka specifies also uid and pwd
    else:
      dbConn = pyodbc.connect(r'DSN=slawek') 
    dbConn.autocommit = False    
    cursor = dbConn.cursor()
    if USE_MySQL:
      cursor.execute("SET wait_timeout=100000")
  elif USE_POSTGRESS:
    import psycopg2
    dbConn=psycopg2.connect(user='slaweks')  #user name
    cursor = dbConn.cursor()
else:
  USE_ODBC=False
  USE_POSTGRESS=False
  USE_MySQL=False    
  OUTPUT_DIR="/output/"
     
from typing import List, Tuple, Optional  
import random
import datetime as dt
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows',50_000)
pd.set_option('display.max_columns', 400)
pd.set_option('display.width',200)
np.set_printoptions(threshold=100_000)

import os
if not 'MKL_NUM_THREADS' in os.environ:
  os.environ['MKL_NUM_THREADS'] = '1'  #conservatively. You should have at least NUM_OF_NETS*MKL_NUM_THREADS cores
  print("MKL_NUM_THREADS not set, setting to 1")
if not 'OMP_NUM_THREADS' in os.environ:  
  os.environ['OMP_NUM_THREADS'] = '1'
  print("OMP_NUM_THREADS not set, setting to 1")
import torch
#from torch_sparse import spmm  #needed for S{1,2}Layers
from torch import Tensor
torch.set_printoptions(threshold=10_000)

print("pytorch version:"+torch.__version__)
print("numpy version:"+np.version.version)
print("curr dir:",os.getcwd())
print('sys.path:',sys.path)
DEBUG_AUTOGRAD_ANOMALIES=False
torch.autograd.set_detect_anomaly(DEBUG_AUTOGRAD_ANOMALIES)

#    72 S3[2],S3[4],S2[7] [0.48,0.035,0.96] LR=3e-3 {5,/3, 6,/10, 7/30} batch=2 {4:5}
RUN='72 S3[2],S2[4],S2[7] [0.485,0.035,0.96] LR=3e-3 {5,/3, 6,/10, 7/30} batch=2 {4:5}'
RUN_SHORT="72" #this will be the subdirectory where to save forecasts if USE_DB==False

NUM_OF_EPOCHS=9
FIRST_EPOCH_TO_START_SAVING_FORECASTS=5
INP_SIZE=None
DISPLAY_SM_WEIGHTS=False
DISPLAY_SM_COEFS=False
STATE_SIZE=100
H_SIZE=40

EPOCH_POW=0.7
NUM_OF_UPDATES_PER_EPOCH=2500
DILATIONS=[[2],[4],[7]] #moving by 1 day
CELLS_NAME=["S3Cell","S2Cell","S2Cell"]
SAVE_TEST_EVERY_STEP=1  
PI_WEIGHT=0.3
INITIAL_LEARNING_RATE=3e-3
NUM_OF_TRAINING_STEPS=50

STEP_SIZE=24
SEASONALITY_IN_DAYS=7
SEASONALITY=STEP_SIZE*SEASONALITY_IN_DAYS #hourly
INPUT_WINDOW=SEASONALITY 

TRAINING_WARMUP_STEPS=3*7
# WARMUP_STEPS are composed of 3 stages: 
# 1. calc initial seasonality coefficients (of size = SEASONALITY), done in Batch
# 2. apply HW using the default perSeriesParams (of size = INPUT_WINDOW)
# 3. apply HW using calculated seasonality smoothing params (the rest, minimum SEASONALITY)
assert TRAINING_WARMUP_STEPS*STEP_SIZE>=SEASONALITY+INPUT_WINDOW+SEASONALITY

TESTING_WARMUP_STEPS=13*7
assert TESTING_WARMUP_STEPS*STEP_SIZE>=3*SEASONALITY

DATES_ENCODE_SIZE=7+31+52 #day of week, day of month, week number. We move by 1 day
DATES_EMBED_SIZE=4
OUTPUT_WINDOW=24 
INPUT_SIZE=INPUT_WINDOW+DATES_EMBED_SIZE+1+OUTPUT_WINDOW #+log(normalizer)+seasonality 
QUANTS=[0.5,0.05,0.95]
TRAINING_QUANTS=[0.485,0.035,0.96]
INITIAL_BATCH_SIZE=2 
BATCH_SIZES={4:5} #at which epoch to change it to what
LEARNING_RATES={5:INITIAL_LEARNING_RATE/3, 6:INITIAL_LEARNING_RATE/10, 7:INITIAL_LEARNING_RATE/30}  #
NUM_OF_QUANTS=len(QUANTS)
TOTAL_OUTPUT_WINDOW=OUTPUT_WINDOW*NUM_OF_QUANTS
TOTAL_OUTPUT_SIZE=TOTAL_OUTPUT_WINDOW+2 #levSm, sSm
NoneT=torch.FloatTensor([-1e38])  #jit does not like Optional etc.
smallNegative=-1e-35
LEV_SM0=-3.5
S_SM0=0.3


#following values ia default, typically overwritten from the command line params
workerNumber=1

interactive=True #default, may be overwritten later, used by trouble() function
def trouble(msg):
  if interactive:
    raise Exception(msg)
  else:
    print(msg)
    import pdb; pdb.set_trace()
    
#################################################################
DATA_DIR="data/"
DATA_PATH=DATA_DIR+'MHL_test_date.csv'
testDates_df=pd.read_csv(DATA_PATH, header=None)
testDates_df.columns
testDates_df.iloc[:,0]
testDates=pd.to_datetime(testDates_df.iloc[:,0])
for i in range(1,len(testDates)):
  diff=testDates[i]-testDates[i-1]
  if diff!=dt.timedelta(hours=1):
    print("WTF")
    break
min(testDates); max(testDates)  #<2018>
len(testDates) #8760
type(testDates);
testDates=testDates.tolist()
len(testDates)
testDates #shows nothing !!!!
testDates[0:10]

DATA_PATH=DATA_DIR+'MHL_train_date.csv'
trainDates_df=pd.read_csv(DATA_PATH, header=None)
trainDates=pd.to_datetime(trainDates_df.iloc[:,0])
for i in range(1,len(trainDates)):
  diff=trainDates[i]-trainDates[i-1]
  if diff!=dt.timedelta(hours=1):
    print("WTF")
    break
min(trainDates); max(trainDates)  #<2016..2017>
trainDates=trainDates.tolist()
len(trainDates)
trainDates[2:5]

DATA_PATH=DATA_DIR+'MHL_train.csv'
train_df=pd.read_csv(DATA_PATH, header=None)
train_df.shape #(17544, 35)
train_df.head(3) 
assert len(trainDates)==len(train_df)
sum(np.isnan(train_df)) #595

DATA_PATH=DATA_DIR+'MHL_test.csv'  #new test
test_df=pd.read_csv(DATA_PATH, header=None)
test_df.shape #(8760, 35)
test_df.head(3) 
assert len(testDates)==len(test_df)
np.where(np.isnan(test_df))
#test_df.iloc[:,6]  #nans
sum(np.isnan(test_df)) #630
len(test_df)

#add to test the warming up area
a_df=train_df.iloc[-TESTING_WARMUP_STEPS*STEP_SIZE:,:]
a_df.shape
type(train_df)
type(a_df)
test1_df=a_df.append(test_df,ignore_index=True)
test1_df.shape
test1_df.head(5)

testDates1=trainDates[-TESTING_WARMUP_STEPS*STEP_SIZE:]+testDates
len(testDates1)
assert len(test1_df)==len(testDates1)
min(testDates1); max(testDates1)

train_np=train_df.to_numpy(np.float32)
sum(sum(np.isnan(train_np))) #8,783
train_np.shape[0]*train_np.shape[1] #614,040 -> 13%

print("num of nans in train:")
icol=0
for icol in range(train_np.shape[1]):
  numOfNans=sum(np.isnan(train_np[:,icol]))
  if numOfNans>0:
    firstNotNan=np.min(np.where(~np.isnan(train_np[:,icol])))
    lastNan=np.max(np.where(np.isnan(train_np[:,icol])))
    print(icol,"numOfNans:",numOfNans,"firstNotNan:",firstNotNan,"lastNan:",lastNan) 
    #1 year missing in the first column. Total 2 years
    

train_t=torch.tensor(train_np)
train_t.shape #torch.Size([17544, 35])
np.nanmin(train_np) #174.0

test_np=test1_df.to_numpy(np.float32)
test_t=torch.tensor(test_np)
test_t.shape #torch.Size([17544, 35])

testDates=testDates1
assert len(testDates)==test_t.shape[0]

print("num of nans in test:")
icol=0
for icol in range(test_np.shape[1]):
  numOfNans=sum(np.isnan(test_np[:,icol]))
  if numOfNans>0:
    firstNotNan=np.min(np.where(~np.isnan(test_np[:,icol])))
    lastNan=np.max(np.where(np.isnan(test_np[:,icol])))
    print(icol,"numOfNans:",numOfNans,"firstNotNan:",firstNotNan,"lastNan:",lastNan) 
#1 year missing in the first column. Total 2 years

maseDenom_l=[]
istep=0
for istep in range(len(train_np)-SEASONALITY):
  diff=train_np[istep+SEASONALITY,]-train_np[istep,]
  maseDenom_l.append(np.abs(diff))
len(maseDenom_l)
maseDenom_a=np.array(maseDenom_l)
maseDenom_a.shape
maseDenom=np.nanmean(maseDenom_a, axis=0)
maseDenom.shape
min(maseDenom)

startingRange=range(TRAINING_WARMUP_STEPS*STEP_SIZE, 
                        len(train_t)-OUTPUT_WINDOW-NUM_OF_TRAINING_STEPS*STEP_SIZE,STEP_SIZE)
startingIndices=list(startingRange)
len(startingIndices)


#lev and seasonality smoothing used only at the seasonality warming area
class PerSeriesParams(torch.nn.Module):
  def __init__(self, series):
    super(PerSeriesParams, self).__init__()

    tep=torch.nn.Parameter(torch.tensor(LEV_SM0))
    self.register_parameter("initLevSm_"+str(series),tep)
    self.initLevSm =tep
    
    tep=torch.nn.Parameter(torch.tensor(S_SM0))
    self.register_parameter("initSSm_"+str(series),tep)
    self.initSSm=tep
      
    


#dat=testDates[2]
#input is a single date
def datesToMetadata(dat): 
  ret=torch.zeros([DATES_ENCODE_SIZE])
  dayOfWeek=dat.weekday() #Monday is 0 and Sunday is 6
  ret[dayOfWeek]=1
  
  dayOfYear=dat.timetuple().tm_yday
  week=min(51,dayOfYear//7)
  ret[7+week]=1
  
  dayOfMonth=dat.day
  ret[7+52+dayOfMonth-1]=1  #Between 1 and the number of days in the given month
  return ret


#batch=[0,2]
class Batch:
  def __init__(self, batch, isTraining=True):
    self.series=batch
    batchSize=len(batch)
    
    if isTraining:
      warmupSteps=TRAINING_WARMUP_STEPS
      startingIndex=random.choice(startingIndices)  
      startupArea=train_t[startingIndex-warmupSteps*STEP_SIZE:startingIndex, batch]
      while torch.any(torch.isnan(startupArea)): 
        startingIndex=random.choice(startingIndices)
        startupArea=train_t[startingIndex-warmupSteps*STEP_SIZE:startingIndex, batch]
        
      reallyStartingIndex=startingIndex-warmupSteps*STEP_SIZE+SEASONALITY
      initialSeasonalityArea=train_t[startingIndex-warmupSteps*STEP_SIZE:
                                   reallyStartingIndex, batch] 
      self.y=train_t[reallyStartingIndex:, batch].t()
      self.dates=trainDates[reallyStartingIndex:]
    else:
      warmupSteps=TESTING_WARMUP_STEPS
      startingIndex=warmupSteps*STEP_SIZE
      initialSeasonalityArea=test_t[0:SEASONALITY, batch] 
      self.y=test_t[SEASONALITY:, batch].t()
      self.dates=testDates[SEASONALITY:]
 
    initialSeasonality_t=initialSeasonalityArea/torch.mean(initialSeasonalityArea, dim=0)
    initialSeasonality=[]
    for ir in range(SEASONALITY):
      initialSeasonality.append(initialSeasonality_t[ir].view([batchSize,1]))  
    
    #and continue calculating levels and seasonality in the main
    self.batchSize=batchSize
    self.maseNormalizer=maseDenom[batch]
    self.initialSeasonality=initialSeasonality

    

#a version of LSTM cell where the output (of size=state_size) is split between h state (of size=h_size) and 
#the real output that goes to the next layer (of size=state_size-h_size)
class LSTM2Cell(torch.nn.Module):
  def __init__(self, input_size, h_size, state_size):
    super(LSTM2Cell, self).__init__()
    self.lxh=torch.nn.Linear(input_size+2*h_size, 4*state_size)  #params of Linear are automatically added to the module params, magically :-)
    self.h_size=h_size
    self.out_size=state_size-h_size

  #jit does not like Optional, so we have to use bool variables and NoneT
  def forward(self, input_t: Tensor, hasDelayedState: bool, hasPrevState : bool,
              prevHState: Tensor=NoneT,
              delayedHstate: Tensor=NoneT,
              cstate: Tensor=NoneT
              ) ->  Tuple[Tensor, Tuple[Tensor, Tensor]]: #outputs: (out, (hState, cState))
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros(input_t.shape[0], 2*self.h_size, dtype=torch.float32)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh(xh)
    
    chunkedGates = torch.chunk(gates,4,dim=1)

    forgetGate = (chunkedGates[0]+1).sigmoid();
    ingate = chunkedGates[1].sigmoid();
    outGate = chunkedGates[2].sigmoid();
    newState = chunkedGates[3].tanh();
    
    if hasPrevState:
      newState = (forgetGate * cstate) + (ingate * newState)
    wholeOutput = outGate * newState.tanh();
    
    output_t, hState =torch.split(wholeOutput, [self.out_size, self.h_size], dim=1)
    return output_t, (hState, newState)
 
    
#Slawek's S3 cel. Weights to inputs
class S3Cell(torch.nn.Module):
  def __init__(self, input_size, h_size, state_size, firstLayer):
    super(S3Cell, self).__init__()
    if firstLayer:
      firstLayerStateSize=input_size+h_size
      self.lxh=torch.nn.Linear(input_size+2*h_size, 4*firstLayerStateSize)  #params of Linear are automatically added to the module params, magically :-)
    else:
      self.lxh=torch.nn.Linear(input_size+2*h_size, 4*state_size)  #params of Linear are automatically added to the module params, magically :-)
    self.lxh2=torch.nn.Linear(input_size+2*h_size, 4*state_size)
    self.h_size=h_size
    self.state_size=state_size
    self.out_size=state_size-h_size

  #jit does not like Optional, so we have to use bool variables and NoneT
  def forward(self, input_t: Tensor, hasDelayedState: bool, hasPrevState : bool,
              prevHState: Tensor=NoneT, delayedHstate : Tensor=NoneT,  
              prevCstate: Tensor=NoneT, delayedCstate: Tensor=NoneT,
              prevHState2: Tensor=NoneT, delayedHstate2 : Tensor=NoneT,  
              prevCstate2: Tensor=NoneT, delayedCstate2: Tensor=NoneT) \
           ->  Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]: #outputs: (out, (hState, cState), (hState2, cState2))
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros(input_t.shape[0], 2*self.h_size, dtype=torch.float)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh(xh)
    chunkedGates = torch.chunk(gates,4,dim=1)  #==torch.split(gates, [self.state_size, self.state_size, self.state_size, self.state_size], dim=1)

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState = forgetGate*weightedCState + (1-forgetGate)*newState;
        
    wholeOutput = outGate * newState
    
    output_t, hState =torch.split(wholeOutput, [wholeOutput.shape[1]-self.h_size, self.h_size], dim=1)

    
    input_t=input_t*torch.exp(output_t)
    prevHState=prevHState2
    prevCstate=prevCstate2
    delayedHstate=delayedHstate2
    delayedCstate=delayedCstate2
    
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros(input_t.shape[0], 2*self.h_size, dtype=torch.float)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh2(xh)
    chunkedGates = torch.chunk(gates,4,dim=1)  #==torch.split(gates, [self.state_size, self.state_size, self.state_size, self.state_size], dim=1)

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState2 = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState2 = forgetGate*weightedCState + (1-forgetGate)*newState2;
        
    wholeOutput = outGate * newState2
    
    output_t, hState2 =torch.split(wholeOutput, [self.out_size, self.h_size], dim=1)
    
    return output_t, (hState, newState), (hState2, newState2)
   

#Slawek's S2 cell: a kind of mix of GRU and LSTM. Also splitting ouput into h and the "real output"
class S2Cell(torch.nn.Module):
  def __init__(self, input_size, h_size, state_size):
    super(S2Cell, self).__init__()
    self.lxh=torch.nn.Linear(input_size+2*h_size, 4*state_size)  #params of Linear are automatically added to the module params, magically :-)
    self.h_size=h_size
    self.state_size=state_size
    self.out_size=state_size-h_size

  #jit does not like Optional, so we have to use bool variables and NoneT
  def forward(self, input_t: Tensor, hasDelayedState: bool, hasPrevState : bool,
               prevHState: Tensor=NoneT,
               delayedHstate : Tensor=NoneT,  
               prevCstate: Tensor=NoneT, 
               delayedCstate: Tensor=NoneT) \
           ->  Tuple[Tensor, Tuple[Tensor, Tensor]]: #outputs: (out, (hState, cState))
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros(input_t.shape[0], 2*self.h_size, dtype=torch.float)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh(xh)
    chunkedGates = torch.chunk(gates,4,dim=1)  #==torch.split(gates, [self.state_size, self.state_size, self.state_size, self.state_size], dim=1)

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState = forgetGate*weightedCState + (1-forgetGate)*newState;
        
    wholeOutput = outGate * newState
    
    output_t, hState =torch.split(wholeOutput, [self.out_size, self.h_size], dim=1)
    return output_t, (hState, newState)
    
  
  
class DilatedSparseRnnStack(torch.nn.Module):
  def resetState(self):
    self.hStates=[]  #first index time, second layers
    self.cStates=[]
    if "S3Cell" in self.cellNames:
      self.hStates2=[]  #first index time, second layers
      self.cStates2=[]
      
  #dilations are like [[1,3,7]] this defines 3 layers + output adaptor layer
  #or [[1,3],[6,12]] - this defines 2 blocks of 2 layers each + output adaptor layer, with a resNet-style shortcut between output of the first block (output of the second layer)
  #and output of the second block (output of 4th layer). 
  def __init__(self, dilations, cellNames, input_size, state_size, output_size, h_size=None, inp_size=None):
    super(DilatedSparseRnnStack, self).__init__()
    numOfBlocks=len(dilations)
    self.dilations=dilations
    self.cellNames=cellNames
    self.input_size=input_size
    self.h_size=h_size
    self.output_size=output_size
    
    out_sizes=[]
    for cellName in cellNames:
      if cellName!="LSTM":
        out_size=state_size-h_size
      else:
        out_size=state_size
      out_sizes.append(out_size)
      
    if not inp_size is None:
      self.inputAdaptor = torch.nn.Linear(input_size, inp_size)
      self.inp_size=inp_size
    else:
      self.inputAdaptor=None
    
        
    self.cells = []
    layer=0; iblock=0; 
    for iblock in range(numOfBlocks):
      for lay in range(len(dilations[iblock])):
        cellName=cellNames[layer]
        if lay==0 and iblock==0:
          firstLayer=True
          if inp_size is None:
            inputSize=input_size
          else:
            inputSize=inp_size
        else:
          inputSize=out_sizes[layer]
          firstLayer=False
          
        if cellName=="LSTM2Cell":
          if interactive:
            cell = LSTM2Cell(inputSize, h_size, state_size)
          else:
            cell = torch.jit.script(LSTM2Cell(inputSize, h_size, state_size)) 
        elif cellName=="S2Cell":
          if interactive:
            cell = S2Cell(inputSize, h_size, state_size)
          else:
            cell = torch.jit.script(S2Cell(inputSize, h_size, state_size))
        elif cellName=="S3Cell":
          if interactive:
            cell = S3Cell(inputSize, h_size, state_size, firstLayer)
          else:
            cell = torch.jit.script(S3Cell(inputSize, h_size, state_size, firstLayer))
        else:
          cell = torch.nn.LSTMCell(inputSize, state_size)
        #print("adding","Cell_{}".format(layer))
        self.add_module("Cell_{}".format(layer), cell)
        self.cells.append(cell)
        layer+=1
    
      self.adaptor = torch.nn.Linear(out_size, output_size)         
      
    self.numOfBlocks=numOfBlocks  
    self.out_size=out_size
    self.resetState()
    
      
  def forward(self, input_t):
    prevBlockOut=torch.zeros([input_t.shape[0], self.out_size])
    self.hStates.append([]) #append for the new t
    self.cStates.append([])
    if "S3Cell" in self.cellNames:
      self.hStates2.append([]) #append for the new t
      self.cStates2.append([])
    t=len(self.hStates)-1
    hasPrevState=t>0
        
    layer=0
    for iblock in range(self.numOfBlocks):
      for lay in range(len(self.dilations[iblock])):
        #print('layer=',layer)
        cellName=self.cellNames[layer]
        if lay==0:
          if iblock==0:
              if not self.inputAdaptor is None:
                input=self.inputAdaptor(input_t)
              else:
                input=input_t
          else:
              input=prevBlockOut #shortcut
        else:
            input=output_t 
        
        ti_1=t-self.dilations[iblock][lay]
        hasDelayedState=ti_1>=0
        if cellName=="S2Cell":
          if hasDelayedState:
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer], delayedHstate=self.hStates[ti_1][layer], 
              prevCstate=self.cStates[t-1][layer], delayedCstate=self.cStates[ti_1][layer])
          elif hasPrevState:   
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],
              prevCstate=self.cStates[t-1][layer])
          else:
            output_t, (hState, newState)=self.cells[layer](input, False, False) 
        elif cellName=="S3Cell":
          if hasDelayedState:
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer], delayedHstate=self.hStates[ti_1][layer], 
              prevCstate=self.cStates[t-1][layer], delayedCstate=self.cStates[ti_1][layer],
              prevHState2=self.hStates2[t-1][layer], delayedHstate2=self.hStates2[ti_1][layer], 
              prevCstate2=self.cStates2[t-1][layer], delayedCstate2=self.cStates2[ti_1][layer])
          elif hasPrevState:   
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],prevCstate=self.cStates[t-1][layer],
              prevHState2=self.hStates2[t-1][layer],prevCstate2=self.cStates2[t-1][layer])
          else:
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, False, False) 
        elif cellName=="LSTM2Cell":
          if hasDelayedState:
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer], delayedHstate=self.hStates[ti_1][layer], 
              cstate=self.cStates[ti_1][layer])
          elif hasPrevState:
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],                                
              cstate=self.cStates[t-1][layer])
          else:
            output_t, (hState, newState)=self.cells[layer](input, False, False) 
        else: #LSTM 
          if hasDelayedState:
            hState, newState=self.cells[layer](input, (self.hStates[ti_1][layer], self.cStates[ti_1][layer]))
          elif hasPrevState:
            hState, newState=self.cells[layer](input, (self.hStates[t-1][layer], self.cStates[t-1][layer]))
          else:
            hState, newState=self.cells[layer](input) 
          output_t=hState
            
        self.hStates[t].append(hState)
        self.cStates[t].append(newState)
        if cellName=="S3Cell":
          self.hStates2[t].append(hState2)
          self.cStates2[t].append(newState2)
        
        layer+=1
      prevBlockOut=output_t+prevBlockOut
      
      output_t = self.adaptor(prevBlockOut)
    return output_t                    
    
                
            
#pinball
#forec_t= output of NN
#actuals_t are the original (although potentially shifted) actuals
#output list(NUM_OF_QANTS) of batchSize
def trainingLossFunc(forec_t, actuals_t, anchorLevel):
  if forec_t.shape[0] != actuals_t.shape[0]:
    trouble("forec_t.shape[0] != actuals_t.shape[0]")

  if forec_t.shape[1] != TOTAL_OUTPUT_WINDOW:
    trouble("forec_t.shape[1] != TOTAL_OUTPUT_WINDOW")
  
  if actuals_t.shape[1] != OUTPUT_WINDOW:
    trouble("actuals_t.shape[1] != OUTPUT_WINDOW")
   
  ret=[]

  nans=torch.isnan(actuals_t).detach() | (actuals_t<=0).detach()
  notNans=(~nans).float()
  numOfNotNans=notNans.sum(dim=1)

  if torch.any(nans):
    actuals_t[nans]=1 #actuals have been cloned outside of this function

  #we do it here, becasue Pytorch is alergic to any opearation including nans, even if removed from the graph later
  #so we first patch nans and execute normalization and squashing and then remove results involving nans
  actualsS_t=actuals_t/anchorLevel

  lower=0; iq=0
  for iq in range(len(TRAINING_QUANTS)):
    quant=TRAINING_QUANTS[iq]
    upper=lower+OUTPUT_WINDOW
    diff=actualsS_t-forec_t[:,lower:upper] #normalized and squashed
    rs=torch.max(diff*quant, diff*(quant-1))
    rs[nans] = 0

    if torch.any(numOfNotNans==0):
      for ib in range(len(numOfNotNans)):
        if numOfNotNans[ib]==0:
          numOfNotNans[ib]+=1

    rc=rs.sum(dim=1)/numOfNotNans #numOfNotNans is vector
    
    if iq==0:
      ret.append(rc)
    else:
      ret.append(rc*PI_WEIGHT)
    lower=upper
  return ret
            
            
#This function is of low importance, its only use is displaying current worker validation results, for quick feedback
# RMSE, bias, MASE, MAPE, pinball loss,  % of exceedance. Return dimenions =[batchSize,3+len(QUANTS)]
# operating on numpy arrays, not tensors
#maseNormalizer=ppBatch.maseNormalizer
def validationLossFunc(forec, actuals, maseNormalizer): 
  if np.isnan(forec.data).sum()>0:
    trouble("NaNs in forecast")
    
  if forec.shape[0] != actuals.shape[0]:
    trouble("forec.shape[0] != actuals.shape[0]")
  
  if forec.shape[1] != TOTAL_OUTPUT_WINDOW:
    trouble("forec.shape[1] != TOTAL_OUTPUT_WINDOW")
  
  if actuals.shape[1] != OUTPUT_WINDOW:  #but they may be all NANs
    trouble("actuals.shape[1] != OUTPUT_WINDOW")
    
  ret=np.zeros([actuals.shape[0],4+len(QUANTS)], dtype=np.float32)+np.nan
  
  #center
  diff=forec[:,0:OUTPUT_WINDOW]-actuals
  rmse=np.sqrt(np.nanmean(diff*diff, axis=1))
  mase=np.nanmean(abs(diff), axis=1)/maseNormalizer
  mape=np.nanmean(abs(diff/actuals), axis=1)
  bias=np.nanmean(diff/actuals, axis=1)
  
  ret[:,0]=rmse
  ret[:,1]=bias
  ret[:,2]=mase
  ret[:,3]=mape
  
  #exceedance
  lower=0; iq=0
  for iq in range(len(QUANTS)):
    quant=QUANTS[iq]
    #print(quant)
    upper=lower+OUTPUT_WINDOW
    diff=actuals-forec[:,lower:upper]
    
    if quant>=0.5:
      xceeded=diff>0
    else:
      xceeded=diff<0
        
    exceeded=np.nanmean(xceeded, axis=1) 
    ret[:,iq+4]=exceeded
    lower=upper
      
  return ret
            



if __name__ == '__main__' or __name__ == 'builtins':
  print(RUN)
  if len(sys.argv)==5:
    print("assuming running from within Eclipse and assuming default params")
    interactive=True
  elif len(sys.argv)==2:
    workerNumber=int(sys.argv[1])
    print('workerNumber:',workerNumber)
    interactive=False
  elif len(sys.argv)==1:
    print("assuming default params")
    interactive=False
  else:
    print("you need to specify 1 param: workerNumber")
    exit(-1)
  
  if not USE_DB:
    outputDir=OUTPUT_DIR+RUN_SHORT+"/"
    if workerNumber==1:
      dirExists = os.path.exists(outputDir)
      if not dirExists:
        os.makedirs(outputDir)
  
  series_list=list(range(train_t.shape[1]))
  numSeries=len(series_list)
        
  if USE_ODBC:   
    MODEL_INSERT_QUERY = "insert into electra4Models(run, workerNo, dateTimeOfPrediction) \
      values(?,?,?)"
      
    INSERT_QUERY = "insert into electra5(dateTimeOfPrediction, workerNo, epoch, forecOriginDate, series "
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+="\n, actual"+str(ih+1) + ", predQ50_"+str(ih+1) + ", predQ05_"+str(ih+1)+ ", predQ95_"+str(ih+1) 
    INSERT_QUERY+=")\n"
    INSERT_QUERY+="values (?,?,?,?,?"
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+=",?,?,?,?";
    INSERT_QUERY+=")"    
  elif USE_POSTGRESS:
    MODEL_INSERT_QUERY = "insert into electra4Models(run, workerNo, dateTimeOfPrediction) \
      values(%s,%s,%s)"
      
    INSERT_QUERY = "insert into electra5(dateTimeOfPrediction, workerNo, epoch, forecOriginDate, series "
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+="\n, actual"+str(ih+1) + ", predQ50_"+str(ih+1) + ", predQ05_"+str(ih+1) + ", predQ95_"+str(ih+1) 
    INSERT_QUERY+=")\n"
    INSERT_QUERY+="values (%s,%s,%s,%s,%s"
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+=",%s,%s,%s,%s";
    INSERT_QUERY+=")" 


  now=dt.datetime.now()
  if USE_DB:
    if USE_ANSI_DRIVER:
      theVals=(bytearray(RUN, encoding='utf-8'), workerNumber, now)
    else:            
      theVals=(RUN, workerNumber, now)  
    cursor.execute(MODEL_INSERT_QUERY,theVals)  
    
  perSeriesParams=[]; perSeriesTrainers=[]
  for series in series_list:
    perSerPars=PerSeriesParams(series)
    perSeriesParams.append(perSerPars)
    perSerTrainer=torch.optim.Adam(perSerPars.parameters(), lr=INITIAL_LEARNING_RATE)
    perSeriesTrainers.append(perSerTrainer)
    
  embed=torch.nn.Linear(DATES_ENCODE_SIZE, DATES_EMBED_SIZE)
  rnn=DilatedSparseRnnStack(DILATIONS, CELLS_NAME, INPUT_SIZE, STATE_SIZE, TOTAL_OUTPUT_SIZE, H_SIZE, INP_SIZE)
  allParams = list(embed.parameters()) + list(rnn.parameters())
  trainer=torch.optim.Adam(allParams, lr=INITIAL_LEARNING_RATE)
  #rnn.adaptor.weight.shape #74, 60 [TOTAL_OUTPUT_SIZE, OUT_SIZE]
  #rnn.adaptor.weight[0].shape #60
  #rnn.adaptor.bias[0]
  if DISPLAY_SM_WEIGHTS:
    print('levSm w:', rnn.adaptor.weight[0].detach().numpy())
    print('levSm b:', rnn.adaptor.bias[0].detach().numpy())

    print('sSm w:', rnn.adaptor.weight[1].detach().numpy())
    print('sSm b:', rnn.adaptor.bias[1].detach().numpy())

  
  learningRate=INITIAL_LEARNING_RATE
  batchSize=INITIAL_BATCH_SIZE
  iEpoch=0; prevNumOfRepeats=0
  print('num of epochs:',NUM_OF_EPOCHS)
  for iEpoch in range(NUM_OF_EPOCHS):  #-<<-----------epoch------------
    nowe=dt.datetime.now()
    print(nowe.strftime("%Y-%m-%d %H:%M:%S"),  'starting epoch:',iEpoch)
    
    if not USE_DB:
      forecast_df=None
      forecastPath=outputDir+"w"+str(workerNumber)+"e"+str(iEpoch)+".csv"
      
    if iEpoch in BATCH_SIZES:
      batchSize=BATCH_SIZES[iEpoch]
      print ("changing batch size to:",batchSize)
    
    if iEpoch in LEARNING_RATES:
      learningRate=LEARNING_RATES[iEpoch]
      for param_group in trainer.param_groups:
          param_group['lr']=learningRate     
      for series in series_list: 
        for param_group in perSeriesTrainers[series].param_groups:
          param_group['lr']=learningRate
      print('changin LR to:', f'{learningRate:.2}' )   
      
    epochTrainingErrors=[];
    epochTrainingQErrors=[]; #training errors per quant
    for iq in range(NUM_OF_QUANTS):
      epochTrainingQErrors.append([]);  
    epochValidationErrors=[];
    
    numOfEpochLoops=int(np.power(NUM_OF_UPDATES_PER_EPOCH*batchSize/numSeries,EPOCH_POW))
    if numOfEpochLoops<1:
      numOfEpochLoops=1  
    if prevNumOfRepeats!=numOfEpochLoops and numOfEpochLoops>1:
      print ("repeating epoch loop "+str(numOfEpochLoops)+" times")
    prevNumOfRepeats=numOfEpochLoops       

    numOfUpdatesSoFar=0; isubEpoch=0 
    while isubEpoch<numOfEpochLoops:
      if numOfUpdatesSoFar>=NUM_OF_UPDATES_PER_EPOCH/3:
        print("number of updates reached",numOfUpdatesSoFar,"stopping earlier this epoch")
        isubEpoch=numOfEpochLoops-1
        
      isTesting=iEpoch>0 and isubEpoch==numOfEpochLoops-1 #for the last subepoch first training then testing 
      
      batches=[]; batch=[]
      random.shuffle(series_list)
      for series in series_list:
        batch.append(series)
        if len(batch) >= batchSize:
          batches.append(batch);
          batch=[]
      if len(batch)>0:
        batches.append(batch)
      random.shuffle(batches)
      
      batch=batches[0]
      for batch in batches:
        if DEBUG_AUTOGRAD_ANOMALIES:
          print(batch)
        ppBatch=Batch(batch)
        
        #oneBatch_df=None # intermediary, for speed  .See below
        rnn.resetState()
        trainingErrors=[];  trainingQErrors=[]; 
        for iq in range(NUM_OF_QUANTS):
          trainingQErrors.append([]);

        #start levels and extend seasonality with a static smoothiong coefs
        ii=0
        levels=[]; seasonality=ppBatch.initialSeasonality.copy()
        levSm=torch.sigmoid(torch.stack([perSeriesParams[x].initLevSm for x in batch])).view([ppBatch.batchSize,1])
        sSm=torch.sigmoid(torch.stack([perSeriesParams[x].initSSm for x in batch])).view([ppBatch.batchSize,1])
        y_l=[]
        for ii in range(INPUT_WINDOW):
          newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
          assert torch.isnan(newY).sum()==0
          y_l.append(newY)    

          if ii==0:
            newLevel=newY/seasonality[0]
          else:
            newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
          levels.append(newLevel)
          
          newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
          seasonality.append(newSeason)
        
        remainingWarmupSteps=TRAINING_WARMUP_STEPS*STEP_SIZE-SEASONALITY-INPUT_WINDOW #we do not count here the first SEASONALITY done in Batch()
        istep=INPUT_WINDOW-1 #index of last level
        for istep in range(INPUT_WINDOW-1, 
          INPUT_WINDOW-1+remainingWarmupSteps+NUM_OF_TRAINING_STEPS*STEP_SIZE, STEP_SIZE):
          
          isTraining = istep>=INPUT_WINDOW-1+remainingWarmupSteps 
          dat=ppBatch.dates[istep]
          if istep>=INPUT_WINDOW:
            ii=istep+1-STEP_SIZE
            for ii in range(istep+1-STEP_SIZE, istep+1):
              newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
              if torch.isnan(newY).sum()>0:
                newY=newY.clone()
                for ib in range(ppBatch.batchSize):
                  if torch.isnan(newY[ib]):
                    assert ii-SEASONALITY>=0
                    newY[ib]=ppBatch.y[ib,ii-SEASONALITY]
              assert torch.isnan(newY).sum()==0
              y_l.append(newY)    
                  
              newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
              levels.append(newLevel)
              
              newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
              seasonality.append(newSeason)
            
          encodedDate0_t=datesToMetadata(dat).repeat(ppBatch.batchSize,1)
          encodedDate_t=embed(encodedDate0_t)
          x0_t=torch.cat(y_l[istep-INPUT_WINDOW+1:istep+1], dim=1)
          anchorLevel=torch.mean(x0_t, dim=1).view([ppBatch.batchSize,1])
          inputSeasonality_t=torch.cat(seasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality_t.shape
          x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel))
          
          outputSeasonality=torch.cat(seasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1) #outputSeasonality.shape
          #prevOutputSeasonality=torch.cat(seasonality[istep+1-SEASONALITY:istep-SEASONALITY+OUTPUT_WINDOW+1],dim=1)
          #diffInOutputSeasonalities=outputSeasonality-prevOutputSeasonality
                
          input_t=torch.cat([x_t, torch.log10(anchorLevel), encodedDate_t, outputSeasonality-1],  dim=1)  
          #input_t.shape
          
          forec0_t=rnn(input_t)
          #print(forec0_t, forec0_t.shape)
          if torch.isnan(forec0_t).sum()>0:
            print(forec0_t)
            trouble("nans in forecast")
            
          if len(forec0_t.shape)==1:
            forec0_t=torch.unsqueeze(forec0_t,dim=0)
          #forec0_t.shape
            
          levSm=torch.sigmoid(forec0_t[:,0]+LEV_SM0).view([ppBatch.batchSize,1])
          sSm=torch.sigmoid(forec0_t[:,1]+S_SM0).view([ppBatch.batchSize,1])
          
          if isTraining:  
            actuals_t=ppBatch.y[:,istep+1:istep+OUTPUT_WINDOW+1].clone()
            forec_t=torch.exp(forec0_t[:,2:])*outputSeasonality.repeat(1,NUM_OF_QUANTS) #outputSeasonality.shape
            loss_lt=trainingLossFunc(forec_t, actuals_t, anchorLevel)
            avgLoss=torch.nanmean(torch.cat(loss_lt))
            if not torch.isnan(avgLoss):
              trainingErrors.append(torch.unsqueeze(avgLoss,dim=0))
            for iq in range(NUM_OF_QUANTS):
              trainingQErrors[iq].append(loss_lt[iq].detach().numpy())   
               
        #batch level     
        if len(trainingErrors)>0:
          trainer.zero_grad()  
            
          avgTrainLoss_t=torch.mean(torch.cat(trainingErrors))    
          assert not torch.isnan(avgTrainLoss_t)  
          avgTrainLoss_t.backward()          
          trainer.step()    
          
          for series in ppBatch.series:
            perSeriesTrainers[series].step()  #here series is integer
                
          epochTrainingErrors.append(avgTrainLoss_t.detach().numpy())
          trainingErrors=[]
          
          for iq in range(NUM_OF_QUANTS):
            epochTrainingQErrors[iq].append(np.mean(trainingQErrors[iq]))
            trainingQErrors[iq]=[]
        #end of batch    
        
              
        if isTesting:
          ppBatch=Batch(batch, False) 
          oneBatch_df=None
          rnn.resetState()
            
          ii=0
          levels=[]; seasonality=ppBatch.initialSeasonality.copy()
          levSm=torch.sigmoid(torch.stack([perSeriesParams[x].initLevSm for x in batch])).view([ppBatch.batchSize,1])
          sSm=torch.sigmoid(torch.stack([perSeriesParams[x].initSSm for x in batch])).view([ppBatch.batchSize,1])
          for ii in range(INPUT_WINDOW):
            newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
            if ii==0:
              newLevel=newY/seasonality[0]
            else:
              newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
            levels.append(newLevel)
            
            newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
            seasonality.append(newSeason)
            
      
          remainingWarmupSteps=TESTING_WARMUP_STEPS*STEP_SIZE-SEASONALITY-INPUT_WINDOW #we do not count here the first SEASONALITY done in Batch()    
          with torch.no_grad():
            istep=INPUT_WINDOW-1 #index of last level
            for istep in range(INPUT_WINDOW-1, ppBatch.y.shape[1]-OUTPUT_WINDOW, STEP_SIZE):
              warmupFinished = istep>=INPUT_WINDOW-1+remainingWarmupSteps  
              dat=ppBatch.dates[istep]
              oneDate_df=None
              if istep>=INPUT_WINDOW:
                for ii in range(istep+1-STEP_SIZE, istep+1):
                  newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
                  if torch.isnan(newY).sum()>0:
                    for ib in range(ppBatch.batchSize):
                      if torch.isnan(newY[ib]):
                        assert ii-SEASONALITY>=0
                        newY[ib]=ppBatch.y[ib,ii-SEASONALITY]
                        ppBatch.y[ib,ii]=newY[ib] #patching input, not output. No gradient needed, so we can overwrite
                  assert torch.isnan(newY).sum()==0
                        
                  newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
                  levels.append(newLevel)
                  
                  newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
                  seasonality.append(newSeason)
              
              encodedDate0_t=datesToMetadata(dat).repeat(ppBatch.batchSize,1)
              encodedDate_t=embed(encodedDate0_t)
              x0_t=ppBatch.y[:,istep-INPUT_WINDOW+1:istep+1] #x0_t.shape
              anchorLevel=torch.mean(x0_t, dim=1).view([ppBatch.batchSize,1])
              inputSeasonality_t=torch.cat(seasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality.shape
              x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel))
              
              outputSeasonality=torch.cat(seasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1)
              #prevOutputSeasonality=torch.cat(seasonality[istep+1-SEASONALITY:istep-SEASONALITY+OUTPUT_WINDOW+1],dim=1)
              #diffInOutputSeasonalities=outputSeasonality-prevOutputSeasonality
                    
              input_t=torch.cat([x_t, torch.log10(anchorLevel), encodedDate_t, outputSeasonality-1],  dim=1)  
                  
              forec0_t=rnn(input_t)
              #print(forec0_t, forec0_t.shape)
              if torch.isnan(forec0_t).sum()>0:
                print('forec:',forec0_t)
                print('input:', input_t)
                trouble("nans in test forecast")
                
              if len(forec0_t.shape)==1:
                forec0_t=torch.unsqueeze(forec0_t,dim=0)
                
              levSm=torch.sigmoid(forec0_t[:,0]+LEV_SM0).view([ppBatch.batchSize,1])
              sSm=torch.sigmoid(forec0_t[:,1]+S_SM0).view([ppBatch.batchSize,1])
                
              if warmupFinished:
                actuals=ppBatch.y[:,istep+1:istep+OUTPUT_WINDOW+1].detach().numpy()
                forec_t=torch.exp(forec0_t[:,2:])*anchorLevel*outputSeasonality.repeat(1,NUM_OF_QUANTS)
                forec=np.maximum(smallNegative,forec_t.detach().numpy())
                loss=validationLossFunc(forec, actuals, ppBatch.maseNormalizer)#rmse, bias, mase 
                epochValidationErrors.append(loss)
          
                saveTesting=iEpoch>=FIRST_EPOCH_TO_START_SAVING_FORECASTS and \
                  random.choice(range(SAVE_TEST_EVERY_STEP))==0
                if saveTesting:
                  iser=0;
                  for iser in range(ppBatch.batchSize):
                    series=ppBatch.series[iser]
                    avgForec=forec[iser]
                    da=dat
                    if USE_DB:
                      if USE_ANSI_DRIVER:
                        theVals=[now, workerNumber,iEpoch, da,  
                                 bytearray(series, encoding='utf-8')] 
                      else:
                        theVals=[now, workerNumber, iEpoch, da,  
                                series]
                          
                      horizon=0
                      for horizon in range(OUTPUT_WINDOW):
                        actu=float(actuals[iser,horizon])
                        if np.isnan(actu):
                          actu=None
                        pQ50=float(avgForec[horizon])
                        pQ05=float(avgForec[horizon+OUTPUT_WINDOW])
                        pQ95=float(avgForec[horizon+2*OUTPUT_WINDOW])
                        theVals.extend([actu, pQ50, pQ05, pQ95])
                      cursor.execute(INSERT_QUERY,theVals)
                    else:
                      oneRow_df=pd.DataFrame([[series, da]], columns=["series","date"])
                      for horizon in range(OUTPUT_WINDOW):
                        actu=float(actuals[iser,horizon])
                        if np.isnan(actu):
                          actu=None
                        pQ50=float(avgForec[horizon])
                        pQ05=float(avgForec[horizon+OUTPUT_WINDOW])
                        pQ95=float(avgForec[horizon+2*OUTPUT_WINDOW])
                        oneHorizon_df=pd.DataFrame([[actu,  pQ50, pQ05, pQ95]], \
                          columns= ["actuals_"+str(horizon), "pQ50_"+str(horizon), "pQ05_"+str(horizon), "pQ95_"+str(horizon)])
                        oneRow_df=pd.concat([oneRow_df, oneHorizon_df],axis=1) #one row=one series at a date
                              
                      if oneDate_df is None:
                        oneDate_df=oneRow_df.copy()
                      else:
                        oneDate_df=oneDate_df.append(oneRow_df.copy())
                              
                  if not USE_DB:        
                    if oneBatch_df is None:
                      oneBatch_df=oneDate_df.copy() #higher loop is through dates, here only one date per series
                    else:
                      oneBatch_df=oneBatch_df.append(oneDate_df.copy())
                          
            if not USE_DB and iEpoch>=FIRST_EPOCH_TO_START_SAVING_FORECASTS:    
              if forecast_df is None:
                forecast_df=oneBatch_df.copy()
              else:
                forecast_df=forecast_df.append(oneBatch_df.copy())
              #print(forecast_df.shape)
                        
        numOfUpdatesSoFar+=1          
      #through batches
      isubEpoch+=1
    #through sub-epoch
    if iEpoch>=FIRST_EPOCH_TO_START_SAVING_FORECASTS:
      if USE_DB: 
        dbConn.commit()      
      else:      
        if not forecast_df is None:
          forecast_df.to_csv(forecastPath, index=False)
      

    if len(epochTrainingErrors)>0:
      avgTrainLoss=np.mean(epochTrainingErrors)
      avgTrainLosses=[]
      for iq in range(NUM_OF_QUANTS):
        avgTrainLosses.append(np.mean(epochTrainingQErrors[iq]))
      print('epoch:',iEpoch,
            ' avgTrainLoss:',f'{avgTrainLoss:.3}', 
            " perQuant:",avgTrainLosses) 
      
    if len(epochValidationErrors)>0:
      validErrors0=np.concatenate(epochValidationErrors, axis=0) 
      validErrors=np.nanmean(validErrors0,axis=0)
      print('valid, RMSE:', f'{validErrors[0]:.3}',
            ' MAPE:', f'{validErrors[3]*100:.3}', 
            ' %bias:', f'{validErrors[1]*100:.3}',  
            ' MASE:', f'{validErrors[2]:.3}', end=', % exceeded:')
      for iq in range(NUM_OF_QUANTS):
        print(' ',QUANTS[iq],':', f'{validErrors[4+iq]*100:.3}', end=',')
        
      minLSm=torch.tensor(100.); maxLSm=torch.tensor(-100.); sumLSm=torch.tensor(0.)
      minSSm=torch.tensor(100.); maxSSm=torch.tensor(-100.); sumSSm=torch.tensor(0.)
      for ipsp, psp in enumerate(perSeriesParams):
        sumLSm+=psp.initLevSm
        sumSSm+=psp.initSSm
        if psp.initLevSm<minLSm:
          minLSm=psp.initLevSm
        if psp.initSSm<minSSm:
          minSSm=psp.initSSm
        if psp.initLevSm>maxLSm:
          maxLSm=psp.initLevSm
        if psp.initSSm>maxSSm:
          maxSSm=psp.initSSm  
      print("\nLSm logit avg:",f'{(sumLSm/len(perSeriesParams)).detach().item():.3}', 
            " min:",f'{minLSm.detach().item():.3}', 
            " max:",f'{maxLSm.detach().item():.3}')
      print("SSm logit avg:",f'{(sumSSm/len(perSeriesParams)).detach().item():.3}', 
            " min:", f'{minSSm.detach().item():.3}', 
            " max:",f'{maxSSm.detach().item():.3}')
      print()
      
    if DISPLAY_SM_WEIGHTS:
      print('levSm w:', rnn.adaptor.weight[0].detach().numpy())
      print('levSm b:', rnn.adaptor.bias[0].detach().numpy())

      print('sSm w:', rnn.adaptor.weight[1].detach().numpy())
      print('sSm b:', rnn.adaptor.bias[1].detach().numpy())
  
  if DISPLAY_SM_COEFS:
    for ipsp, psp in enumerate(perSeriesParams):
      print(ipsp, "initLSm:", psp.initLevSm.detach().item(), "initSSm:", psp.initSSm.detach().item())
        
  print("done.")

            
          
"""
 CREATE TABLE electra4Models(
  run varchar(300) NOT NULL,
  workerNo tinyint NOT NULL,
  dateTimeOfPrediction datetime NOT NULL,
 CONSTRAINT electra4Models_PK PRIMARY KEY
(
  run ASC,
  workerNo asc
))
  
  query="CREATE TABLE electra5(\
    dateTimeOfPrediction datetime NOT NULL,\
    workerNo tinyint NOT NULL,\
    epoch tinyint NOT NULL,\
    forecOriginDate datetime NOT NULL,\
    series varchar(50) NOT NULL,\n"
  for ih in range(OUTPUT_WINDOW):
    query+="actual"+str(ih+1)+" real,"
    query+=" predQ50_"+str(ih+1)+" real,"
    query+=" predQ05_"+str(ih+1)+" real,"
    query+=" predQ95_"+str(ih+1)+" real,"
  query+="\nCONSTRAINT electra5_PK PRIMARY KEY (\
    dateTimeOfPrediction ASC,\
    workerNo ASC,\
    epoch ASC, \
    forecOriginDate ASC, \
    series ASC))"
  print(query)
  

#SQL Server or mySQL  . Postgress is almost the same, just syntax for rounding is a bit different
#validation
query="with avgValues as \n(select run, epoch, forecOriginDate, series, \
  count(distinct d.workerNo) workers, count(*) kount \n"
for ih in range(1,OUTPUT_WINDOW+1):
  query+=", avg(actual"+str(ih)+") actual"+str(ih) +\
    ", avg(predQ50_"+str(ih)+") predQ50_"+str(ih) +\
    ", avg(predQ05_"+str(ih)+") predQ05_"+str(ih) +\
    ", avg(predQ95_"+str(ih)+") predQ95_"+str(ih)
query+="\n from electra5 d with (nolock), electra4Models m with (nolock) \
  \n where d.dateTimeOfPrediction =m.dateTimeOfPrediction  and d.workerNo=m.workerNo \
  \n group by run, epoch, forecOriginDate, series)"
  
query+="\n, perForecMetrics as ( select run, epoch, forecOriginDate, series, workers \n"
for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="case when actual"+str(ih)+">predQ50_"+str(ih)+" then 100. else 0. end"
query+=")/"+str(OUTPUT_WINDOW)+" as exceed50 \n"

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="case when actual"+str(ih)+"<predQ05_"+str(ih)+" then 100. else 0. end"
query+=")/"+str(OUTPUT_WINDOW)+" as exceed05 \n"  


for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="case when actual"+str(ih)+">predQ95_"+str(ih)+" then 100. else 0. end"
query+=")/"+str(OUTPUT_WINDOW)+" as exceed95 \n"  
  
for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="abs(predQ50_"+str(ih)+"-actual"+str(ih)+")/actual"+str(ih) 
query+=")/"+str(OUTPUT_WINDOW)+" as MAPE \n"  

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="(predQ50_"+str(ih)+"-actual"+str(ih)+")/actual"+str(ih) 
query+=")/"+str(OUTPUT_WINDOW)+" as MPE \n"  

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="(predQ50_"+str(ih)+"-actual"+str(ih)+")*(predQ50_"+str(ih)+"-actual"+str(ih)+")" 
query+=")/"+str(OUTPUT_WINDOW)+" as MSE \n"  

query+="from avgValues),"
#aggregate over forecasts
query+="perSeries as (select run, series, epoch, count(*) kount, \n\
 avg(MAPE) MAPE, avg(MPE) pcBias, \n\
 sqrt(avg(MSE)) RMSE,  \n\
 avg(exceed50) exceed50, avg(exceed05) exceed05, avg(exceed95) exceed95,  \n\
 count(distinct forecOriginDate) numForecasts, max(workers) workers \n\
 from perForecMetrics \n\
 group by run, series, epoch) \n\
select run, epoch, count(*) kount,  \n\
 round(avg(MAPE)*100,3) MAPE, round(avg(pcBias)*100,3) pcBias, \n\
 round(avg(RMSE),3) RMSE,  \n\
 round(avg(exceed50),2) exceed50, round(avg(exceed05),2) exceed05, round(avg(exceed95),2) exceed95, \n\
 avg(numForecasts) numForecasts, max(workers) workers \n\
 from perSeries \n\
 group by run, epoch \n\
 order by run, epoch\n" 
  
print(query)
  
#useful query to cleanup the db, by keeping forecasts only from best epoch (as displayed by the query above):
delete from electra5 where dateTimeOfPrediction in
(select dateTimeOfPrediction from electra4Models
where run='a particular run')
and epoch not in (9);
  
# --export
GOOD_RUN="72 S3[2],S3[4],S3[7] [0.48,0.035,0.96] LR=3e-3 {5,/3, 6,/10, 7/30} batch=2 {4:5}"
GOOD_EPOCH=8
query="with avgValues as \n(select series, forecOriginDate \n"
for ih in range(1,OUTPUT_WINDOW+1):
  query+=", avg(predQ50_"+str(ih)+") predQ50_"+str(ih)
query+="\n"   
for ih in range(1,OUTPUT_WINDOW+1):
  query+=", avg(actual"+str(ih)+") actual"+str(ih)
query+="\n" 
for ih in range(1,OUTPUT_WINDOW+1):
  query+=", avg(predQ05_"+str(ih)+") predQ05_"+str(ih)
query+="\n" 
for ih in range(1,OUTPUT_WINDOW+1):
  query+=", avg(predQ95_"+str(ih)+") predQ95_"+str(ih)
query+=",count(*) kount\n" 
query+="from electra5 d with (nolock), electra4Models m with (nolock) \n\
  where d.dateTimeOfPrediction =m.dateTimeOfPrediction  and d.workerNo=m.workerNo \n\
  and run='"+GOOD_RUN+"' and epoch="+str(GOOD_EPOCH) +\
  " group by forecOriginDate, series) \n\
  select * from avgValues order by series, forecOriginDate"
print(query)

  """
