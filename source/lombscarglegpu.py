import os
import numpy.ctypeslib as npct
from ctypes import *
import csv
import numpy as np
from contextlib import contextmanager
import sys

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]


# This function converts an input numpy array into a different
# data type and ensure that it is contigious.  
def convert_type(in_array, new_dtype):

    ret_array = in_array
    
    if not isinstance(in_array, np.ndarray):
        ret_array = np.array(in_array, dtype=new_dtype)
    
    elif in_array.dtype != new_dtype:
        ret_array = np.array(ret_array, dtype=new_dtype)

    if ret_array.flags['C_CONTIGUOUS'] == False:
        ret_array = np.ascontiguousarray(ret_array)

    return ret_array


#from stackoverflow 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
def redirect_stdout():
    print ("Verbose mode is false. Redirecting C shared library stdout to /dev/null")
    sys.stdout.flush() # <--- important when redirecting to files
    newstdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')

#from stackoverflow 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    print ("Verbose mode is false. Redirecting C shared library stdout to /dev/null")
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

#from here: https://www.py4u.net/discuss/15884
class SuppressStream(object): 

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()

#https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def computeIndexRangesForEachObject(objId):
    start_index_arr=[]
    end_index_arr=[]
    unique_obj_ids_arr=[]
    lastId=objId[0]
    unique_obj_ids_arr.append(objId[0])

    start_index_arr.append(0)
    for x in range(0, len(objId)):
        if(objId[x]!=lastId):
            end_index_arr.append(x-1)
            start_index_arr.append(x)
            lastId=objId[x]
            #update the list of unique object ids
            unique_obj_ids_arr.append(objId[x])

    #last one needs to be populated
    end_index_arr.append(len(objId)-1)

    start_index_arr=np.asarray(start_index_arr, dtype=int)
    end_index_arr=np.asarray(end_index_arr, dtype=int)
    unique_obj_ids_arr=np.asarray(unique_obj_ids_arr)

    return start_index_arr, end_index_arr, unique_obj_ids_arr


#https://www.codeforests.com/2020/11/05/python-suppress-stdout-and-stderr/    

@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr

def enumerateObjects(start_index_arr, end_index_arr):
    enumObjectId=[]
    for x in range (start_index_arr.size):
        numElems=end_index_arr[x]-start_index_arr[x]+1
        # print("Num elems: %d" %(numElems))
        enumObjectId.extend(numElems*[x])

    enumObjectId=np.asarray(enumObjectId, dtype=int)    
    # print("Total number of lines after enumeration: %d" %(enumObjectId.size))    
    # print("Total number of unique objects after enumeration: %d" %(np.unique(enumObjectId).size))    
    return enumObjectId


#Use the formulation in Richards et al. 2011
def computeNumFreqAuto(objId, timeX, fmin, fmax):

    start_index_arr, end_index_arr, _ = computeIndexRangesForEachObject(objId)     

    timeXLocal=np.asfarray(timeX)

    observing_window_arr=[]
    for x in range (0, start_index_arr.size):
        idxStart=start_index_arr[x]
        idxEnd=end_index_arr[x]
        observing_window_arr.append(timeXLocal[idxEnd]-timeXLocal[idxStart])

    observing_window_arr=np.asarray(observing_window_arr, dtype=float)

    maximumObservingWindow=np.max(observing_window_arr)

    

    deltaf=0.1/maximumObservingWindow

    num_freqs=(fmax-fmin)/deltaf
    num_freqs=int(num_freqs)
    print("*********************")
    print("Automatically generating the number of frequencies based on maximum observing window:")
    print("Max. Observing Window: %f, Delta f: %f" %(maximumObservingWindow, deltaf))
    print("Number of frequencies: ", num_freqs)
    print("*********************")
    return num_freqs


#wrapper to enable the verbose option
def lombscargle(objId, timeX, magY, minFreq, maxFreq, error, mode, magDY=None, freqToTest="auto", dtype="float", verbose=False):
    if(verbose==False):
        with HideOutput():
            ret_uniqueObjectIdsOrderedWrapper, ret_periodsWrapper, ret_pgramWrapper = lombscarglemain(objId, timeX, magY, minFreq, maxFreq, error, mode, magDY, freqToTest, dtype)            
    else:
        ret_uniqueObjectIdsOrderedWrapper, ret_periodsWrapper, ret_pgramWrapper = lombscarglemain(objId, timeX, magY, minFreq, maxFreq, error, mode, magDY, freqToTest, dtype, verbose)

    return ret_uniqueObjectIdsOrderedWrapper, ret_periodsWrapper, ret_pgramWrapper

#main L-S function
def lombscarglemain(objId, timeX, magY, minFreq, maxFreq, error, mode, magDY=None, freqToTest="auto", dtype="float", verbose=False):

    
    #store the minimum/maximum frequencies (needed later for period calculation)
    minFreqStandard=minFreq
    maxFreqStandard=maxFreq
    #convert oscillating frequencies into angular
    minFreq=2.0*np.pi*minFreq
    maxFreq=2.0*np.pi*maxFreq

    ###############################
    #Check for valid parameters and set verbose mode and generate frequencies for auto mode 

    #prevent C output from printing to screen
    # if (verbose==False):
        # redirect_stdout()
        # stdout_redirected()

    #if the user doesn't specify the number of frequencies
    if (freqToTest=="auto"):
        freqToTest=computeNumFreqAuto(objId, timeX, minFreqStandard, maxFreqStandard)

    #check which mode to use in the C shared library
    # 1- GPU Batch of Objects Lomb-Scargle")
    # 2- GPU Single Object Lomb-Scargle")
    # 3- None
    # 4- CPU Batch of Objects Lomb-Scargle")
    # 5- CPU Single Object Lomb-Scargle")
        
    numObjects=np.size(np.unique(objId))    
    if (mode=="GPU" and numObjects>1):
        setmode=1
    elif (mode=="GPU" and numObjects==1):
        setmode=2
    elif (mode=="CPU" and numObjects>1):
        setmode=4
    elif (mode=="CPU" and numObjects==1):
        setmode=5        
    

    #check that if the error is true, that magDY is not None (None is the default parameter)
    if (error==True and magDY==None):
        print("[Python] Error: No input error array, but the error mode is True. Set error mode to False.")
        exit(0)      

    ###############################
    

    #enumerate objId so that we can process objects with non-numeric Ids
    #original objects are stored in ret_uniqueObjectIdsOrdered
    start_index_arr, end_index_arr, ret_uniqueObjectIdsOrdered = computeIndexRangesForEachObject(objId)     
    objId = enumerateObjects(start_index_arr, end_index_arr)

    # Create variables that define C interface
    array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='CONTIGUOUS')
    array_1d_float = npct.ndpointer(dtype=c_float, ndim=1, flags='CONTIGUOUS')
    array_1d_unsigned = npct.ndpointer(dtype=c_uint, ndim=1, flags='CONTIGUOUS')

    #load the shared library (either the noerror/error and float/double versions)
    lib_path = os.getcwd()
    if (error==False and dtype=="float"):
        liblombscarglenoerrorfloat = npct.load_library('libpylsnoerrorfloat.so', lib_path)
    elif (error==False and dtype=="double"):     
        liblombscarglenoerrordouble = npct.load_library('libpylsnoerrordouble.so', lib_path)
    elif (error==True and dtype=="float"):     
        liblombscargleerrorfloat = npct.load_library('libpylserrorfloat.so', lib_path)
    elif (error==True and dtype=="double"):     
        liblombscargleerrordouble = npct.load_library('libpylserrordouble.so', lib_path)    


    #total number of rows in file
    sizeData=len(objId)
    print("[Python] Number of rows in file: %d" %(sizeData))

    #convert input from lists to numpy arrays
    objId=np.asarray(objId, dtype=int)
    timeX=np.asfarray(timeX)
    magY=np.asfarray(magY)
    if(error==True):
        magDY=np.asfarray(magDY)

    #if error is false, we still need to send dummy array to C shared library
    #set all values to 1.0, although we don't use it for anything
    if(error==False):
        magDY=np.full(objId.size, 1.0)    

    #convert to CTYPES
    if (dtype=="float"):
        c_objId=convert_type(objId, c_uint)
        c_timeX=convert_type(timeX, c_float)
        c_magY=convert_type(magY, c_float)
        c_magDY=convert_type(magDY, c_float)
    elif (dtype=="double"):     
        c_objId=convert_type(objId, c_uint)
        c_timeX=convert_type(timeX, c_double)
        c_magY=convert_type(magY, c_double)
        c_magDY=convert_type(magDY, c_double)

    df=(maxFreq-minFreq)/freqToTest*1.0
    dfstandard=(maxFreqStandard-minFreqStandard)/freqToTest*1.0

    # Allocate arrays for results 
    uniqueObjects=np.size(np.unique(objId))
    print("[Python] Unique objects: %d" % (uniqueObjects))

    if (dtype=="float"):
        ret_pgram = np.zeros(uniqueObjects*freqToTest, dtype=c_float)
        pgramDataGiB=((ret_pgram.size*4.0)/(1024*1024*1024))    
    elif (dtype=="double"): 
        ret_pgram = np.zeros(uniqueObjects*freqToTest, dtype=c_double)      
        pgramDataGiB=((ret_pgram.size*8.0)/(1024*1024*1024))    

    print("[Python] Size of pgram in elems: %d (%f GiB)" %(ret_pgram.size, pgramDataGiB))

    #without error -- float
    if (error==False and dtype=="float"):
        #define the argument types
        liblombscarglenoerrorfloat.LombScarglePy.argtypes = [array_1d_unsigned, array_1d_float, array_1d_float, 
        array_1d_float, c_uint, c_double, c_double, c_uint, c_int, array_1d_float]
        #call the library
        liblombscarglenoerrorfloat.LombScarglePy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_int(setmode), ret_pgram)

    #without error -- double
    if (error==False and dtype=="double"):    
        #define the argument types
        liblombscarglenoerrordouble.LombScarglePy.argtypes = [array_1d_unsigned, array_1d_double, array_1d_double, 
        array_1d_double, c_uint, c_double, c_double, c_uint, c_int, array_1d_double]
        #call the library
        liblombscarglenoerrordouble.LombScarglePy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_int(setmode), ret_pgram)

    #with error -- float    
    if (error==True and dtype=="float"):
        #define the argument types
        liblombscargleerrorfloat.LombScarglePy.argtypes = [array_1d_unsigned, array_1d_float, array_1d_float, 
        array_1d_float, c_uint, c_double, c_double, c_uint, c_int, array_1d_float]
        #call the library
        liblombscargleerrorfloat.LombScarglePy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_int(setmode), ret_pgram)
    
    #with error -- double
    if (error==True and dtype=="double"):    
        #define the argument types
        liblombscargleerrordouble.LombScarglePy.argtypes = [array_1d_unsigned, array_1d_double, array_1d_double, 
        array_1d_double, c_uint, c_double, c_double, c_uint, c_int, array_1d_double]
        #call the library
        liblombscargleerrordouble.LombScarglePy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(minFreq), c_double(maxFreq), c_uint(freqToTest), c_int(setmode), ret_pgram)

    
    


    

    
    #for convenience, reshape the pgrams as a 2-D array
    ret_pgram=ret_pgram.reshape([uniqueObjects, freqToTest])



    ret_periods=np.zeros(uniqueObjects)

    #to compute best periods, work back in regular oscillating frequencies (not angular)
    for x in range(0, uniqueObjects):
    	# ret_periods[x]=1.0/(minFreq+(df*np.argmax(ret_pgram[x])))
        ret_periods[x]=1.0/(minFreqStandard+(dfstandard*np.argmax(ret_pgram[x])))


    if(uniqueObjects>1):    
        print("[Python] Sum of all periods: %f" %(np.sum(ret_periods)))
    else:
        print("[Python] Period for object: %f" %ret_periods[0])

    # sys.stdout.flush()



    return ret_uniqueObjectIdsOrdered, ret_periods, ret_pgram    




