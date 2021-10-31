import csv
import numpy as np

#load the Lomb-Scargle Python library
import lombscarglegpu as ls


def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]

def importDataForExamples(fname, error):
    print("[Python] Filename: "+fname)
    objId=getColumn(fname, 0)
    timeX=getColumn(fname, 1)
    magY=getColumn(fname, 2)
    if(error==True):
        magDY=getColumn(fname, 3)
        return objId, timeX, magY, magDY
    else:    
        return objId, timeX, magY


if __name__ == "__main__":

    #parameters
    #input min and max frequency range in standard oscillating frequencies (not angular frequencies)
    min_f=0.1 #Minimum frequency range
    max_f=10.0 #Maximum frequency range
    N_f=330000 #Number of frequencies
    verbose=False #True/False --- this is the C output from the shared library
    dtype="float"
    
    print("******************")
    print("Example 1: Compute single object on the GPU with error")
    print("******************")

    error=True #Use photometric error 
    lsmode="GPU" #Mode (CPU or GPU)
    fname="8205_normalized_with_error.txt"
    objIdArr, timeXArr, magYArr, magDYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, magDYArr,  N_f, dtype, verbose)    

    print("******************")
    print("Example 2: Compute single object on the CPU with error (same as above, but on the CPU)")
    print("******************")

    error=True #Use photometric error 
    lsmode="CPU" #Mode (CPU or GPU)
    fname="8205_normalized_with_error.txt"
    objIdArr, timeXArr, magYArr, magDYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, magDYArr,  N_f, dtype, verbose)        

    print("******************")
    print("Example 3: Compute single object on the GPU without error")
    print("******************")

    error=False #Do not use photometric error 
    lsmode="GPU" #Mode (CPU or GPU)
    fname="8205_normalized.txt"
    objIdArr, timeXArr, magYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, freqToTest=N_f, dtype=dtype, verbose=verbose)

    print("******************")
    print("Example 4: Compute single object on the CPU without error (same as above, but on the CPU)")
    print("******************")

    error=False #Do not use photometric error 
    lsmode="CPU" #Mode (CPU or GPU)
    fname="8205_normalized.txt"
    objIdArr, timeXArr, magYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, freqToTest=N_f, dtype=dtype, verbose=verbose)


    print("******************")
    print("Example 5: Compute batch of objects on the GPU with error")
    print("******************")

    error=True #Use photometric error 
    lsmode="GPU" #Mode (CPU or GPU)
    fname="SDSS_stripe82_band_z.txt"
    objIdArr, timeXArr, magYArr, magDYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, magDYArr,  N_f, dtype, verbose)    
    print("[Python] Period of first object: objId: %s period: %f" %(objIds[0],periods[0]))
    print("[Python] Period of second object: objId: %s period: %f" %(objIds[1],periods[1]))
    print("[Python] Period of third object: objId: %s period: %f" %(objIds[2],periods[2]))
    
    print("******************")
    print("Example 6: Compute batch of objects on the CPU with error (same as above, but on the CPU)")
    print("******************")

    error=True #Use photometric error 
    lsmode="CPU" #Mode (CPU or GPU)
    fname="SDSS_stripe82_band_z.txt"
    objIdArr, timeXArr, magYArr, magDYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, magDYArr,  N_f, dtype, verbose)        
    print("[Python] Period of first object: objId: %s period: %f" %(objIds[0],periods[0]))
    print("[Python] Period of second object: objId: %s period: %f" %(objIds[1],periods[1]))
    print("[Python] Period of third object: objId: %s period: %f" %(objIds[2],periods[2]))
    
    print("******************")
    print("Example 7: Compute batch of objects on the GPU without error")
    print("******************")

    error=False #Do not use photometric error 
    lsmode="GPU" #Mode (CPU or GPU)
    fname="normalized_alltargs.200724_1_log_normal_obs.dat"
    objIdArr, timeXArr, magYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, freqToTest=N_f, dtype=dtype, verbose=verbose)    
    print("[Python] Period of first object: objId: %s period: %f" %(objIds[0],periods[0]))
    print("[Python] Period of second object: objId: %s period: %f" %(objIds[1],periods[1]))
    print("[Python] Period of third object: objId: %s period: %f" %(objIds[2],periods[2]))
    
    print("******************")
    print("Example 8: Compute batch of objects on the CPU without error (same as above, but on the CPU)")
    print("******************")

    error=False #Do not use photometric error 
    lsmode="CPU" #Mode (CPU or GPU)
    fname="normalized_alltargs.200724_1_log_normal_obs.dat"
    objIdArr, timeXArr, magYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, freqToTest=N_f, dtype=dtype, verbose=verbose)
    print("[Python] Period of first object: objId: %s period: %f" %(objIds[0],periods[0]))
    print("[Python] Period of second object: objId: %s period: %f" %(objIds[1],periods[1]))
    print("[Python] Period of third object: objId: %s period: %f" %(objIds[2],periods[2]))

    print("******************")
    print("Example 9: Compute batch of objects on the GPU without error using auto-generated frequencies")
    print("******************")

    error=False #Do not use photometric error 
    lsmode="GPU" #Mode (CPU or GPU)
    fname="normalized_alltargs.200724_1_log_normal_obs.dat"
    objIdArr, timeXArr, magYArr = importDataForExamples(fname, error)
    objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, dtype=dtype, verbose=verbose)    
    print("[Python] Period of first object: objId: %s period: %f" %(objIds[0],periods[0]))
    print("[Python] Period of second object: objId: %s period: %f" %(objIds[1],periods[1]))
    print("[Python] Period of third object: objId: %s period: %f" %(objIds[2],periods[2]))
    


    

    #use default parameters
    #objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, magDYArr, min_f, max_f, lsmode)    
    
    #with all parameters assigned -- with error    
    # if (error==True):
    #     objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, magDYArr,  N_f, dtype, verbose)    

    #with all parameters assigned -- without error    
    # if (error==False):
    #     objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode,  freqToTest=N_f, dtype=dtype, verbose=verbose)    
        # objIds, periods, pgrams = ls.lombscargle(objIdArr, timeXArr, magYArr, min_f, max_f, error, lsmode, dtype=dtype, verbose=verbose)    

    

    # print("[Python] Obj 0: objId: %s period: %f" %(objIds[0],periods[0]))

    #this is only used so that we can run multiple examples at once and surpress the C stdout
    if(verbose==False):
        ls.redirect_stdout()

