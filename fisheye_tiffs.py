import glob
import os
import multiprocessing as mp
import csv
import itertools

inputpath = '/home/raghu/Documents/glare_analysis_cnn/fisheye'
outputpath = '/home/raghu/Documents/glare_analysis_cnn/fisheye_bmp_90'

for dirpath, dirnames, filenames in os.walk(inputpath):
    for dirname in dirnames:
        structure = os.path.join(outputpath, dirname)
        if not os.path.isdir(structure):
            os.mkdir(structure)

def task(input, output):
    #cmd='/usr/local/radiance/bin/ra_tiff -l '+input+'.hdr'+' '+output+'.tif'
    cmd = '/usr/local/radiance/bin/ra_bmp -e human ' + input + '.hdr' + ' ' + output + '.bmp'
    print(cmd)
    os.system(cmd)


def func(args):
    return task(*args)
job_args=[]

for  dirname in os.listdir(inputpath):
    for file in os.listdir(inputpath+'/'+dirname):
    #for dirname,filename in zip(dirnames,filenames):
        #for filename in filenames:
        if '90' in file:
            input = os.path.join(inputpath, dirname, file.split('.')[0])
            output = os.path.join(outputpath, dirname,file.split('.')[0])
            comb=[input,output]
            job_args.append(comb)

pool=mp.Pool(processes=16)
pool.map(func,job_args)
pool.close()
pool.join()