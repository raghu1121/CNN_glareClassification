import glob
import os
import multiprocessing as mp
import csv
import itertools

inputpath = '/home/raghu/Documents/glare_analysis_cnn/fisheye/Intolerable'
#outputpath = '/home/raghu/Documents/glare_analysis_cnn/fisheye_bmp_90'

# for dirpath, dirnames, filenames in os.walk(inputpath):
#     for dirname in dirnames:
#         structure = os.path.join(outputpath, dirname)
#         if not os.path.isdir(structure):
#             os.mkdir(structure)

def task(input,file):
    #cmd1='echo -n '+file+' >> stats.csv'
    cmd2 = '/usr/local/radiance/bin/evalglare -vta -vv 180 -vh 180 -d '+input+' | tail -1  | awk \'{ printf \"'+file+' %20s %20s\\n\",$2, $4}\' >> stats.csv '
    #print(cmd1)
    print(cmd2)
    #os.system(cmd1)
    os.system(cmd2)


def func(args):
    return task(*args)
job_args=[]

# for  dirname in os.listdir(inputpath):
#     for file in os.listdir(inputpath+'/'+dirname):
#     #for dirname,filename in zip(dirnames,filenames):
for file in os.listdir(inputpath):
    if '90' in file:
        input = os.path.join(inputpath, file)
        # output = os.path.join(outputpath, dirname,file.split('.')[0])
        comb=[input,file.split('.')[0]]
        job_args.append(comb)

pool=mp.Pool(processes=16)
pool.map(func,job_args)
pool.close()
pool.join()