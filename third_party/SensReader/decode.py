# decod the scanNet dataset, with 5 frame interval 
import glob 
import os 
import subprocess
import ipdb

def _read_split_file( filepath):
    '''
    Read data split txt file provided for Robust Vision
    ''' 
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ]
    return trajs 

def decode_sens_file( in_out, cmd_prex='./sens' ):
    input_file = in_out[0]
    output_fldr = in_out[1]
    if not os.path.exists(output_fldr):
        os.makedirs( output_fldr)
    cmd = '%s %s %s'%(cmd_prex, input_file, output_fldr)
    subprocess.call(cmd, shell= True) 

def main():
    import argparse
    parser = argparse.ArgumentParser() 

    parser.add_argument('--dataset_path', required =True, type=str, 
        help='The path to the scannet dataset, suppose the data is organized as ${dataset_path}/scene####_##/')
    parser.add_argument('--output_path', required =True, type=str, 
        help='The path to the output folder path')
    parser.add_argument('--split_file', required =True, type=str, 
        help='The path to the split txt file ')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_base_path =  args.output_path 
    trajs = _read_split_file( args.split_file )
    
    #sample_fldrs = sorted(glob.glob('%s/scene*'%(dataset_path)))
    sample_fldrs = [ '%s/%s'%(dataset_path, traj) for traj in trajs ]
    
    for idx, sample_fldr in enumerate( sample_fldrs) :
        sample_idx = sample_fldr[-7:]
        sens_file_path = '%s/scene%s.sens'%(sample_fldr, sample_idx)
    
        # make dir 
        output_fldr = '%s/scene%s'%(output_base_path, sample_idx)
        if not os.path.exists( output_fldr):
            os.makedirs( output_fldr)
    
        # do the decoding 
        cmd = './sens %s %s'%(sens_file_path, output_fldr)
        print('traj %d of %d '%(idx, len(sample_fldrs)) )
        print('Decoding %s ...'%(sens_file_path) )
        subprocess.call(cmd, shell=True) 


if __name__== "__main__":
    main()

