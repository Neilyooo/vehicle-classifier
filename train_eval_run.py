from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    ###参数定义####
    parser.add_argument('--dataset_name', type=str, default='vehicle')#./slim/datasets/vehicle.py
    parser.add_argument('--model_name', type=str, default='resnet_v2')#./slim/nets/
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--checkpoint_exclude_scopes', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--optimizer', type=str, default='adam')#_configure_optimizer(line 273 in train_image_classifier.py)
    parser.add_argument('--learning_rate', type=float, default='0.01')
    parser.add_argument('--clone_on_cpu', type=bool, default=False)#True on gpu
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--dataset_split_name', type=str, default='validation')
    parser.add_argument('--eval_dir', type=str, default='validation')
    parser.add_argument('--max_num_batches', type=int, default=128)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
##命令行运行
trian_cmd = 'python ./trian_image_classifier.py --dataset_name={dataset_name}  --dataset_dir={dataset_dir} --checkpoint_path={checkpoint_path} --model_name={model_name} --checkpoint_exclude_scopes={checkpoint_exclude_scopes} --train_dir={train_dir} --learning_rate={learning_rate} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'

##run
if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)
    #train a epoch
    epoch = 50000 // FLAGS.batch_size #epoch_size大概 1562steps
    for i in range(40):#40*1562
        steps = int(epoch * (i+1)) #max steps
        print('**************    train   ******************')
        run = os.popen(train_cmd.format(**{'dataset_name':FLAGS.dataset_name, 'model_name':FLAGS.model_name, 'checkpoint_path':FLAGS.checkpoint_path,
                                         'dataset_dir':FLAGS.dataset_dir, 'checkpoint_exclude_scopes':FLAGS.checkpoint_exclude_scopes
                                         'train_dir':FLAGS.train_dir, 'optimizer':FLAGS.optimizer, 'learning_rate':FLAGS.learning_rate,
                                         'clone_on_cpu':FLAGS.clone_on_cpu, 'batch_size':FLAGS.batch_size, 'max_number_of_steps':steps}))
        for line in run:
            print(run.strip())
        print('**************    eval    *******************')
        run = os.popen(eval_cmd.format(**{'dataset_name':FLAGS.dataset_name, 'dataset_dir':FLAGS.dataset_dir,
                                          'checkpoint_path': FLAGS.train_dir, 'model_name':FLAGS.model_name,
                                          'dataset_split_name': 'validation', 'batch_size':FLAGS.batch_size,
                                          'eval_dir': FLAGS.eval_dir, 'max_num_batches':FLAGS.max_num_batches}))
        for line in run:
            print(run.strip())
