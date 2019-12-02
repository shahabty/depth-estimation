import numpy as np
import torchvision.transforms as standard_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from metric_depth_model import MetricDepthModel
from load_dataset import CustomerDataLoader
from utils import load_ckpt,save_ckpt,resize_image
from evaluate import evaluate_err
cfg = {
'mode': 'train',
'device': 'cuda:0',
'RESNET_BOTTLENECK_DIM':[64, 256, 512, 1024, 2048],
'LATERAL_OUT':[512, 256, 256, 256],
'FCN_DIM_IN': [512, 256, 256, 256, 256, 256],
'FCN_DIM_OUT': [256, 256, 256, 256, 256],
'ENCODER': 'resnext101_32x4d_body_stride16',
'INIT_TYPE': 'xavier',
'LOAD_IMAGENET_PRETRAINED_WEIGHTS': False,
'CROP_SIZE': (385, 385), # (height, width)
'DECODER_OUTPUT_C': 150,
'FREEZE_BACKBONE_BN': False,
'ROOT_DIR':'/home/shahab/depth',
'MODEL_REPOSITORY': 'datasets/pretrained_model',
'PRETRAINED_WEIGHTS': 'resnext101_32x4d.pth',
'DEPTH_MIN': 0.01, # Minimum depth after data augmentation
'DEPTH_MAX': 1.7,# Maximum depth
'DEPTH_MIN_LOG':0,
'DEPTH_BIN_INTERVAL':None,
'DEPTH_BIN_BORDER':None,
'RGB_PIXEL_MEANS': (0.485, 0.456, 0.406),  # (102.9801, 115.9465, 122.7717)
'RGB_PIXEL_VARS': (0.229, 0.224, 0.225),  # (1, 1, 1)
}

cfg['DEPTH_MIN_LOG'] = np.log10(cfg['DEPTH_MIN'])
cfg['DEPTH_BIN_INTERVAL'] = (np.log10(cfg['DEPTH_MAX']) - np.log10(cfg['DEPTH_MIN']))/cfg['DECODER_OUTPUT_C']
cfg['DEPTH_BIN_BORDER'] = np.array([np.log10(cfg['DEPTH_MIN']) + cfg['DEPTH_BIN_INTERVAL']*(i+0.5) for i in range(cfg['DECODER_OUTPUT_C'])])


test_args = {
'phase':'test',
'phase_anno':'test',
'thread':1,
'batchsize':1,
'dataset':'nyudv2',
'dataroot':'NYUDV2',
'load_ckpt':None,
'start_step':0,
'start_epoch':0,
'save_dir':'',
}

train_args = {
'phase':'train',
'phase_anno':'train',
'epoch':20,
'batchsize':4,
'thread':8,
'dataset':'nyudv2',
'dataroot':'NYUDV2',
'load_ckpt':'pretrained_baselines/nyu_rawdata.pth',
'start_step':0,
'start_epoch':0,
'save_dir':'',
'lr':1e-4,

}

writer = SummaryWriter(comment='code')

def train(model,data_loader_train,data_loader_test,optimizer,criterion,cfg,train_args,test_args):
    if train_args['load_ckpt'] is not None:
        load_ckpt(train_args,model)
    model.train()
    lr = train_args['lr']
    for epoch in range(train_args['epoch']):
        print('epoch #: %d'%epoch)
        for i,data in tqdm(enumerate(data_loader_train)):
            target = data['B_bins'].squeeze().long().to(cfg['device'])
            

            output,pred_depth = model.train_nyuv2(data)
            output_softmax = output['b_fake_softmax']
            output_logit = output['b_fake_logit']

#            weights = torch.mean(torch.exp(-1*torch.pow((pred_depth - target.float()),2)))#*output_softmax )
            #print(pred_depth.shape)          
            #print(weights.shape)
            loss = criterion(output_softmax,target)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            lr = poly_lr_scheduler(optimizer,train_args['lr'],i + epoch*len(data_loader_train))
            if i%10 == 0:
                Img = vutils.make_grid(data['A'].data.cpu(),normalize = True,scale_each = True)
                GT_depth = vutils.make_grid(data['B'].data.cpu(),normalize = True,scale_each = True)
                Estimated_depth = vutils.make_grid(pred_depth.data.cpu(),normalize = True,scale_each = True)
                Edge = vutils.make_grid(data['E'].unsqueeze(1).repeat(1,3,1,1).data.cpu(),normalize = True,scale_each = True)
                inputs = vutils.make_grid((data['A']*data['E'].unsqueeze(1).repeat(1,3,1,1)).data.cpu(),normalize = True,scale_each = True) #x*e.repeat(1,3,1,1)
                writer.add_image('RGB',Img,i + epoch*len(data_loader_train))
                writer.add_image('GT_Depth',GT_depth,i + epoch*len(data_loader_train))
                writer.add_image('Predicted_Depth',Estimated_depth,i + epoch*len(data_loader_train))
                writer.add_image('Edge',Edge,i + epoch*len(data_loader_train))
                writer.add_image('inputs',inputs,i + epoch*len(data_loader_train))
            del target,output,pred_depth,loss
        print(lr)

        test(model,data_loader_test,cfg,test_args)
        save_ckpt(train_args['batchsize'],save_dir = '.',step = i + epoch*len(data_loader_train),epoch = epoch,model = model,optimizer = optimizer)

def test(model,data_loader,cfg,test_args):
#    if test_args['load_ckpt'] is not None:
#        load_ckpt(test_args,model)
    model.eval()
    error_total = {'err_absRel': 0.0, 'err_squaRel': 0.0, 'err_rms': 0.0,
                         'err_silog': 0.0, 'err_logRms': 0.0, 'err_silog2': 0.0,
                         'err_delta1': 0.0, 'err_delta2': 0.0, 'err_delta3': 0.0,
                         'err_log10': 0.0, 'err_whdr': 0.0}
    n_pxl_total = 0
    eval_num_total = 0
    for i, data in enumerate(tqdm(data_loader)):
        output = model.inference(data)
        pred_depth = torch.squeeze(output['b_fake'])
        img_path = data['A_paths']
        invalid_side = data['invalid_side'][0]
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth = pred_depth / data['ratio'].to(cfg['device']) # scale the depth
        pred_depth = resize_image(pred_depth, torch.squeeze(data['B_raw']).shape)
        error_batch,n_pxl,eval_num = evaluate_err(pred_depth, data['B_raw'], mask=(45, 471, 41, 601), scale=10.)
        for (k1,v1), (k2,v2) in zip(error_total.items(), error_batch.items()):
            error_total[k1] += error_batch[k2]
        #error_total = error_batch
        n_pxl_total = n_pxl_total + n_pxl
        eval_num_total = eval_num_total + eval_num
    error = calculate_average_error(error_total,n_pxl_total,eval_num_total)
    print('----------------------------------------------------------')
    print('absREL: %f'%error['err_absRel'])
    print('silog: %f'%np.sqrt(error['err_silog2'] - (error['err_silog'])**2))
    print('log10: %f'%error['err_log10'])
    print('RMS: %f'%error['err_rms'])
    print('delta1: %f'%error['err_delta1'])
    print('delta2: %f'%error['err_delta2'])
    print('delta3: %f'%error['err_delta3'])
    print('squaRel: %f'%error['err_squaRel'])
    print('logRms: %f'%error['err_logRms'])

    print('----------------------------------------------------------')
    del error,output,pred_depth,img_path,invalid_side
    model.train()

#def calculate_weights(output,target,output_softmax,cfg):
    
#    for i in range(cfg['DECODER_OUTPUT_C']):
#        output_logit[:,i,:,:] - 
#    torch.mean(torch.exp(-1*torch.pow((output_logit[:,target,:,:]),2))*output_softmax )


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=0.9,
                      max_iter=14500, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def calculate_average_error(error,n_pxl,eval_num):
    error_avg = {'err_absRel': 0.0, 'err_squaRel': 0.0, 'err_rms': 0.0,
                         'err_silog': 0.0, 'err_logRms': 0.0, 'err_silog2': 0.0,
                         'err_delta1': 0.0, 'err_delta2': 0.0, 'err_delta3': 0.0,
                         'err_log10': 0.0, 'err_whdr': 0.0}

    error_avg['err_absRel'] = error['err_absRel'] / n_pxl
    error_avg['err_squaRel'] = error['err_squaRel'] / n_pxl
    error_avg['err_rms'] = error['err_rms'] / n_pxl
    error_avg['err_silog'] = error['err_silog'] / n_pxl
    error_avg['err_logRms'] = error['err_logRms'] / n_pxl
    error_avg['err_silog2'] = error['err_silog2'] / n_pxl
    error_avg['err_delta1'] = error['err_delta1'] / n_pxl
    error_avg['err_delta2'] = error['err_delta2'] / n_pxl
    error_avg['err_delta3'] = error['err_delta3'] / n_pxl
    error_avg['err_log10'] = error['err_log10'] / n_pxl
    error_avg['err_whdr'] = error['err_whdr'] / eval_num

    return error_avg


def main(cfg,train_args,test_args):
    model = MetricDepthModel(cfg).to(cfg['device']) 
    if cfg['mode'] == 'train':
        optimizer = optim.SGD(model.parameters(), lr=train_args['lr'], momentum=0.9, weight_decay = 0.0005)
        criterion = nn.CrossEntropyLoss(weight=None, ignore_index = 151,reduction='elementwise_mean').to(cfg['device'])
        data_loader_train = CustomerDataLoader(cfg,train_args)
        data_loader_test = CustomerDataLoader(cfg,test_args)
        train(model,data_loader_train,data_loader_test,optimizer,criterion,cfg,train_args,test_args)

    if cfg['mode'] == 'test':
        data_loader = CustomerDataLoader(cfg,test_args)
        print(len(data_loader))
        test(model,data_loader,cfg,test_args)   


main(cfg,train_args,test_args)
