import os
import torch
import cv2
#from lib.core.config import cfg
import numpy as np
import torch.nn.functional as F
import dill

def bins_to_depth(cfg,depth_bin):
    """
    Transfer n-channel discrate depth bins to 1-channel conitnuous depth
    :param depth_bin: n-channel output of the network, [b, c, h, w]
    :return: 1-channel depth, [b, 1, h, w]
    """
    if type(depth_bin).__module__ != torch.__name__:
        depth_bin = torch.tensor(depth_bin, dtype=torch.float32).cuda()
    depth_bin = depth_bin.permute(0, 2, 3, 1) #[b, h, w, c]
    if type(cfg['DEPTH_BIN_BORDER']).__module__ != torch.__name__:
        cfg['DEPTH_BIN_BORDER'] = torch.tensor(cfg['DEPTH_BIN_BORDER'], dtype=torch.float32).cuda()
    depth = depth_bin * cfg['DEPTH_BIN_BORDER']
    depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
    depth = 10 ** depth
    depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]
    return depth


def resize_image(img, size):
    if type(img).__module__ != np.__name__:
        img = img.cpu().numpy()
    img = cv2.resize(img, (size[1], size[0]))
    return img


def kitti_merge_imgs(left, middle, right, img_shape, crops):
    """
    Merge the splitted left, middle and right parts together.
    """
    left = torch.squeeze(left)
    right = torch.squeeze(right)
    middle = torch.squeeze(middle)
    out = torch.zeros(img_shape, dtype=left.dtype, device=left.device)
    crops = torch.squeeze(crops)
    band = 5

    out[:, crops[0][0]:crops[0][0] + crops[0][2] - band] = left[:, 0:left.size(1)-band]
    out[:, crops[1][0]+band:crops[1][0] + crops[1][2] - band] += middle[:, band:middle.size(1)-band]
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[2][0] + crops[2][2]] += right[:, crops[1][0] + crops[1][2] - 2*band-crops[2][0]:]

    out[:, crops[1][0]+band:crops[0][0] + crops[0][2] - band] /= 2.0
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[1][0] + crops[1][2] - band] /= 2.0
    out = out.cpu().numpy()

    return out

def load_ckpt(args, model, optimizer=None, scheduler=None, val_err=[]):
    """
    Load checkpoint.
    """
    if os.path.isfile(args['load_ckpt']):
        print("loading checkpoint %s", args['load_ckpt'])
        checkpoint = torch.load(args['load_ckpt'], map_location=lambda storage, loc: storage, pickle_module=dill)
        for k,v in checkpoint.items():
            print(k)
        model.load_state_dict(checkpoint['model_state_dict'])
        if len(checkpoint) > 1:
            args['batchsize'] = checkpoint['batch_size']
            args['start_step'] = checkpoint['step']
            args['start_epoch'] = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if 'val_err' in checkpoint:  # For backward compatibility
                val_err[0] = checkpoint['val_err']
        del checkpoint
        torch.cuda.empty_cache()

def save_ckpt(batchsize,save_dir, step, epoch, model, optimizer, scheduler, val_err={}):
    """Save checkpoint"""
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'epoch%d_step%d.pth' %(epoch, step))
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'epoch': epoch,
        'batch_size': batchsize,
        'scheduler': scheduler.state_dict(),
        'val_err': val_err,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        save_name, pickle_module=dill)
    print('save model: %s', save_name)
