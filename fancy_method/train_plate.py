# Basic func
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import glob
import yaml
import math
import time
import random

# My func
import test_plate  # import test.py to get mAP after each epoch
from models.yolo import Model
import utils.common_utils as common_utils
import utils.google_utils as google_utils
import utils.torch_utils as torch_utils
from utils.datasets import LoadImagesAndLabels


# def train(hyp):
def train(hyp, **kwargs):
    # 参数捕获
    # data = kwargs.get('')
    epochs = kwargs.get('epochs', 100)  # Int：迭代次数
    batch_size = kwargs.get('batch_size', 8)  # Int：分组大小
    cfg = kwargs.get('cfg', 'models/yolov5s.yaml')  # Str：模型配置路径
    data = kwargs.get('data', 'data/license_plate.yaml')  # Str：数据yaml文件路径
    img_size = kwargs.get('img_size', [640])  # [Int]：训练，预测图像大小
    rect = kwargs.get('rect', False)  # Bool：是否非正方形训练（即删除图像扩充成正方形时的灰色填充部分）
    resume = kwargs.get('resume', False)  # Bool：是否从上次训练点继续训练 last.pt
    nosave = kwargs.get('nosave', False)  # Bool：是否只储存最终训练点。默认保存best和last两个点
    notest = kwargs.get('notest', False)  # Bool：是否只测试最后一次传播
    noautoanchor = kwargs.get('noautoanchor', False)  # Bool：是否禁用自动描点（自动描点为初始化检测框，有助于加速训练）
    evolve = kwargs.get('evolve', False)  # Bool：是否使用遗传参数（yolo5默认参数）
    bucket = kwargs.get('bucket', '')  # Str：谷歌云盘，默认不用
    cache_images = kwargs.get('cache_images', False)  # Bool：是否提前缓存图片到内存，开启会加快训练速度
    weights = kwargs.get('weights', '')  # Str：初始化预训练权重路径 为空则从头训练
    name = kwargs.get('name', '')  # Str：重命名输出结果
    device = kwargs.get('device', 0)  # Int/Str：Cuda device 0, 1, 2, 3 or cpu or ''
    adam = kwargs.get('adam', False)  # Bool：是否使用Adam优化器
    multi_scale = kwargs.get('multi_scale', False)  # Bool：是否启用多尺度训练（开启后随机调整图像尺度进行训练，更稳定）
    single_cls = kwargs.get('single-cls', False)  # Bool：是否是单类别训练
    wdir = kwargs.get('wdir', 'weights' + os.sep)  # Str：weights 路径
    last = kwargs.get('last', wdir + 'last.pt')  # Str：last.pt默认路径
    best = kwargs.get('best', wdir + 'best.pt')  # Str：best.pt默认路径
    results_file = kwargs.get('results_file', 'results.txt')  # Str：结果文件默认路径

    # 参数操作
    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
        mixed_precision = False  # not installed

    os.makedirs(wdir, exist_ok=True)

    weights = last if resume else weights
    cfg = common_utils.check_file(cfg)  # check file
    data = common_utils.check_file(data)  # check file
    img_size.extend([img_size[-1]] * (2 - len(img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(device, apex=mixed_precision, batch_size=batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # Configure
    common_utils.init_seeds(1)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Create model
    model = Model(cfg).to(device)
    assert model.md['nc'] == nc, '%s nc=%g classes but %s nc=%g classes' % (data, nc,  cfg, model.md['nc'])

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [common_utils.check_img_size(x, gs) for x in img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0']) if adam else \
        optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Load Model
    google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
                % (weights,  cfg,  weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        start_epoch = ckpt['epoch'] + 1
        del ckpt

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # distributed backend
                                init_method='tcp://127.0.0.1:9999',  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, imgsz, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=rect,  # rectangular training
                                  cache_images=cache_images,
                                  single_cls=single_cls)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc,  cfg)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=cache_images,
                                                                 single_cls=single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = common_utils.labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = data_dict['names']

    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    if tb_writer:
        common_utils.plot_labels(labels)
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    if not noautoanchor:
        common_utils.check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Exponential moving average
    ema = torch_utils.ModelEMA(model)

    # Start training
    t0 = time.time()
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = common_utils.labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = common_utils.tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = common_utils.compute_loss(pred, targets.to(device), model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = 'train_batch%g.jpg' % i  # filename
                res = common_utils.plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not notest or final_epoch:  # Calculate mAP
            results, maps, times = test_plate.test(data,
                                                   batch_size=batch_size,
                                                   imgsz=imgsz_test,
                                                   save_json=final_epoch and data.endswith(os.sep + 'coco.yaml'),
                                                   model=ema.ema,
                                                   single_cls=single_cls,
                                                   dataloader=testloader,
                                                   fast=epoch < epochs / 2)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(name) and bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (bucket,  name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = common_utils.fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not nosave) or (final_epoch and not evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    n = name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                common_utils.strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2,  bucket)) if bucket and ispt else None  # upload

    if not evolve:
        common_utils.plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    # Hyper parameters
    hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 5e-4,  # optimizer weight decay
           'giou': 0.05,  # giou loss gain
           'cls': 0.58,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.20,  # iou training threshold
           'anchor_t': 4.0,  # anchor-multiple threshold
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 0.0,  # image rotation (+/- deg)
           'translate': 0.0,  # image translation (+/- fraction)
           'scale': 0.5,  # image scale (+/- gain)
           'shear': 0.0}  # image shear (+/- deg)
    print('------ hyper parameters ------', hyp)

    # Overwrite hyp with hyp*.txt (optional)
    f = glob.glob('hyp*.txt')
    if f:
        print('Using %s' % f[0])
        for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
            hyp[k] = v

    # Print focal loss if gamma > 0
    if hyp['fl_gamma']:
        print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

    # Train
    tb_writer = SummaryWriter()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    train(hyp)
