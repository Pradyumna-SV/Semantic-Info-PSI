from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
from .tools.PairwiseSemanticInformation import PSI
from .tools.NearestneighborContrastiveLoss import NNConLoss
from .tools.nonGraphConNet import GCN


class Trainer_teacher(object):
    def __init__(self, encoder,args, memory=None):
        super(Trainer_teacher, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.Lambda1 = args.Lambda1
        self.Lambda2 = args.Lambda2
        self.loss1 = args.nncl
        self.loss2 = args.rcl
        self.use_gcn = args.gcn
        if args.nncl:
            self.nncl = NNConLoss(k=args.knn)
        if args.rcl:
            self.psi = PSI(sigma=1, delta=1, topk=10)
        if self.use_gcn:
            self.gcn_run = GCN(k1=args.gcnk1, k2=args.gcnk2)
        print('Lambda1, Lambda2, nncl, rcl, gcn, knn, gcnk1, gcnk2:',
              args.Lambda1, args.Lambda2, args.nncl, args.rcl, args.gcn, args.knn, args.gcnk1, args.gcnk2)
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels,cids, indexes = self._parse_data(inputs)

            # forward
            f_out, f_out_up, f_out_down = self._forward(inputs)
            if self.use_gcn:
                feat_t_g = self.gcn_run(f_out.detach(), cids.cpu().numpy())
            else:
            	feat_t_g = f_out.detach()



            loss = self.memory(f_out, f_out_up, f_out_down, labels, epoch)
            if self.loss1:
            	loss = loss + self.nncl(f_out, feat_t_g) * self.Lambda1
            else:
                loss = loss + self.psi(f_out, feat_t_g) * self.Lambda2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, camids, indexes = inputs
        return imgs.cuda(), pids.cuda(),camids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class Trainer(object):
    def __init__(self, encoder, encoder_teacher,args, memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.encoder_teacher = encoder_teacher
        self.memory = memory
        self.Lambda1 = args.Lambda1
        self.Lambda2 = args.Lambda2
        self.loss1 = args.nncl
        self.loss2 = args.rcl
        self.use_gcn = args.gcn
        if args.nncl:
            self.nncl = NNConLoss(k=args.knn)
        if args.rcl:
            self.psi = PSI(sigma=1, delta=1, topk=10)
        if self.use_gcn:
            self.gcn_run = GCN(k1=args.gcnk1, k2=args.gcnk2)
        print('Lambda1, Lambda2, nncl, rcl, gcn, knn, gcnk1, gcnk2:',
              args.Lambda1, args.Lambda2, args.nncl, args.rcl, args.gcn, args.knn, args.gcnk1, args.gcnk2)

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        self.encoder_teacher.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels,cids,indexes = self._parse_data(inputs)

            # forward
            f_out, f_out_up, f_out_down = self._forward(inputs)
            if self.use_gcn:
                feat_t_g = self.gcn_run(f_out.detach(), cids.cpu().numpy())
            else:
                feat_t_g = f_out.detach()
            with torch.no_grad():
                f_out_teacher, f_out_up_teacher, f_out_down_teacher = self.encoder_teacher(inputs)


            loss = self.memory(f_out, f_out_up, f_out_down, f_out_teacher, f_out_up_teacher, f_out_down_teacher, labels, epoch)
            if self.loss1:
                loss = loss + self.nncl(f_out, feat_t_g) * self.Lambda1
            if self.loss2:
                loss = loss + self.psi(f_out, feat_t_g) * self.Lambda2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, camids, indexes = inputs
        return imgs.cuda(), pids.cuda(),camids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


