import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
import cannyedge.canny as canny
import hed.run as hed

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        #self.scheduler = utility.make_scheduler(args, self.optimizer)

        # if self.args.load != '':
        #     self.optimizer.load_state_dict(
        #         torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
        #     )
        #     for _ in range(len(ckp.log)): self.scheduler.step()

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step() #self.scheduler.step()
        epoch = self.optimizer.get_last_epoch() + 1 #epoch = self.scheduler.last_epoch + 1
        print('PrintEpochTrain: ', epoch, self.optimizer.get_last_epoch())
        lr = self.optimizer.get_lr() #lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        idx_scale = 0
        timer_data, timer_model = utility.timer(), utility.timer()
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            ##added Test
            # B, C, H, W = lr.size() 
            # for bt in range(len(lr)):
            #     edge0 = canny.canny(lr[bt,:,:,:], use_cuda=True) 
            #     tt = edge0.reshape([1, 1, H, W])
            #     tt = torch.Tensor(tt)
            #     if bt > 0:
            #         t2 = torch.cat((tt, t2), dim=0)
            #     else:
            #         t2 = tt
            # edge_map = t2.cuda()
            B, C, H, W = lr.size() 
            for bt in range(len(lr)):
                #edge0 = hed.hedResult(lr[bt,:,:,:]) 
                edge0 = canny.canny(lr[bt,:,:,:], use_cuda=True) 
                tt = edge0.reshape([1, 1, H, W])
                tt = torch.Tensor(tt)
                if bt > 0:
                    t2 = torch.cat((tt, t2), dim=0)
                else:
                    t2 = tt
            edge_map = t2.cuda()
            #######

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(lr, edge_map, idx_scale)
            loss = self.loss(sr, hr)
            # if loss.item() < self.args.skip_threshold * self.error_last:
            #     loss.backward()
            #     self.optimizer.step()
            # else:
            #     print('Skip this batch {}! (Loss: {})'.format(
            #         batch + 1, loss.item()
            #     ))
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        epoch = self.optimizer.get_last_epoch() #+ 1 #epoch = self.scheduler.last_epoch + 1
        print('PrintEpochTest: ', epoch, self.optimizer.get_last_epoch())
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test[0].dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test[0], ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                     ##added Test
                    # B, C, H, W = lr.size() 
                    # for bt in range(len(lr)):
                    #     edge0 = canny.canny(lr[bt,:,:,:], use_cuda=True) 
                    #     tt = edge0.reshape([1, 1, H, W])
                    #     tt = torch.Tensor(tt)
                    #     if bt > 0:
                    #         t2 = torch.cat((tt, t2), dim=0)
                    #     else:
                    #         t2 = tt
                    # edge_map = t2.cuda()
                    B, C, H, W = lr.size() 
                    for bt in range(len(lr)):
                        #edge0 = hed.hedResult(lr[bt,:,:,:]) 
                        edge0 = canny.canny(lr[bt,:,:,:], use_cuda=True) 
                        tt = edge0.reshape([1, 1, H, W])
                        tt = torch.Tensor(tt)
                        if bt > 0:
                            t2 = torch.cat((tt, t2), dim=0)
                        else:
                            t2 = tt
                    edge_map = t2.cuda()
                    #######
                    sr = self.model(lr,edge_map, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=self.loader_test[0]
                        )
                        # eval_acc += utility.calc_psnr(
                        #     sr, hr, scale, self.args.rgb_range,
                        #     benchmark=self.loader_test[0].dataset.benchmark
                        # )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(self.loader_test[0], filename[0], save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test[0]) #self.ckp.log[-1, idx_data, idx_scale] = eval_acc / len(self.loader_test[0]) 
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][ idx_scale],
                        best[1][ idx_scale] + 1
                    )
                )

        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
        
        self.ckp.write_log('Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)


    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1 #self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

