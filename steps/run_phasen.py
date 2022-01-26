
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import torch.optim as optim
import time
sys.path.append(os.path.dirname(sys.path[0]))
from run_config import Config
from model.phasen import PHASEN as Model
from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model
from tools.time_dataset import make_loader, DataReader

import soundfile as sf
import warnings
warnings.filterwarnings("ignore")


import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device_ids = [3, 4]

def train(model, args, device):
    print('preparing data...')
    dataloader, dataset = make_loader(
        args.tr_list,
        args.batch_size,
        num_workers=args.num_threads,
            )
    print_freq = 1000
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    # val_loss, val_sisnr = validation(model, args, lr, -1, device)
    for epoch in range(start_epoch, args.max_epoch):
        print('start training')
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        sisnr_total = 0.0
        sisnr_print = 0.0
        mix_loss_total = 0.0 
        mix_loss_print = 0.0 
        amp_loss_total = 0.0 
        amp_loss_print = 0.0
        phase_loss_total = 0.0
        phase_loss_print = 0.0

        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            optimizer.zero_grad()
            outputs, wav = data_parallel(model, (inputs,),device_ids=device_ids,output_device=device_ids[1])
            outputs = outputs.to(device)
            wav = wav.to(device)

            loss = model.loss(outputs, labels, mode='Mix')
            sisnr = model.loss(wav, labels, mode='SiSNR')
            loss[0].backward()
            # if epoch > 1:
            #     loss[0].backward()
            # else:
            #     loss_sisnr = -sisnr
            #     loss_sisnr.backward()
            # print('loss :',loss)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            step += 1
            # if torch.isnan(loss[0]).any():
            #     # print('jjjjjjjjjjjjj')
            #     continue

            mix_loss_total += loss[0].data.cpu()
            mix_loss_print += loss[0].data.cpu()
            
            amp_loss_total += loss[1].data.cpu()
            amp_loss_print += loss[1].data.cpu()
            
            phase_loss_total += loss[2].data.cpu()
            phase_loss_print += loss[2].data.cpu()
            
            sisnr_print += sisnr.data.cpu()
            sisnr_total += sisnr.data.cpu()

            del outputs, labels, inputs, loss, wav
            if (idx+1) % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, step, args.exp_dir)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                mix_loss_print_avg = mix_loss_print / print_freq
                amp_loss_print_avg = amp_loss_print / print_freq
                phase_loss_print_avg = phase_loss_print / print_freq
                sisnr_print_avg = sisnr_print / print_freq
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'
                      '{:2.3f}s/batches '
                      '| Mixloss {:2.4f}'
                      '| AMPloss {:2.4f}'
                      '| Phaseloss {:2.4f}'
                      '| SiSNR {:2.4f}'
                      .format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, 
                          mix_loss_print_avg,
                          amp_loss_print_avg,
                          phase_loss_print_avg,
                          -sisnr_print_avg
                    ))
                sys.stdout.flush()
                mix_loss_print = 0. 
                amp_loss_print = 0.
                phase_loss_print = 0. 
                sisnr_print = 0.

        eplashed = time.time() - stime
        mix_loss_total_avg = mix_loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
        print(
            'Training AVG.LOSS |'
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |'
            ' {:2.3f}s/batch | time {:3.2f}mins |'
            ' Mixloss {:2.4f}'
            ' SiSNR {:2.4f}'
            .format(
                                    epoch + 1, args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                                    mix_loss_total_avg,
                                    -sisnr_total_avg
                ))
        val_loss, val_sisnr = validation(model, args, lr, epoch, device)
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir, mode='best_model')
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    dataloader, dataset = make_loader(
            args.cv_list,
            args.batch_size,
            num_workers=args.num_threads,
        )
    model.eval()
    loss_total = 0. 
    sisnr_total = 0.
    num_batch = len(dataloader)
    stime = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, wav = data_parallel(model, (inputs, ),device_ids=device_ids)
            outputs = outputs.to(device)
            wav = wav.to(device)
            loss = model.loss(outputs, labels,mode='Mix')[0]
            sisnr = model.loss(wav, labels, mode='SiSNR')
            loss_total += loss.data.cpu()
            sisnr_total += sisnr.data.cpu()
            del loss, data, inputs, labels, wav, outputs
        etime = time.time()
        eplashed = (etime - stime) / num_batch
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
    print('CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} '
          '| lr {:.4e} | {:2.3f}s/batch| time {:2.1f}mins '
          '| Mixloss {:2.4f} | SiSNR {:2.4f}'.format(epoch + 1, args.max_epoch, lr, eplashed,
                                  (etime - stime)/60.0, loss_total_avg.item(), -sisnr_total_avg.item()))
    sys.stdout.flush()
    return loss_total_avg, sisnr_total_avg


def decode(model, args, device):
    model.eval()

    # If set true, there may be impulse noise 
    # in the border on segements.
    # If you can not bear it, you can set False

    decode_do_segement=True
    with torch.no_grad():
        
        data_reader = DataReader(
                args.tt_list,
            )
        output_wave_dir = os.path.join(args.exp_dir, 'rec_wav/')
        if not os.path.isdir(output_wave_dir):
            os.mkdir(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            inputs, utt_id, nsamples = data_reader[idx]
            
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            window = int(args.sample_rate*6) # 4s
            b,t = inputs.size()
            if t > int(1.5*window) and decode_do_segement:
                outputs = np.zeros(t)
                stride = int(window*0.75)
                give_up_length=(window - stride)//2
                current_idx = 0
                while current_idx + window < t:
                    tmp_input = inputs[:,current_idx:current_idx+window]
                    tmp_output = model(tmp_input,)[1][0].cpu().numpy()
                    if current_idx == 0:
                        outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

                    else:
                        outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
                    current_idx += stride 
                if current_idx < t:
                    tmp_input = inputs[:,current_idx:current_idx+window]
                    tmp_output = model(tmp_input)[1][0].cpu().numpy()
                    length = tmp_output.shape[0]
                    outputs[current_idx+give_up_length:current_idx+length] = tmp_output[give_up_length:]
            else:
                outputs = model(inputs)[1][0].cpu().numpy()
            outputs = outputs[:nsamples]
            # this just for plot mask 
            #amp, mask, phase = model(inputs)[2] 
            #np.save(utt_id, [amp.cpu().numpy(), mask.cpu().numpy(), phase.cpu().numpy()]) 
            sf.write(os.path.join(output_wave_dir, utt_id), outputs, args.sample_rate) 

        print('Decode Done!!!')


def main(args):
    # device = torch.device('cuda' if args.use_cuda else 'cpu')
    # device = torch.device("cuda:4")
    # torch.cuda.set_device(device)

    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(
        rnn_nums=args.rnn_nums,
        win_len=args.win_len,
        win_inc=args.win_inc,
        fft_len=args.fft_len,
        win_type=args.win_type
    )
    device = device_ids[0]
    model = model.cuda(device=device)
    # model.to(device)
    if not args.decode:
        train(model, FLAGS, device)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda)
    decode(model, args, device)

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint
def write_audio(file, data, sr):
    return sf.write(file, data, sr)
def process_predict(wav_path):
    def audioread(path):
        data, fs = sf.read(path)
        if len(data.shape) > 1:
            data = data[0]
        return data
    data = audioread(wav_path).astype(np.float32)
    inputs = np.reshape(data, [1, data.shape[0]])
    return inputs,data.shape[0]
def predict_one_wav(wav_path,use_cuda = True):
    args = Config()
    model = Model(
        rnn_nums=args.rnn_nums,
        win_len=args.win_len,
        win_inc=args.win_inc,
        fft_len=args.fft_len,
        win_type=args.win_type
    )
    checkpoint = load_checkpoint('/home/zmw/big_space/zhangmeiwei_space/asr_res_model/dns/phasen/model.ckpt-12.pt', use_cuda)
    model.load_state_dict(checkpoint['model'], strict=False)
    inputs,nsamples = process_predict(wav_path)
    inputs = torch.from_numpy(inputs)
    # inputs = inputs.to(device)
    window = int(16000 * 6)  # 4s
    b, t = inputs.size()
    if t > int(1.5 * window):
        outputs = np.zeros(t)
        stride = int(window * 0.75)
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window < t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            tmp_output = model(tmp_input, )[1][0].cpu().detach().numpy()
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[
                                                                                              give_up_length:-give_up_length]
            current_idx += stride
        if current_idx < t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            tmp_output = model(tmp_input)[1][0].cpu().detach().numpy()
            length = tmp_output.shape[0]
            outputs[current_idx + give_up_length:current_idx + length] = tmp_output[give_up_length:]
    else:
        outputs = model(inputs)[1][0].cpu().detach().numpy()
    outputs = outputs[:nsamples]
    sf.write('./test1212.wav', outputs, 16000)



if __name__ == "__main__":

    FLAGS = Config()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()

    # stringing
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)

    # testing
    # predict_one_wav('/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/noise_dev/BAC009S0724W0132_-5.725_zOtr4awwLLo_5.725.wav')
