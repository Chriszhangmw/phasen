


class Config:
    decode = 0 # if decode
    exp_dir = '/home/zmw/big_space/zhangmeiwei_space/asr_res_model/dns/phasen'
    tr_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/train.lst'
    cv_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/dev.lst'
    tt_list = ''
    learn_rate = 0.001
    max_epoch = 20
    rnn_nums = 300
    batch_size = 16
    use_cuda = 1
    seed = 2022
    num_threads = 10
    win_len = 400 #the window-len in enframe
    win_inc = 100 #the window include in enframe
    fft_len = 512 #the fft length when in extract feature
    win_type = 'hamming'
    num_gpu = 2
    weight_decay = 0.00001
    clip_grad_norm = 400.
    sample_rate = '16k'
    retrain = 0













