



new = open('/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_0.lst','w',encoding='utf-8')

with open('/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test.lst','r',encoding='utf-8') as f:
    data = f.readlines()
    f.close()
noise = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/noise_test_0'
clean = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/clean_test_0'
for line in data:
    line = line.strip()
    line = line.replace('./clean_test_0/','')
    n = noise + '/' + line
    c = clean + '/' + line
    new.write(n + ' ' + c + '\n')







