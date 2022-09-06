def read_file(file_path):
    with open(file_path,'r', encoding='utf-8') as f:
        res = f.readlines()
    return [x.strip() for x in res]

def write_to_file(str_list,file_path):
    with open(file_path,'w', encoding='utf-8') as f:
        f.writelines([x+'\n' for x in str_list])

def main():
    ori_test_in = read_file('train/intent_seq_ori.in')
    ori_test_out = read_file('train/intent_seq_ori.out')

    tar_test_in = []
    tar_test_out = []

    for i,o in zip(ori_test_in, ori_test_out):
        intent_label = o.split()[0]
        if '#' not in intent_label:
            tar_test_in.append(i)
            tar_test_out.append(o)

    write_to_file(tar_test_in, 'train/intent_seq.in')
    write_to_file(tar_test_out, 'train/intent_seq.out')

main()