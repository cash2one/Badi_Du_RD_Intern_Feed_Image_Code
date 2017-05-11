import sys
import os



def transform(input_file_path, output_file_path):
    target_file = open(input_file_path, 'r')
    result_file = open(output_file_path, 'w+')

    for line in target_file:
        cols = line.strip().split('\t')
        #url = cols[0]
        label = cols[0] 
        b64 = cols[1]
        if label not in dic:
            continue
        else:
            result_file.write('%s\t%s\t%s\n' % (dic[label][0], dic[label][1], b64))
    result_file.flush()
    result_file.close()
    target_file.close()


if __name__ == '__main__':
    target_file_path = sys.argv[1]
    dictionary_path = './weight_label_50'
    convert_file_path = sys.argv[2]

    path_split_result = target_file_path.split('/')[2]
    convert_result_file_path = convert_file_path + path_split_result
    os.mkdir(convert_result_file_path)


    dic = {}
    with open(dictionary_path, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            trans_label = cols[0]
            text = cols[1]
            weight = cols[2]
            num = cols[3]
            original_label = cols[4]
            if original_label not in dic:
                dic[original_label] = [trans_label, weight]

    files = os.listdir(target_file_path)
    for item in files:
        absolute_path = os.path.join(target_file_path, item)
        result_file_path = os.path.join(convert_result_file_path, item)

        transform(absolute_path, result_file_path)
    

