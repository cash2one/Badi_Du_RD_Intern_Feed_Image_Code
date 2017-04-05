# -*- coding: utf8 -*-

import math

feature_map = {}

def get_similar_batch(target_file_path, cand_file_path, topN, result_file_path):
    feature_map = load_features(cand_file_path)
    target_file = open(target_file_path)
    result_file = open(result_file_path, 'w+')
    target_lines = target_file.readlines();
    target_lines_num = len(target_lines)
    count = 0
    for line in target_lines:
        img_name = line.split('\t')[0].strip()
        sim_img = get_similar_images(img_name, topN, cand_file_path)
        result_file.write('%s\t%s\n' % (img_name, str(sim_img)))
        count += 1
        print 'processed similar images %d/%d,  img_name=%s' % (count, target_lines_num,img_name)
    target_file.close()
    result_file.flush()
    result_file.close()

def get_similar_images(image_name, topN, feature_file_path):
    global feature_map
    if len(feature_map) == 0:
        feature_map = load_features(feature_file_path)
    if image_name not in feature_map:
        return []
    target_feature = feature_map[image_name]
    top_similar_images = {}
    for cand_name in feature_map:
        if cand_name == image_name:
           continue
        cand_feature = feature_map[cand_name]
        sim = map_multiply(target_feature, cand_feature)
        top_similar_images[cand_name] = sim
        # sort every 5000 items, and return topN items
        if len(top_similar_images) >= 5000:
            top_similar_images = sort_dict(top_similar_images, topN, 1, True)
    res = sorted(top_similar_images.iteritems(), key=lambda asd: asd[1], reverse=True)
    if len(res) <= topN:
        return res
    return res[0:topN]


def load_features(feature_file_path):
    global feature_map
    feature_file = open(feature_file_path)
    for line in feature_file:
        try:
            cols = line.strip().split('\t')
            image_name = cols[0].strip()
            feature = {}
            features_str = cols[1].strip()
            feature_arr = features_str.split(' ')
            size = len(feature_arr)/2
            #print 'feature size: %d' % size
            for i in range(size):
                feature[int(feature_arr[2*i])] = float(feature_arr[2*i+1])
            feature_map[image_name] = feature
            if len(feature_map) % 100 == 0:
                print 'load %d features' % len(feature_map)
        except Exception as e:
            print line
            print e
    print 'load features ok, total count: %d' % len(feature_map)
    return feature_map

def load_name_url_map(name_url_map_file_path):
    name_url_map = {}
    map_file = open(name_url_map_file_path)
    for line in map_file:
        cols = line.split('\t')
        name = cols[2]
        url = cols[3]
        name_url_map[name] = url
    print 'load name_url_map OK.'
    map_file.close()
    return name_url_map

def map_multiply(map1, map2):
    if len(map1) != len(map2):
        return -1
    product = 0
    map1_quadratic_sum = 0
    map2_quadratic_sum = 0
    for key in map1:
        if key in map2:
            product += map1[key] * map2[key]
        map1_quadratic_sum += map1[key] * map1[key]
    for key in map2:
        map2_quadratic_sum += map2[key] * map2[key]
    #print 'feature1 length: %f, feature2 length: %f' % (math.sqrt(map1_quadratic_sum), math.sqrt(map2_quadratic_sum))
    cos = product*1.0 / math.sqrt(map1_quadratic_sum) / math.sqrt(map2_quadratic_sum)
    return cos

def sort_dict(src_dict, topN, sort_by_key_or_value, is_reverse):
    if src_dict is None or len(src_dict) == 0:
        return None
    res = sorted(src_dict.iteritems(), key=lambda asd: asd[sort_by_key_or_value], reverse=is_reverse)
    new_dict = {}
    for i in range(topN):
        new_dict[res[i][0]] = res[i][1]
    return new_dict

def gen_html_file(similar_img_file_path, name_url_map_file_path, html_file_path):
    name_url_map = load_name_url_map(name_url_map_file_path)
    similar_img_file = open(similar_img_file_path)
    html_file = open(html_file_path, 'w+')
    head_str = '''<html><body width="100%"><table border="1" width="600px">'''
    html_file.write(head_str)

    count = 0
    valid_count = 0
    valid_sim_total_count = 0
    sim_pro_threshold = 0.8
    for line in similar_img_file:
        cols = line.split('\t')
        name = cols[0]
        if name not in name_url_map:
            continue
        url = name_url_map[name]
        similar_img_list = eval(cols[1])

        origin_tr = '''<tr><td><img src="%s" height="300" width="300" /></td><td>原图<br><br>sim_pro>=%f</td></tr>''' % (url, sim_pro_threshold)

        valid_sim_count = 0
        sim_img_urls = set()
        for sim_img in similar_img_list:
            if sim_img[0] not in name_url_map:
                continue
            sim_img_url = name_url_map[sim_img[0]]
            sim_pro = sim_img[1]
            if sim_pro >= sim_pro_threshold and sim_pro < 0.9999:
                if sim_img_url in sim_img_urls:
                    continue
                else:
                    sim_img_urls.add(sim_img_url)
                tr = '''<tr><td><img src="%s" height="300" width="300" /></td> \
                        <td style="word-wrap;break-word">similarity: %f</td></tr>''' \
                    % (sim_img_url, sim_pro)
                valid_sim_count += 1
                valid_sim_total_count += 1
                if valid_sim_count == 1:
                    html_file.write(origin_tr)
                    valid_count += 1
                html_file.write(tr)

        count += 1
        if valid_count >= 5000:
            break
        print 'processed %d images' % count
        if valid_sim_count > 0:
            tr = '''<tr><td></td><td style="word-wrap;break-word"><br><br><br><br></td></tr>'''
            html_file.write(tr)
    html_file.write('''</table></body></html>''')

    similar_img_file.close()
    html_file.flush()
    html_file.close()

    print 'sim_pro >= %f pic count: %d/%d' % (sim_pro_threshold, valid_count, count)
    print 'valid_sim_total_count = %d' % valid_sim_total_count
    print 'generate html file OK.'


if __name__ == '__main__':
    target_img_feature_file_path = '/Users/panping01/Documents/target_img_features.rtf'
    cand_img_feature_file_path = '/Users/panping01/Documents/target_img_features.rtf'

    sim_img_file_path = '/Users/panping01/Documents/sim_img_result.rtf'
    name_url_map_file_path = '/Users/panping01/Documents/name_url_map_total.rtf'
    html_file_path = '/Users/panping01/Documents/sim_img_html.html'

    #get_similar_batch(target_img_feature_file_path, cand_img_feature_file_path, 5, sim_img_file_path)
    gen_html_file(sim_img_file_path, name_url_map_file_path, html_file_path)
