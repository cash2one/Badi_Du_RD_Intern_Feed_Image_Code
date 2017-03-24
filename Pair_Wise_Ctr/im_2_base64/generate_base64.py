#coding=gbk
import sys
import os
import base64
from PIL import Image
import json

if len(sys.argv) < 4:
	print 'Usage: python %s input_file output_file image_storage_dir' % sys.argv[0]
	sys.exit(1)

def get_base64(file_name):
	fin = open(file_name)
	data = fin.read()
	fin.close()
	return 'data:image/jpeg;base64,' + base64.b64encode(data)

fin = open(sys.argv[1])
lines = fin.readlines()
fin.close()


fout = open(sys.argv[2], 'w')
view_cnt = 0
for line in lines:
	if view_cnt % 1000 == 0:
		print view_cnt
	view_cnt += 1
	kgid, value = line.strip('\r\n').split('\t')
	json_value = json.loads(value)

	name = json_value["name"].encode('gb18030')
	try:
		small_img_url = json_value["75_75"].encode('gb18030')
	except:
		print 'error[75_75 no find]: ' + kgid
		continue

	tmp_original_file = sys.argv[3] + '/' + kgid + '.png'
	#tmp_optimize_file = sys.argv[3] + '/' + kgid + '_compressed.png'

	cmd_line = 'wget -t 5 -T 20 --referer="www.baidu.com" "' \
			+ small_img_url + '" -O ' \
			+ tmp_original_file
	ret = os.system(cmd_line)
	if ret != 0:
		# 重试
		ret = os.system(cmd_line)
		if ret != 0:
			print 'error[wget]: ' + kgid + '\t' + small_img_url
			continue
	#img = Image.open(tmp_original_file)
	#img.save(tmp_optimize_file, "PNG", optimize = True)
	fout.write("%s\t%s\t%s\n" % (kgid, name, \
			get_base64(tmp_original_file)))
fout.close()
