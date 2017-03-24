#coding=gbk
import sys
import base64
import struct
from PIL import Image

'''
	对图片进行格式化，对应编码和解码双向的操作
'''

# 都统一转成RGB
# input: base64, output: image array
def b64_parser_array(image_b64, tmp_file = "tmp_file"):
	image_b64 = image_b64[len('data:image/jpeg;base64,'): ]
	read_data = base64.b64decode(image_b64)
	fout = open(tmp_file, 'wb')
	fout.write(read_data)
	fout.close()
	img = Image.open(tmp_file)
	# 都转成 RGB
	if img.mode != "RGB":
		img = img.convert('RGB')
	x, y = img.size
	z = len(img.getbands())
	#if x != 75 or y != 75 or z == 1:
	if x != 75 or y != 75 or z != 3:
		return None, None, None, None
	seq = img.getdata()
	ans = []
	for each_pix in list(seq):
		ans.append(each_pix[0])
		ans.append(each_pix[1])
		ans.append(each_pix[2])
	return x, y, z, ans

# TODO not run
def array_deparser_img(array, save_file):
	img = Image.fromarray(array)
	img.save(save_file, "PNG") # optimize = True
	#return img

# input: int list
# NOTICE: has \n
def ilist_parser_binary(int_list):
	binary_str = ""
	for each_int in int_list:
		binary_str += struct.pack("h", each_int)
	return binary_str

def binary_deparser_ilist(binary_str):
	num_int = len(binary_str) / 2
	int_list = struct.unpack(str(num_int) + 'h', binary_str)
	return list(int_list)

if __name__ == '__main__':
	#fout = open("b_file", 'w') # 跟 wb 一样
	fout = open("b_file", 'wb')
	for line in sys.stdin:
		kgid, query, pos, is_click, name, image_b64 \
				= line.strip('\r\n').split('\t')
		# TODO
		if int(pos) < 0 or int(pos) >= 12: continue

		x, y, z, seq = b64_parser_array(image_b64)
		if seq is None:
			sys.stderr.write("error[bad image]: %s\n" % kgid)
			continue
		binary_str = ilist_parser_binary(seq)
		fout.write("%s\t%s\t%s\t%s\t%s\t%s\n" \
				% (kgid, query, pos, is_click, \
				name, binary_str))
	fout.close()
