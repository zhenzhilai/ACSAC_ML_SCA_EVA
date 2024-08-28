import urllib
import urllib.request
import urllib.error
import time
import os
from tqdm import tqdm
import argparse
import sys
import numpy as np


recons_dir = sys.argv[1]
output_path= sys.argv[2]
target_dir=sys.argv[4]
file_limit = int(sys.argv[3])
print("Reading from: ", recons_dir)
print("Output to: ", output_path)

def face_compare(http_url, key, secret, filename1, filename2, max_try=10):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filename1, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file1')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    fr = open(filename2, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file2')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s--\r\n' % boundary)
 
    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')
 
    http_body = b'\r\n'.join(data)
    req = urllib.request.Request(url=http_url, data=http_body)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try_nums = 0
    confidence = -2
    thresholds = None
    while True:
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            qrcont = resp.read()
            mydict = eval(qrcont)
            if len(mydict['faces1']) and len(mydict['faces2']):
                confidence = mydict['confidence']
                thresholds = mydict['thresholds']
            break
        except:
            if try_nums > max_try:
                print('Error!')
                confidence = -1
                break
            try_nums += 1
            # time.sleep(1)
    return confidence, thresholds

http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
key = ''
secret = ''

name_list = sorted(os.listdir(recons_dir))
if len(name_list) > file_limit:
    name_list = name_list[:file_limit]
total_num = 0
same_num = 0
correct_num = 0

counter = 0
confidences_array = []
thresholds_array = []


# for name in tqdm(name_list):
tbar = tqdm(range(len(name_list)), ncols=135)
# fix seed
np.random.seed(0)

for name in name_list:
    recons_path = os.path.join(recons_dir, name)
    target_path = os.path.join(target_dir, name)
    
    ## random target_path: If you want to evaluate Image A with Image B
    #while target_path == os.path.join(target_dir, name):
    #	target_path = os.path.join(target_dir, np.random.choice(name_list))
   
    # target_path = os.path.join(target_dir, name)

    # if counter % 3 == 0:
    #     time.sleep(1)
    time.sleep(0.1)
    results = face_compare(http_url, key, secret, recons_path, target_path)
    confidence = results[0]
    thresholds = results[1]

    confidences_array.append(confidence)
    thresholds_array.append(thresholds)
        
    if thresholds is None:
        total_num += 1
    elif confidence > thresholds["1e-3"]:
        correct_num += 1
        total_num += 1
    else:
        total_num += 1
    if total_num == 1000:
        print(sys.argv[1], '1k --> Acc: %f' % (correct_num / total_num))
    counter += 1

    if int(sys.argv[5]):
        tbar.set_description('Acc: %f, Results: %s' % (correct_num / total_num, str(results)))
    else:
        tbar.set_description('Acc: %f' % (correct_num / total_num))
    tbar.update(1)
tbar.close()
print(sys.argv[1], '10k --> Acc: %f' % (correct_num / total_num))

with open(output_path, 'w') as f:
    f.write("name confidence threshold0 threshold1 threshold2" + "\n")
    for i, name in enumerate(name_list):
        if confidences_array[i] is None:
            f.write(name + " " + str(confidences_array[i]) + " -99 -99 -99" + "\n")
        else:
            try:
                f.write(name + " " + str(confidences_array[i]) + " " + str(thresholds_array[i]["1e-3"]) + " " + str(thresholds_array[i]["1e-4"]) + " " + str(thresholds_array[i]["1e-5"]) + "\n")
            except:
                f.write(name + " " + str(confidences_array[i]) + " -99 -99 -99" + "\n")
