#!/usr/bin/env python
# -*- coding: gb18030 -*-

# Pw @ 2017-03-02 11:26:22
# @author XXX

import base64
import json
import urllib
import urllib2
import datetime
import time
import hashlib
import cStringIO
#import Image
import re 
import sys
import socket

class InnerToken:
    TOKEN_TYPE = '11'
    timestamp = ''
    def sign(self, appid, uid, sk):
        md5 = hashlib.md5()
        self.timestamp = str(time.mktime(datetime.datetime.now().timetuple())).split('.')[0]
        md5.update(self.timestamp + str(uid) + str(appid) + str(sk))
        return md5.hexdigest()

    def generateToken(self, appid, uid, sk):
        sign = self.sign(appid, uid, sk);
        token = self.TOKEN_TYPE + '.' + sign + '.' + self.timestamp + '.' + str(uid) + "-" + str(appid)
        return token

'''
def url_tesurllib2.HTTPError:t(url):
    if re.match(r'^https?:/{2}\w.+$',url):
        try:
            urllib2.urlopen(url,None,5)
            return 1
        except urllib2.HTTPError:
            return 0
        except urllib2.URLError:
            return 0
       # except Exception,e:
       #     return 0

'''
def classify_ans(url):
    #if url_test(url):
    try:
        html_data=urllib2.urlopen(url,None,10)
        file=cStringIO.StringIO(html_data.read())
    except urllib2.HTTPError:
        return 0,"error"
    except urllib2.URLError:
        return 0,"error"
    except socket.timeout:
        return 0,"error"
    except Exception:
        return 0,"error"
    else:
        #file=cStringIO.StringIO(html_data.read())
        img=file.read()
        file.close()
        postdata={    'image': base64.b64encode(img), }
        access_token=InnerToken()
        code=access_token.generateToken(9346036,'1884599608','F57nsFNlccehffKwiv5IM1KXx1Gr2i7h')
        strUrl='http://inner.openapi.baidu.com/rest/2.0/vis-classify/v1/classify/general?access_token='+code
        postData=urllib.urlencode(postdata)
        req=urllib2.Request(strUrl,postData)
        response=urllib2.urlopen(req).read()

        ans=json.loads(response)

        for key in ans:
            if key=="error_code":
                return 2,"error"
            if key=="result":
                return 1, ans[key]
    #except urllib2.HTTPError:
    #    return 0
    #except urllib2.URLError:
    #    return 0
    #else:
    #    return 0,"error"
    #img=urllib2.urlopen(url).read()
    #urllib.urlretrieve(url,'./Save_img.jpg')
    
    #file=open('baiheliang-5.jpg','rb')
    #img=file.read()
    #file.close()
    #img=Image.open(file)
    #binary_data=open(file,'rb')
 
    #img=binary.read()
    #print img
    #file.close()
    '''
    postdata={    'image': base64.b64encode(img), }    
    access_token=InnerToken()
    code=access_token.generateToken(9346036,'1884599608','F57nsFNlccehffKwiv5IM1KXx1Gr2i7h')
    strUrl='http://inner.openapi.baidu.com/rest/2.0/vis-classify/v1/classify/general?access_token='+code
    '''
    #print strUrl
    #access_token.generateToken(9346036,'1884599608','F57nsFNlccehffKwiv5IM1KXx1Gr2i7h')
    '''
    postData=urllib.urlencode(postdata)
    req=urllib2.Request(strUrl,postData)
    response=urllib2.urlopen(req).read()
    '''
    '''
    ans=json.loads(response)
    for key in ans:
        if key=="error_code":
            return 0,"error"
        if key=="result":
            return 1, ans[key]
    '''
    '''
            for each in ans[key]:
                for k,v in each.items():
                    if k=="class_name":
                        print '%s' % (v.encode('gb2312')),
                        #print type(v)
                    else:
                        print '\t%f' % (v),
                print
    '''

def sleeptime(hour,min,sec):
    return hour*3600+min*60+sec;
second=sleeptime(0,0,0.5)

for line in sys.stdin:
    line=line.strip()
    each_record=line.split('\t')
    #sign=each_record[0].strip()
    url=each_record[0].strip()
    #att=each_record[2].strip()
    flag,result=classify_ans(url)
    if flag==1:
        cache=""
        count=0
        for each_item in result:
            if count>0:
                break
            for k , v in each_item.items():
                if k=="class_name":
                    cache=cache+v.encode('gb2312')+':'
                else:
                    cache=cache+str(v)
            count=count+1
        #final_ans='   ***Class***  '+cache
        #print '%s\t%s\t%s\t%s' % (sign,url,att,final_ans)
        print '%s\t%s' % (url,cache)
    #else:
        #print '%d\t%s\t%s' % (flag,sign,url)
    time.sleep(second)
    #else:
        #print '%s\t%s' % (sign,url)

