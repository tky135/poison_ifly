import base64
import hashlib
import hmac
import json
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import os
import requests
from generalRequest import Gen_req_url
import torchaudio
from tqdm import tqdm
import torch
import numpy as np
import pydub
# 填写在开放平台申请的APPID、APIKey、APISecret
# 相应编码音频base64编码后数据(不超过4M)

def gen_req_body(apiname, APPId, file_path=None, *args, **kwargs):
    """
    生成请求的body
    :param apiname
    :param APPId: Appid
    :param file_name:  文件路径
    :return:
    """
    if apiname == 'createFeature':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "createFeature",
                    "groupId": kwargs["groupId"],
                    "featureId": kwargs["featureId"],
                    "featureInfo": kwargs["featureInfo"],
                    "createFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'createGroup':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "createGroup",
                    "groupId": kwargs["groupId"],
                    "groupName": kwargs["groupName"],
                    "groupInfo": kwargs["groupInfo"],
                    "createGroupRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'deleteFeature':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3

            },
            "parameter": {
                "s782b4996": {
                    "func": "deleteFeature",
                    "groupId": kwargs["groupId"],
                    "featureId": kwargs["featureId"],
                    "deleteFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'queryFeatureList':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "queryFeatureList",
                    "groupId": kwargs["groupId"],
                    "queryFeatureListRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'searchFea':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "searchFea",
                    "groupId": kwargs["groupId"],
                    "topK": 2,
                    "searchFeaRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'searchScoreFea':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "searchScoreFea",
                    "groupId": kwargs["groupId"],
                    "dstFeatureId": kwargs["dstFeatureId"],
                    "searchScoreFeaRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'updateFeature':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "updateFeature",
                    "groupId": kwargs["groupId"],
                    "featureId": kwargs["featureId"],
                    "featureInfo": kwargs["featureInfo"],
                    "updateFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'deleteGroup':
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "deleteGroup",
                    "groupId": kwargs["groupId"],
                    "deleteGroupRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    else:
        raise Exception(
            "输入的apiname不在[createFeature, createGroup, deleteFeature, queryFeatureList, searchFea, searchScoreFea,updateFeature]内，请检查")
    return body

def req_url(APIKey, APISecret, body):
    """
    开始请求
    :param APPId: APPID
    :param APIKey:  APIKEY
    :param APISecret: APISecret
    :param file_path: body里的文件路径
    :return:
    """
    gen_req_url = Gen_req_url()
    # body = gen_req_body(apiname=apiname, APPId=APPId, file_path=file_path)
    request_url = gen_req_url.assemble_ws_auth_url(requset_url='https://api.xf-yun.com/v1/private/s782b4996', method="POST", api_key=APIKey, api_secret=APISecret)

    headers = {'content-type': "application/json", 'host': 'api.xf-yun.com', 'appid': '$APPID'}
    response = requests.post(request_url, data=json.dumps(body), headers=headers)
    tempResult = json.loads(response.content.decode('utf-8'))
    # print(tempResult)

    return tempResult

class iFlytek_SV(object):
    def __init__(self, enable_check=True, *args, **kwargs):
        self.APPId = kwargs['APPId']
        self.APISecret = kwargs['APISecret']
        self.APIKey = kwargs['APIKey']
        self.feature_set = {}
        self.group_created = False
        self.groupId = None
        self.groupName = None
        self.groupInfo = None

        self.check = enable_check

    def create_group(self, groupId, groupName, groupInfo):
        self.groupId = groupId
        self.groupName = groupName
        self.groupInfo = groupInfo

        body = gen_req_body(apiname='createGroup', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=None,

                            groupId=groupId,
                            groupName=groupName,
                            groupInfo=groupInfo)

        result = req_url(self.APIKey, self.APISecret, body)

        if result['header']['message'] == 'success':
            self.group_created = True
        else:
            raise Exception("create_group not successful")

        return self.parse_message(result)

    def create_feature(self, file_path, featureId, featureInfo):
        '''
            create a group before creating features
        '''
        if self.check:
            assert self.group_created, \
                        "No group exists."
        
        body = gen_req_body(apiname='createFeature', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=file_path,

                            groupId=self.groupId,
                            featureId=featureId,
                            featureInfo=featureInfo)

        result = req_url(self.APIKey, self.APISecret, body)

        if result['header']['message'] == 'success':
            self.feature_set[featureId] = {
                'file': file_path,
                'featureInfo': featureInfo
            }

        return self.parse_message(result)

    def delete_feature(self, featureId):
        if self.check:
            assert self.group_created, \
                        "No group exists."
            assert featureId in self.feature_set, \
                        f"The featureId does not exist: {featureId}"

        body = gen_req_body(apiname='deleteFeature', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=None,

                            groupId=self.groupId,
                            featureId=featureId)

        result = req_url(self.APIKey, self.APISecret, body)

        if result['header']['message'] == 'success':
            self.feature_set.pop(featureId)

        return self.parse_message(result)

    def query_feature_list(self):
        if self.check:
            assert self.group_created, \
                        "No group exists."
            assert self.feature_set, \
                        "No feature exists."

        body = gen_req_body(apiname='queryFeatureList', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=None,

                            groupId=self.groupId)

        result = req_url(self.APIKey, self.APISecret, body)

        return self.parse_message(result)

    def search_feature(self, file_path):
        if self.check:
            assert self.group_created, \
                        "No group exists."

        body = gen_req_body(apiname='searchFea', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=file_path,

                            groupId=self.groupId)

        result = req_url(self.APIKey, self.APISecret, body)

        return self.parse_message(result)

    def search_score_feature(self, file_path, dstFeatureId):
        if self.check:
            assert self.group_created, \
                        "No group exists."

        body = gen_req_body(apiname='searchScoreFea', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=file_path,

                            groupId=self.groupId,
                            dstFeatureId=dstFeatureId)

        result = req_url(self.APIKey, self.APISecret, body)

        return self.parse_message(result)

    def update_feature(self, file_path, featureId, featureInfo):
        if self.check:
            assert self.group_created, \
                        "No group exists."
            assert featureId in self.feature_set, \
                        f"The featureId does not exist: {featureId}"

        body = gen_req_body(apiname='updateFeature', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=file_path,

                            groupId=self.groupId,
                            featureId=featureId,
                            featureInfo=featureInfo)

        result = req_url(self.APIKey, self.APISecret, body)

        if result['header']['message'] == 'success':
            self.feature_set[featureId] = {
                'file': file_path,
                'featureInfo': featureInfo
            }

        return self.parse_message(result)

    def delete_group(self):
        if self.check:
            assert self.group_created, \
                        "No group exists."

        body = gen_req_body(apiname='deleteGroup', 
                            APPId=self.APPId,
                            # APIKey=self.APIKey, 
                            # APISecret=self.APISecret, 
                            file_path=None,

                            groupId=self.groupId)

        result = req_url(self.APIKey, self.APISecret, body)

        if result['header']['message'] == 'success':
            self.feature_set = {}
            self.group_created = False
            self.groupId = None
            self.groupName = None
            self.groupInfo = None

        return self.parse_message(result)

    def parse_message(self, result):
        if result['header']['message'] == 'success':
            key = list(result['payload'].keys())[0]
            result['payload'][key]['text'] = base64.b64decode(result['payload'][key]['text'])
            # print(result)
            return result
        else:
            # print(result)
            return result



    def enroll_all(self, enrollset):
        if self.check:
            assert self.group_created, \
                        "No group exists."

        if self.feature_set:
            print("Warning: already enrolled.")
            print(self.feature_set)

        for spk, uttes in tqdm(enrollset.items()):
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple())).replace(' ', '_').replace(',', '')
            self.create_feature(uttes[0], spk, spk+'_'+date)

    def get_ifly_score(self, testset, n_same_dict):
        thesame = []
        thediff = []

        print("Getting scores...")
        for spk, uttes in tqdm(testset.items()):
            for u in uttes[:n_same_dict[spk]]:
                # featureId has to be consistent with spk
                raw = safe_request(self.search_score_feature, u, spk)
                result = self.parse_message(raw)
                if result:
                    result = eval(result)
                    thesame.append(float(result['score']))

                else:
                    pass

            for u in uttes[n_same_dict[spk]:]:
                # featureId has to be consistent with spk
                raw = safe_request(self.search_score_feature, u, spk)
                result = self.parse_message(raw)
                if result:
                    result = eval(result)
                    thediff.append(float(result['score']))

                else:
                    pass
        import numpy as np
        return np.array(thesame), np.array(thediff) 

def safe_request(func, *args, **kwargs):

    count = 0
    while True:
        try:
            raw = func(*args, **kwargs)
        except Exception as e:
            count += 1
            print(f"function ({func}) error: {e}")
            print(f"Retrying {count} time(s)")
        else:
            break
    return raw


ifly  = iFlytek_SV(enable_check=True, APIKey = "8ad9c4b60301c12b0d02b7c45f07df14", APPId = "b87c286e", APISecret = "M2FiYWFlYjY5NDAyNjk4OGI2ZDE2OWMz")
ifly.create_group(groupId="poison_attack", groupName="poison_attack", groupInfo="poison_attack")
ifly.create_feature(file_path="mp3s/attacker.mp3", featureId="attacker", featureInfo="attacker")
ifly.create_feature(file_path="mp3s/victim.mp3", featureId="victim", featureInfo="victim")


def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[0] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def get_loss(tensor):
    """
    tensor must be 2-d with [N, D]
    """
    tensor = tensor.cpu()
    N, D = tensor.shape
    # generate mp3s
    if not os.path.exists("mp3s"):
        os.mkdir("mp3s")

    losses = []    
    for i in range(tensor.shape[0]):
        tensor[i] = tensor[i] / max(tensor[i].max(), -tensor[i].min()) * np.iinfo(np.int16).max / (-np.iinfo(np.int16).min)
        # write(os.path.join("mp3s", str(i) + ".mp3"), 16000, tensor[i], normalized=True)
        # scale to [-1, 1]
        torchaudio.save(filepath=os.path.join("mp3s", str(i) + ".mp3"), src=tensor[i].unsqueeze(0), sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")

        result = ifly.search_score_feature(file_path=os.path.join("mp3s", str(i) + ".mp3"), dstFeatureId="attacker")
        code = int(result['header']['code'])
        if code != 0:
            print("Error code: %d at %d.mp3, min: %f, max: %f" %(code, i, tensor[i].min(), tensor[i].max()))
            losses.append(torch.nan)
            continue
        attacker_score = json.loads(result['payload']['searchScoreFeaRes']['text'].decode("UTF-8"))["score"]

        result = ifly.search_score_feature(file_path=os.path.join("mp3s", str(i) + ".mp3"), dstFeatureId="victim")
        code = int(result['header']['code'])
        if code != 0:
            print("Error code: %d at %d.mp3, min: %f, max: %f" %(code, i, tensor[i].min(), tensor[i].max()))
            losses.append(torch.nan)
            continue
        victim_score = json.loads(result['payload']['searchScoreFeaRes']['text'].decode("UTF-8"))["score"]

        loss = (1 - victim_score) * 0.7 + (1 - attacker_score) * 0.3
        # print("file: %d.mp3, attacker score: %f, victim_score: %f, loss: %f, min: %f, max: %f" % (i, attacker_score, victim_score, loss, tensor[i].min(), tensor[i].max()))
        losses.append(loss)
    # print(losses)
    return torch.tensor(np.array(losses), dtype=torch.float32).reshape(N, 1)

def get_loss_fast(tensor):
    """
    tensor must be 2-d with [N, D]
    """
    tensor = tensor.cpu()
    N, D = tensor.shape
    # generate mp3s
    if not os.path.exists("mp3s"):
        os.mkdir("mp3s")

    losses = []    
    for i in range(tensor.shape[0]):
        tensor[i] = tensor[i] / max(tensor[i].max(), -tensor[i].min()) * np.iinfo(np.int16).max / (-np.iinfo(np.int16).min)
        # write(os.path.join("mp3s", str(i) + ".mp3"), 16000, tensor[i], normalized=True)
        # scale to [-1, 1]
        torchaudio.save(filepath=os.path.join("mp3s", str(i) + ".mp3"), src=tensor[i].unsqueeze(0), sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")

        # result = ifly.search_score_feature(file_path=os.path.join("mp3s", str(i) + ".mp3"), dstFeatureId="attacker")
        result = safe_request(ifly.search_feature, file_path=os.path.join("mp3s", str(i) + ".mp3"))
        code = int(result['header']['code'])
        if code != 0:
            print("Error code: %d at %d.mp3, min: %f, max: %f" %(code, i, tensor[i].min(), tensor[i].max()))
            losses.append(torch.nan)
            continue
        score_list = json.loads(result['payload']['searchFeaRes']['text'].decode("UTF-8"))["scoreList"]

        if len(score_list) != 2:
            raise Exception("length incorrect")

        for dict in score_list:
            if dict['featureId'] == 'victim':
                victim_score = dict['score']
            elif dict['featureId'] == 'attacker':
                attacker_score = dict['score']
            else:
                raise Exception("Not Implemented")
        loss = (1 - victim_score) * 0.7 + (1 - attacker_score) * 0.3
        # print("(", victim_score, attacker_score, "%.2f" % loss, ")", end="\t", flush=True)
        # print("file: %d.mp3, attacker score: %f, victim_score: %f, loss: %f, min: %f, max: %f" % (i, attacker_score, victim_score, loss, tensor[i].min(), tensor[i].max()))
        losses.append(loss)
    # print(losses)
    # print('\n')
    return torch.tensor(np.array(losses), dtype=torch.float32).reshape(N, 1)

if __name__ == "__main__":

    import torchaudio
    import matplotlib.pyplot as plt
    # victim enroll
    # victim_file = "NES/victim_enroll/2961/2961-960-0000.flac"
    # print(torchaudio.info(victim_file))
    # print(torchaudio.load(victim_file)[0].shape)
    # torchaudio.save(filepath="mp3s/victim.mp3", src=torchaudio.load(victim_file)[0], sample_rate=16000,  channels_first=True, compression=-4.5, format="mp3")
    # print(torchaudio.info("mp3s/victim.mp3"))


    # # # attacker enroll
    # attacker_file = "NES/attacker/6930/6930-75918-0001.flac"
    # print(torchaudio.info(attacker_file))
    # print(torchaudio.load(attacker_file)[0].shape)
    # torchaudio.save(filepath="mp3s/attacker.mp3", src = torchaudio.load(attacker_file)[0], sample_rate=16000, channels_first=True, compression=-4.5, format="mp3")
    # print(torchaudio.info("mp3s/attacker.mp3"))

    # # attacker test
    # attacker_file = "flacs/attacker/6930/6930-75918-0002.flac"
    # print(torchaudio.info(attacker_file))
    # print(torchaudio.load(attacker_file)[0].shape)
    # torchaudio.save(filepath="mp3s/tester.mp3", src = torchaudio.load(attacker_file)[0], sample_rate=16000, channels_first=True, compression=256.5)
    # print(torchaudio.info("mp3s/tester.mp3"))


    ifly  = iFlytek_SV(enable_check=True, APIKey = "8ad9c4b60301c12b0d02b7c45f07df14", APPId = "b87c286e", APISecret = "M2FiYWFlYjY5NDAyNjk4OGI2ZDE2OWMz")
    # ifly.create_group(groupId="poison_attack", groupName="poison_attack", groupInfo="poison_attack")
    # ifly.create_feature(file_path="mp3s/attacker.mp3", featureId="attacker", featureInfo="attacker")
    # ifly.create_feature(file_path="mp3s/victim.mp3", featureId="victim", featureInfo="victim")
    # ifly.query_feature_list()
    result = ifly.search_feature(file_path="mp3s/42.mp3")
    print(result)
    # print(json.loads(result['payload']['searchFeaRes']['text'].decode('UTF-8')))
    # ifly.create_feature(file_path="mp3s/450.mp3", featureId="test", featureInfo="test")
    # while 1:
    #     result = ifly.search_score_feature(file_path="mp3s/500.mp3", dstFeatureId="attacker")

    #     if result['header']['code'] == 0:
    #         break
    #     else:
    #         print("Error, retrying...")

    # ifly.search_score_feature(file_path="mp3s/432.mp3", dstFeatureId="victim")
    # ifly.search_feature("mp3s/449.mp3")
    # ifly.delete_group()

