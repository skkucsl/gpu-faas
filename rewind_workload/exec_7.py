import sys, json
import boto3
import time
from PIL import Image, ImageFilter

TMP = '/tmp/'

def s3_connection():
    s3 = boto3.client(
        service_name="s3",
        region_name="ap-northeast-2", # 자신이 설정한 bucket region
        aws_access_key_id="AKIARPQUZ627QGCVDIWQ",
        aws_secret_access_key="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",
    )
    return s3

def s3_put_object(s3, bucket, filepath, access_key):
    s3.upload_file(
        Filename=filepath,
        Bucket=bucket,
        Key=access_key
    )

def s3_get_object(s3, bucket, filepath, access_key):
    s3.download_file(
        Filename=filepath,
        Bucket=bucket,
        Key=access_key
    )

def flip(image, file_name):
    path_list = []
    path = "flip-left-right-" + file_name
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(TMP + path)
    path_list.append(path)

    path = "flip-top-bottom-" + file_name
    img = image.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(TMP + path)
    path_list.append(path)

    return path_list


def rotate(image, file_name):
    path_list = []
    path = "rotate-90-" + file_name
    img = image.transpose(Image.ROTATE_90)
    img.save(TMP + path)
    path_list.append(path)

    path = "rotate-180-" + file_name
    img = image.transpose(Image.ROTATE_180)
    img.save(TMP + path)
    path_list.append(path)

    path = "rotate-270-" + file_name
    img = image.transpose(Image.ROTATE_270)
    img.save(TMP + path)
    path_list.append(path)

    return path_list


def img_filter(image, file_name):
    path_list = []
    path = "blur-" + file_name
    img = image.filter(ImageFilter.BLUR)
    img.save(TMP + path)
    path_list.append(path)

    path = "contour-" + file_name
    img = image.filter(ImageFilter.CONTOUR)
    img.save(TMP + path)
    path_list.append(path)

    path = "sharpen-" + file_name
    img = image.filter(ImageFilter.SHARPEN)
    img.save(TMP + path)
    path_list.append(path)

    return path_list


def gray_scale(image, file_name):
    path = "gray-scale-" + file_name
    img = image.convert('L')
    img.save(TMP + path)
    return [path]


def resize(image, file_name):
    path = "resized-" + file_name
    image.thumbnail((128, 128))
    image.save(TMP + path)
    return [path]

def image_processing(file_name, img_path):
    path_list = []
    timestamp = dict()
    timestamp['start'] = time.time()
    with Image.open(img_path) as image:
        timestamp['open'] = time.time()
        tmp = image
        path_list += flip(image, file_name)
        timestamp['flip'] = time.time()
        path_list += rotate(image, file_name)
        timestamp['rotate'] = time.time()
        path_list += img_filter(image, file_name)
        timestamp['filter'] = time.time()
        path_list += gray_scale(image, file_name)
        timestamp['gray'] = time.time()
        path_list += resize(image, file_name)
        timestamp['resize'] = time.time()
    
    return timestamp, path_list

def main(args):
    startTime = time.time()
    
    download_key = args.get("file", "dog.PNG")
    filepath = '/tmp/'+download_key

    s3 = s3_connection()
    s3_get_object(s3, 'bskim-s3-test', filepath, download_key)

    timestamp, file_list = image_processing(download_key, filepath)
    
    for upload_key in file_list:
        filepath = '/tmp/'+upload_key
        s3_put_object(s3, 'bskim-s3-test', filepath, upload_key)
    
    t8 = time.time()

    tmp = {"startTime": startTime, "functionTime": t8-startTime, "latency_detail": timestamp}
    print(json.dumps(tmp))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = dict()
    main(args)
