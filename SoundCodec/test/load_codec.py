from SoundCodec.codec import list_codec

if __name__ == '__main__':
    for codec_name in list_codec():
        print(codec_name)