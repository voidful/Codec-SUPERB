from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SoundCodec',
    version='1.7',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'all': [
            'funasr',
            's3tokenizer',
            'bigcodec',
            'encodec',
            'descript-audio-codec',
            'funcodec @ git+https://github.com/voidful/FunCodec.git',
            'AudioDec @ git+https://github.com/voidful/AudioDec.git',
            'wavtokenizer @ git+https://github.com/voidful/WavTokenizer.git'
        ]
    }
)
