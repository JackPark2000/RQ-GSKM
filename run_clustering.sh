#!/bin/bash
python3 clustering.py 512 key 1024
python3 clustering.py 512 value 1024
python3 clustering.py 128 key 1024
python3 clustering.py 128 value 1024
python3 clustering.py 32 key 1024
python3 clustering.py 32 value 1024
python3 clustering.py 8 key 1024
python3 clustering.py 8 value 1024

python3 clustering.py 8 key 256
python3 clustering.py 8 value 256
python3 clustering.py 32 key 256
python3 clustering.py 32 value 256
python3 clustering.py 128 key 256
python3 clustering.py 128 value 256
python3 clustering.py 512 key 256
python3 clustering.py 512 value 256
