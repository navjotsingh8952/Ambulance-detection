```commandline 
cd ~/app/Ambulance-detection
source .venv/bin/activate
python main.py  --webcam /dev/video1 --serial /dev/ttyUSB0
```

If any issue with camera check port with below command
```commandline
v4l2-ctl --list-devices
```

If any issue with usb check the usb 
```commandline
ls /dev/tty*
```