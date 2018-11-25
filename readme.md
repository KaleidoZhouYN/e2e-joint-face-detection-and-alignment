replace your picture in '/picture/demo.jpg' and run 'python predict_m.py'

the result is  '/picture/result_demo.jpg'

some of the predict code is copy and rewrite from:[faceboxes](https://github.com/XiaXuehai/faceboxes)

wider result:

|easy|medium|hard|
|------|------|------|
|0.846|0.840|0.455|

demo:
![pnet_2_299.pt](/picture/result_demo.jpg)


update @ 2018/11/25:

we add rotation augmentation and train a new model,these model run the same time but more robust to each rotation of the picture:

demo:

![pnet_rotate.pt](/picture/result_4_r2.jpg)

usage:

replace the checkpoint to /weight/msos_pnet_rotate.pt & /weight/msos_onet_rotate.pt 
