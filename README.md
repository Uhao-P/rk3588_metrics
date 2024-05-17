# 提供分类模型与目标检测模型的指标评估

## 分类模型目录结构

* data
  * 0
    * **.jpg
    * **.jpg
  * 1
    * **.jpg
    * **.jpg
  * 2
    * **.jpg
    * **.jpg
  * ...


## 检测模型目录结构

* data
  * imgs
    * 00001.jpg
    * 00002.jpg
    * 00003.jpg
    * 00004.jpg
    * ...
  * detections
    * 00001.txt
    * 00002.txt
    * 00003.txt
    * 00004.txt
    * ...
  * groundtruths
    * 00001.txt
    * 00002.txt
    * 00003.txt
    * 00004.txt
    * ...
  * res
    * 00001.jpg
    * 00002.jpg
    * 00003.jpg
    * 00004.jpg
    * ...

### detections文件夹下’00001.txt‘内容示例
```bash
# idClass  confidence  x  y  w  h
object .88 5 67 31 48
object .70 119 111 40 67
object .80 124 9 49 67
```
### groundtruths文件夹下’00001.txt‘内容示例
```bash
# idClass  x  y  w  h
object 25 16 38 56
object 129 123 41 62
```


## 分类模型准确率评估
> 需要按照上面介绍的‘分类模型目录结构’规范要验证的数据

```./classification   model_path   dataset_path```



## 检测模型准确率评估(以yolov5/7为例)
> 需要按照上面介绍的‘分类模型目录结构’规范要验证的数据

```./detection    model_path   dataset_path``` 

接下来请参考https://github.com/rafaelpadilla/Object-Detection-Metrics 中计算方式
