# YOLOv8m-Custom-Detection
fine tunning a pretrained model, YOLO v8m is used in this repo to train a custom data (transfer learning YOLO v8m)
## [google colab notebook link](https://colab.research.google.com/drive/1nJGCEivaMo_pkcFdzQYIhmeB8ofy0nyv?usp=sharing)
google colab is used to save your hard disk space and to speed up the training if you have slow GPU

## how to use
1- Download the google colab and upload it to you drive and open it to use google colab 
2- [this link to label your custom dataset](https://www.youtube.com/watch?v=LNwODJXcvt4)
3- if you want to freeze some layers:
this function used to freeze some layers and\
the variabke num_freeze is the number of needed layer to freeze
> def freeze_layer(trainer):
>  model = trainer.model
>  num_freeze = 10\
>  print(f"Freezing {num_freeze} layers")\
>  freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze\
>  for k, v in model.named_parameters():\
>      v.requires_grad = True  # train all layers\
>      if any(x in k for x in freeze):\
>          print(f'freezing {k}')\
>          v.requires_grad = False\
>  print(f"{num_freeze} layers are freezed.")

then add it to custom callback as follow
>>model = YOLO("yolov8m.pt")
>>model.add_callback("on_train_start", freeze_layer)
>>model.train(data='/content/drowsy-1/data.yaml')

NOTE: yolov8m.pt is the yolo medium version, you can use large and nano\
NOTE: data='/content/drowsy-1/data.yaml' is the path to my data.yaml
