
quick and dirty implementation of mIOU for DeepLabV3 + ADE20K 31 classes 

1. prepare 512x512 jpegs and ground truth file. With the following script

```python
import tensorflow as tf
import deeplab.input_preprocess
from PIL import Image as Image

tf.enable_eager_execution()

ADE20K_PATH='/home/freedom/tf-models/research/deeplab/datasets/ADE20K/ADEChallengeData2016/'

for i in range(1, 2001):
    image_jpeg = ADE20K_PATH+f'images/validation/ADE_val_0000{i:04}.jpg'
    label_png = ADE20K_PATH+f'annotations/validation/ADE_val_0000{i:04}.png'
    # print(image_jpeg)
    image_jpeg_data = tf.io.read_file(image_jpeg)
    image_tensor = tf.io.decode_jpeg(image_jpeg_data)
    label_png_data = tf.io.read_file(label_png)
    label_tensor = tf.io.decode_jpeg(label_png_data)
    o_image, p_image, p_label = deeplab.input_preprocess.preprocess_image_and_label(image_tensor, label_tensor, 512, 512, 512, 512, is_training=False)

    target_image_jpeg = f'/tmp/ade20k_512/images/validation/ADE_val_0000{i:04}.jpg'
    target_label_png = f'/tmp/ade20k_512/annotations/validation/ADE_val_0000{i:04}.png'
    target_label_raw = f'/tmp/ade20k_512/annotations/raw/ADE_val_0000{i:04}.raw'
    
    resized_image = Image.fromarray(tf.reshape(tf.cast(p_image, tf.uint8), [512, 512, 3]).numpy())
    resized_image.save(target_image_jpeg)
    resized_label = Image.fromarray(tf.reshape(tf.cast(p_label, tf.uint8), [512, 512]).numpy(), 'L')
    resized_label.save(target_label_png)
    tf.reshape(tf.cast(p_label, tf.uint8), [512, 512]).numpy().tofile(target_label_raw)
```
we can get resized padded 512x512 jpeg files and corresponding ground truth files.
With tools such as Imagemagick we can convert jpeg files to bitmap files. Assuming we have .bmp
files in `/data/local/tmp/ade20k_512/images/bmp/` and raw ground truth files in
`/data/local/tmp/ade20k_512/annotations/raw/`.

2. We can build this little program by
`bazel build --config android_arm64 //tensorflow/lite/examples/deeplab:deeplabv3_ade20k_31`, push it
to an android device, and run it with something like
```
adb shell /data/local/tmp/deeplabv3_ade20k_31 \
  /data/local/tmp/freeze_quant_ops16_32c_clean.tflite
```
to get results like

```
IOU class 1: 51993428, 13954744, 9183336, 0.692032
IOU class 2: 37762466, 5061182, 4104980, 0.804679
IOU class 3: 36685612, 1579548, 1658101, 0.918903
IOU class 4: 21864135, 4787525, 3163743, 0.733317
IOU class 5: 17125399, 5039213, 3355424, 0.671057
IOU class 6: 13653436, 2592605, 1891551, 0.75277
IOU class 7: 13715374, 2438483, 2298392, 0.74329
IOU class 8: 6636985, 1013491, 909880, 0.775316
IOU class 9: 4697097, 2038848, 2150655, 0.52856
IOU class 10: 5755600, 2416509, 2263967, 0.55151
IOU class 11: 4272301, 1182732, 2476729, 0.538632
IOU class 12: 3166637, 1555335, 1889848, 0.478936
IOU class 13: 5684214, 1397072, 1189196, 0.687289
IOU class 14: 2917989, 3045641, 3989258, 0.29318
IOU class 15: 1572482, 1151868, 2645865, 0.292815
IOU class 16: 2494686, 1378419, 1688921, 0.448521
IOU class 17: 4979044, 1707100, 2147865, 0.563622
IOU class 18: 2563926, 1396489, 2996979, 0.368518
IOU class 19: 2438997, 700225, 1355891, 0.542589
IOU class 20: 2078830, 1090054, 1734781, 0.423934
IOU class 21: 2625704, 471434, 546160, 0.720694
IOU class 22: 2097794, 1035119, 1246680, 0.478993
IOU class 23: 1342227, 439067, 743923, 0.531529
IOU class 24: 1345220, 568035, 604890, 0.534211
IOU class 25: 1364801, 752967, 1231703, 0.407468
IOU class 26: 1365864, 883064, 1285886, 0.386403
IOU class 27: 2589354, 937528, 384436, 0.662016
IOU class 28: 726398, 259656, 922719, 0.380558
IOU class 29: 615945, 253399, 1017087, 0.326513
IOU class 30: 1376606, 1661232, 1439404, 0.307467
IOU class 31: 479282, 493527, 763861, 0.275978
mIOU over_all: 0.542623
```

