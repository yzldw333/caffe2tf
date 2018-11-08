# caffe2tf
cffe2tf half-auto conversion tool

***

# Conversion steps
1. Step1
    - modify caffe2tf.py and run step1 function
    - open *.csv, [tf's var name, caffe's var name] are shown in first cols, make them as example.csv shows.
    - notice, gamma <--> scale0 , beta <--> scale1
2. Step2
    - run step2 function to generate pkl file.
3. load pkl in your training scripts!

