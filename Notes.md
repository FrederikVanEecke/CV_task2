# Notes

- Use binary cross-entropy for multi-label classification, 
  not categorical cross-entropy (categorical CE doesn't take into account false
  positives) <https://towardsdatascience.com/keras-accuracy-metrics-8572eb479ec7>
- Use a sigmoid activation instead of softmax
- Use an [F-score](https://en.wikipedia.org/wiki/F-score) instead of the
  standard validation accuracy metric 
  <https://godatadriven.com/blog/keras-multi-label-classification-with-imagedatagenerator/#:~:text=Let's%20talk%20about%20metrics%20for%20a%20multi-label%20problem%20like%20this>
- Use class weights or sample weights when classes are unbalanced
  <https://androidkt.com/set-sample-weight-in-keras>
- Many papers use region hypotheses for multi-label classification of images
