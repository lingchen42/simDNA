# simDNA
**Visualize the grammatical pattern of DNA sequences learned by the CNNs** <br>
In computer vision, the neurons in the internal layers of convolutional neural networks (CNNs) have been show to learn [complex patterns]( http://yosinski.com/deepvis). Transcription factors cooperatively bind to regulatory sequences to regulate gene expression. However, it is unclear what grammartical rules they follow to orchestrate the regulatory sequences. Therefore, it is interesting to study the regulatory grammar through visualizing the CNNs trained on regulatory DNA sequences. Here we simulated tens of thousands enhancer sequences with complex regulatory grammar of transcription factor binding motifs and implemented CNN models and visualization strategies to retrieve the regualtory grammar.

Here is a example of retrieved regulatory grammar from interpreing the CNNs: <br>
<br>
![regulatory module heatmap](https://github.com/lingchen42/simDNA/blob/master/examples/retrieved_grammars.png)
