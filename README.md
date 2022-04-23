# Catalan Text to Speech 🇪🇸

Based on Microsoft's [FastSpeech](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)

Catalan Version of [FastSpeech Repo](https://github.com/as-ideas/ForwardTacotron)

<p align="center">
  <img src="assets/model.png" width="700" />
</p>
<p align="center">
  <b>Figure 1:</b> Model Architecture.
</p>

The model has following advantages:
- **Robustness:** No repeats and failed attention modes for challenging sentences.
- **Speed:** The generation of a mel spectogram takes about 0.04s on a GeForce RTX 2080.
- **Controllability:** It is possible to control the speed of the generated utterance.
- **Efficiency:** In contrast to FastSpeech and Tacotron, the model of ForwardTacotron
does not use any attention. Hence, the required memory grows linearly with text size, which makes it possible to synthesize large articles at once.




Check out the latest [audio samples](https://mehdihosseinimoghadam.github.io//posts/2022/04/Catalan-Text-To-Speech/) (ForwardTacotron + WaveRNN)!


## 🔈 Samples

[Can be found here.](https://mehdihosseinimoghadam.github.io//posts/2022/04/Catalan-Text-To-Speech/)

The samples are generated with a model trained on 2 hoours of data from [Catalan Common Voice](https://commonvoice.mozilla.org/en/datasets) and vocoded with WaveRNN, [MelGAN](https://github.com/seungwonpark/melgan), or [HiFiGAN](https://github.com/jik876/hifi-gan). 
You can try out the latest pretrained model with the following notebook:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehdihosseinimoghadam/Catalan-Text-to-Speech/blob/master/Catalan_Text_To_Speeh_Demo.ipynb)




## References

* [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
* [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/abs/2006.06873)
* [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
* [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)

## Acknowlegements

* [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
* [https://github.com/seungwonpark/melgan](https://github.com/seungwonpark/melgan)
* [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
* [https://github.com/xcmyz/LightSpeech](https://github.com/xcmyz/LightSpeech)
* [https://github.com/resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
* [https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)

## Maintainers

* Mehdi Hosseini Moghadam, github: [cschaefer26](https://github.com/cschaefer26)
* Christian Schäfer, github: [mehdihosseinimoghadam](https://github.com/mehdihosseinimoghadam)

## Copyright

See [LICENSE](LICENSE) for details.
