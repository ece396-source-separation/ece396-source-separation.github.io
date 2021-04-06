---
layout: default
---
<img class="top-image" src="assets/images/cocktail.webp"> 

Imagine walking into a crowded, noisy restaurant - 

Well, these are still quarantine times, so this type of scenario is less common, but presumably, when life returns to normal such a scenario could occur.

You sit across the table from your friend. It's noisy, with many other conversations going on around you, making it rather difficult to hear what she is saying.

## Cocktail Party Problem

The Cocktail Party Effect is the, "ability for people to focus their auditory attention on one source," whether that be a friend at a party, or a waiter in a restaurant. It has been said that listeners are able to segregate the different audio sources, and tune into one, as opposed to the others[^1]. 

However, the ability to "tune in" to a single voice is highly dependent on a number of features, including speaker pitch, location, rate of speech, and the listener's hearing capability. In other words: different people, when presented with a situation with multiple speakers, will only be able to pick out a selection of the words that are being spoken, and not necessarily only from the speaker of interest.

Moreover, if a person only has one functional ear, or is hard of hearing, the task becomes even more difficult: with only one ear, it is difficult to determine locality of the speaker, and when hard of hearing, all sounds come through with limited fidelity.

So what if there were ways that we could make a device that could 'tune out' conflicting voices, listening only to the speaker of interest? Such a device would not only have to be able separate voices from a mixture, but also do it in a semi-real time fashion in order for it to be useful in a restaurant setting. Let's look more at ways to accomplish that first requirement.

## Blind Source Separation

The traditional way to separate voices from a mixture is Blind Source Separation (BSS). "Blind" refers to the fact that the process by which the voices were mixed is unknown. BSS algorithms assume properties of the signal sources and the mixing processes and use those assumptions to try to reconstruct the original audio.

One such algorithm is **Independent Component Analysis** which requires that there are at least as many microphones as there are voices in the mixture, and relies on the assumption that the signals are non-Gaussian and independent, which are not necessarily always true.

In addition, the necessity for multiple microphones makes this algorithm difficult to deploy in practice.
Watch-- er, listen to what happens when you use ICA with only a single microphone:

Mixture (i.e. the overlapping voices)
<audio controls>
<source src="/assets/audio/ICA/mix.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Recovered Sources
<audio controls>
<source src="/assets/audio/ICA/recon_source_1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="/assets/audio/ICA/recon_source_2.wav" type="audio/wav">Your browser does not support the audio element.</audio><br/>

Evidently, not much unmixing was done.

## Neural Networks for Audio Separation

Artificial Neural Networks, or also referred to as neural networks, have proven to be very useful in a wide variety of tasks, including source separation. Neural networks, using large amounts of training data, can capture complex relationships that can be used for inference. In the case of source separation, a neural network can characterize how much of each audio slice belongs to each speaker.

Neural networks are not limited in the same way that BSS methods like ICA are - so long as the training data are representative of the testing data, there are fewer limitations on the properties of the original sources or the mixture.

> ### Aside: Spectrograms
> A commonly used tool in the field of audio processing is the **spectrogram**, which is a 2D representation of an audio signal, generated using a Short Time Fourier Transform (STFT) with frequencies on one axis and time on another. The intensity of each 'pixel' represents the intensity of a frequency at any given time. Conventional wisdom was always that spectrograms are _vital_ tools for source separation, as intuitively, separating the frequencies should assist with the separation.
>
> However, in recent literature, it was found that neural networks could achieve very accurate results without performing the time-consuming STFT operation and its inverse.

Specifically, we are using the Conv-TasNet[^2] architecture, which is a convolutional network that operates solely in the time domain (see the appendix for more details), but is still able to produce relatively accurate results:

Raw Audio:

<audio controls>
<source src="/assets/audio/sx98_raw.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="/assets/audio/sable_raw.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Mixed Audio:

<audio controls>
<source src="/assets/audio/mixed-sable.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Unmixed Audio:

<audio controls>
<source src="/assets/audio/mixed-sable_est1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="/assets/audio/mixed-sable_est2.wav" type="audio/wav">Your browser does not support the audio element.</audio><br/>

The unmixed audio sounds almost perfectly like the original, save for some small artifacts in the left estimation when the speaker says the word "question" (headphones make it easier to hear this artifact). These high accuracy estimations show promise for creating a source separation system.

### Real-Time Considerations

In most fields, real-time performance is very difficult, as each part of the pipeline must be optimized in order to minimize latency. In the case of source separation, audio collection from the input microphone must be parallelized with the neural network that is processing the data. In addition, the data must be divided into chunks which are passed through the neural network. However, if the neural network cannot process the input audio faster than it is coming in, the latency will still accumulate. In essence, this becomes a [producer-consumer problem](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem).

Some options for decreasing latency include: multithreading the python code, so audio can be recorded while the neural network performs computations, truncating the floating point precision, and using a faster language, such as C++. We are currently in the process of experimenting with these optimizations.

### Results and Next Steps

As of now, we have implemented a multithreaded python program which is able to chunk the data and process it through the neural network; however, the program has a reconstruction error that results in choppy sounding audio. The effects of this reconstruction error can be mitigated by increasing the chunk size, but that in turn increases latency. 

> To Be populated in with audio files

On the one hand, we plan on investigating traditional reconstruction techniques to see if we can mitigate the choppiness. On the other, we also plan on training a neural network on data of the same length as our chosen chunk size, so that the training data is representative of our testing conditions.

In order to decrease latency, we are looking into the [ONNX standard](https://github.com/onnx/onnx) and [TensorRT](https://developer.nvidia.com/tensorrt), which should better optimize the neural network for fast compuation.
 
[^1]: https://www.ee.columbia.edu/~dpwe/papers/Cherry53-cpe.pdf

[^2]: https://arxiv.org/pdf/1809.07454.pdf
