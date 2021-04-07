---
layout: default
---
<img class="top-image" src="assets/images/cocktail.webp"> 

Imagine walking into a crowded, noisy restaurant - 

Well, these are still quarantine times, so this type of scenario is less common, but presumably, when life returns to normal, such a scenario could occur.

You sit across the table from your friend. It's noisy, with many other conversations going on around you, making it difficult to hear what she is saying.

## Cocktail Party Problem

The Cocktail Party Effect is the "ability for people to focus their auditory attention on one source,"[^1] whether that be a friend at a party, or a waiter in a restaurant. Humans, in general, are able to listen to mixed audio from many sources, and hone their focus on individual audio sources. 

{% include image.html url="assets/images/shadowing.png" description="Shadowing, a dichotic listening experiment, presents a participant with two different voice messages, and asks the participant to focus on one of the messages and repeat it aloud as a way of measuring perceptual ability." %}

However, this ability is not perfect: a listener may not necessarily pick up words and sounds only from the speaker of interest. The ability to "tune in" to a single voice is highly dependent on a number of features, including speaker pitch, location, rate of speech, and the listener's hearing capability. If a person only has one functional ear, or is hard of hearing, focusing on a single voice can be very difficult: with only one ear, it is difficult to determine locality of the speaker, and when hard of hearing, all sounds come through with limited fidelity.

So what if there was a way that we could make a device that could 'tune out' conflicting voices, listening only to the speaker of interest? Such a device would not only have to separate voices from a mixture, but also do it in a semi-real time fashion in order for it to be useful for day-to-day usage.

Let's start by looking into ways to accomplish voice separation.

## Separating Voice Mixtures

### Blind Source Separation

The traditional way to separate voices from a mixture is Blind Source Separation (BSS). "Blind" refers to the fact that the process by which the voices were mixed is unknown. BSS algorithms assume properties of the signal sources and the mixing processes, and then they use those assumptions to try to reconstruct the original audio.

One such algorithm is **Independent Component Analysis**, which requires that there are at least as many microphones as there are voices in the mixture, and relies on the assumption that the signals are non-Gaussian and independent.

The necessity for multiple microphones makes this algorithm particularly tricky, as the location of the mics and the hardware they are connected to can introduce phase delays that prevent the final audio samples from aligning properly.

Let us try an example of ICA to demonstrate this.

Microphones 1 and 2:
<audio controls>
<source src="assets/audio/ICA/mic_1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="assets/audio/ICA/mic_2.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Recovered Sources:
<audio controls>
<source src="assets/audio/ICA/recon_source_1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="assets/audio/ICA/recon_source_2.wav" type="audio/wav">Your browser does not support the audio element.</audio><br/>

Evidently, not much unmixing was done.

### Neural Networks for Audio Separation

Artificial Neural Networks, or also referred to as neural networks, have proven to be very useful in a wide variety of tasks, including source separation. Neural networks, using large amounts of training data, can capture complex relationships that can be used for inference. In the case of source separation, a neural network can characterize how much of each audio slice belongs to each speaker.

Neural networks are not limited in the same way that BSS methods like ICA are: so long as the training data are representative of the testing data, there are fewer limitations on the properties of the original sources or the mixture.

>### _Aside: Spectrograms_
><img class="top-image" src="assets/images/spec.png">
>
>A commonly used tool in the field of audio processing is the **spectrogram**, which is a 2D representation of an audio signal, generated using a Short Time Fourier Transform (STFT) with frequencies on one axis and time on the other. The color of each 'pixel' represents the intensity of a frequency at any given time. This transform allows us to see the individual frequency components of an sound clip. Conventional wisdom always said that spectrograms are _vital_ tools for source separation, as intuitively, separating the frequencies should assist with the separation process.
>
> However, in recent literature, it was found that neural networks could achieve very accurate results without performing the time-consuming STFT operation and its inverse.


We are using the _Convolutional Time Domain Audio Separation Network_ (Conv-TasNet)[^2] architecture, which is a network that operates solely in the time domain (see the appendix for more details), but is still able to produce relatively accurate results. We chose this network because we found it to be a fast neural network archictecture that still produced good results: lower latency of this network makes it more amenable to real time separation, even if the network does not achieve state of the art results.

Let's look at an example of a Conv-TasNet separating audio.

First, the raw audio:

<audio controls>
<source src="assets/audio/conv-tasnet/adam-savage.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="assets/audio/conv-tasnet/sable.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Next, we mix the two audio sources together:

<audio controls>
<source src="assets/audio/conv-tasnet/mix.wav" type="audio/wav">Your browser does not support the audio element.</audio>

Finally, we can run the audio through the network and generate two mixtures:

<audio controls>
<source src="assets/audio/conv-tasnet/est1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="assets/audio/conv-tasnet/est2.wav" type="audio/wav">Your browser does not support the audio element.</audio><br/>

The unmixed audio sounds relatively close to the original, but there are some artifacts still present in the beginning of the estimation on the right. However, the relatively clean results show that this network is a viable choice for source separation.

## Hardware and Real Time Considerations

### Hardware

A large focus in using deep learning for audio source separation lies in achieving the highest accuracy reconstructions. Separation is performed on machines with very powerful graphics cards, and the audio reconstructions are computed very quickly. However, in the field, it is not reasonable to expect that such a computer would be readily available.

The trade-off then becomes one of latency and computational speed: a more powerful computer is faster, but is impractical for deployment; a less powerful computer is slower, but more faithful to what would be available in the field.

<img class="med-image" src="assets/images/jetson.jpg">

We chose to focus on deploying a source separation pipeline to the [Nvidia Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit), a single board computer which offers a compromise between computational power and portability. The Jetson Nano is similar to other single board computers like the [Raspberry Pi](https://www.raspberrypi.org/), except it comes with a more powerful on-board GPU that gives it an advantage when performing deep learning computations.

### Real Time Considerations

In many fields, real time performance is a very difficult task, requiring each part of the input pipeline to be optimized to minimize latency. In the case of Conv-TasNet, the network relies on contextual data to perform separation, meaning it must operate on chunks of audio data to operate in real time. This means that even if the network performed inference instantaneously, there would still be a time delay of the chunk size at the output. Moreover, if the neural network cannot process the input audio faster than it is coming in, the latency will accumulate for every chunk and result in completely non-synced output. In essence, this becomes a [producer-consumer problem](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem).

Some options for alleviating these issues include:

 1. Using multiple processor cores to parallelize audio collection and neural network computation. This optimization is almost essential to prevent gaps in the recorded audio.
 2. Using optimization frameworks such as [TensorRT](https://developer.nvidia.com/tensorrt) to reduce model complexity and reduce the precision of the network weights.
 3. Using a language faster than python, such as C++.

### Results and Next Steps

As of now, we have implemented a multithreaded python program which is able to chunk the data and process it through the neural network. We leverage multiple cores to parallelize audio collection and separation.

It is somewhat difficult to show latency in a demo recording, but on a desktop machine with a dedicated GPU, we are able to reduce the latency down to roughly 700 ms, which is low, but still a very noticeable latency. On the Jetson, we are still experiencing nearly 60 second delay. In addition, the network is unable to process the data faster than it is coming in, resulting in a filling of the input queue, which leads to some dropped audio segments. Our next course of action would be to reduce model complexity to improve separation time (bullet point 2 above).

Let's hear a sample of a reconstruction performed on the Jetson:

<audio controls>
<source src="assets/audio/jetson/est1.wav" type="audio/wav">Your browser does not support the audio element.</audio>

<audio controls>
<source src="assets/audio/jetson/est2.wav" type="audio/wav">Your browser does not support the audio element.</audio><br/>

One issue with the reconstructions is the random assignment of separated audio. The current model we are using was trained to pick the output configuration that yields the optimal results (Permutation Invariant Training), which is fine for normal separation where the model can see the entire audio recording at once. When processing with chunks, however, the model cannot see future or past chunks, so it essentially randomly assigns the separations to one of the reconstructions. Thus, even if the audio separation is working, the reconstructed audio is still very difficult to understand. One method of solving this would be to train the model without the permutation invariance, which is where we are currently investigating.
 
## Appendix: Conv-TasNet Architecture

Conv-TasNet has three stages: an encoder, a "separation module" which forms the masks, and a decoder. For starters, the encoder convolves the mixture waveform with a number of filters (512 in the original authors' implementation).

Moving onto the separation module, we note that it is the \\(T \times \text{enc\_dim}\\) encoding that is being masked, not the \\(T \times 1\\) signal, making it somewhat unconventional to call these masks "masks". As for how the masks are created, depthwise separable convolutions are used to form one \\(T \times \text{enc\_dim}\\) mask for each source, and the convolutions use exponentially increasing dilation factors to detect both short-term and long-term dependencies.

Lastly, the decoder performs a transposed convolution on each masked encoding, resulting in the separated sources, each with the same shape as the mixture waveform.

As for training, Conv-TasNet's objective is to maximize the scale-invariant signal-to-distortion ratio for each training example.

[^1]: [Edward Colin Cherry's Experiments](https://www.ee.columbia.edu/~dpwe/papers/Cherry53-cpe.pdf)

[^2]: [Conv-TasNet by Luo et al.](https://arxiv.org/pdf/1809.07454.pdf)
