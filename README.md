# PSL — Pakistani Sign Language Translator

Real-time translation of **Pakistani Sign Language (PSL)** into English sentences.

Built by **Abdul Rehman**.

---

> 🚧 **Coming soon**
>
> The model is being retrained with improvements to the training pipeline.
> The code, the trained model and setup instructions will be published here
> once the new results are verified.

---

## What it does

```
video ──► hand & body landmarks ──► sign classifier ──► sign words ──► LLM ──► English sentences
```

- **Landmark extraction** with MediaPipe Holistic (hands + body pose)
- **Sign recognition** with a Temporal Convolutional Network (TCN), 24 PSL signs
- **Continuous signing** through a sliding window over live video
- **Natural sentences**: detected signs are turned into fluent English by an LLM
- **REST API** for video upload, used by a companion Flutter mobile app

## Vocabulary (first release)

I, come, eat, friend, go, good, help, home, learn, like, morning, no, phone,
please, salam, school, sorry, student, teacher, thank you, want, water, you

## What's in progress

- [ ] Fixes to data augmentation
- [ ] Body-relative landmark normalisation
- [ ] Leave-one-signer-out evaluation for reliable accuracy numbers
- [ ] Retrained model and published results
- [ ] Flutter mobile app
- [ ] Documentation

⭐ Star or watch the repo to get notified when the first release is out.
