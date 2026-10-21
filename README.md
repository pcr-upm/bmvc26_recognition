# On Structured Disentangled Representation and the Limits of Global Scores

If you use this code for your own research, you must reference our conference paper:

```
On Structured Disentangled Representation and the Limits of Global Scores
Jorge Elías, Roberto Valle, José M. Buenaposada, Luis Baumela.
37th British Machine Vision Conference, BMVC 2026, Lancaster, UK, November 23-26.
```

#### Requisites
The required dependencies are installed in the [`Dockerfile`](./Dockerfile#L36).

#### Usage
```
usage: bmvc26_recognition_latent.py [-h] [--show-viewer]
```

* Use the --show-viewer option to show results visually.
```
usage: Recognition --database DATABASE
```

* Use the --database option to select the database model.
```
usage: BMVC26Recognition [--nlats NLATS] [--ndims NDIMS]
```

* Use the --nlats option to set the number of latent units.

* Use the --ndims option to set the dimensionality of the latent space.
```
> python test/bmvc26_recognition_latent.py --database shapes3d --nlats 6 --ndims 4
```

#### Acknowledgements
The training of each model builds on the official implementations of the diffusion-based disentanglement methods evaluated in the paper:

- DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models
  Official repository: https://github.com/ThomasMrY/DisDiff

- Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models
  Official repository: https://github.com/youngjun-jun/dis-dis

We thank the authors for releasing the code.