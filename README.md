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
usage: bmvc26_recognition_latent.py [-h] [--database DATABASE] [--gpu GPU] [--nlats NLATS] [--ndims NDIMS] [--slot SLOT]
```

* Use `--database` to select the database model. Supported models include `shapes3d` and `celeba`.
* Use `--gpu` to set the GPU identifier (negative value indicates CPU mode).
* Use `--nlats` to set the number of latent units (default: `10`).
* Use `--ndims` to set the dimensionality of each latent unit (default: `32`).
* Use `--slot` to specify a latent unit whose value is kept fixed during conditional generation. The option can be repeated (default: `--slot 1 --slot 2`).
```
usage: bmvc26_recognition_generation.py [-h] [--input-data INPUT_DATA] [--database DATABASE] [--gpu GPU] [--nlats NLATS] [--ndims NDIMS] [--slot SLOT] [--show-viewer] [--save-image] 
```

* Use `--input-data` to select the input latent identifier (default: `176977`).
* The `--database`, `--gpu`, `--nlats`, `--ndims`, and `--slot` options have the same meaning described above.
* Use `--show-viewer` to show the results visually.
* Use `--save-image` to save the processed images.

Latent-unit probing example:
```
> python test/bmvc26_recognition_latent.py --database shapes3d --gpu 0 --nlats 6 --ndims 4
```

Conditional-generation example:
```
> python test/bmvc26_recognition_generation.py --database shapes3d --gpu 0 --nlats 6 --ndims 4 --slot 1 --slot 2 --save-image
```

#### Acknowledgements
The training of each model builds on the official implementations of the diffusion-based disentanglement methods evaluated in the paper:

- DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models
  Official repository: https://github.com/ThomasMrY/DisDiff

- Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models
  Official repository: https://github.com/youngjun-jun/dis-dis

We thank the authors for releasing the code.