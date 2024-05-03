<p align="center"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/7520973/249193988-89f07b66-f8b9-4df7-97ba-500f0cbebaab.png"></p>

_TrustML_ is a modular and extensible package to support the definition, assessment and monitoring of custom-built trustworthiness indicators for AI models. _TrustML_ allows data scientists to define trustworthiness indicators by selecting a set of metrics from a catalog of trustworthy-related metrics and grouping them into higher-level metric aggregations.

_TrustML_ also provides different assessment methods to compute and monitor the indicators previously defined. _TrustML_ enables and supports the development of trustworthy AI models, aiming to provide assistance not only during their construction phase, but also in production environments, as a mechanism to continuously monitor such trust and enable mitigation activities when required.

The package makes use of existing packages meant to compute each of the included trustworthiness metrics, check the requirements.txt file in the root of the source code repository for details.

The API documentation is available on https://martimanzano.github.io/TrustML/.
The wiki with tutorials on the package's usage and extension is available on https://github.com/martimanzano/TrustML/wiki/Home/.

The _TrustML_ package is free software distributed under the Apache License 2.0. If you are interested in participating in this project, please use the [GitHub repository](https://github.com/martimanzano/TrustML); and review the [Contributing page](https://github.com/martimanzano/trustML/blob/modular/CONTRIBUTING.md), the [Code of Conduct](https://github.com/martimanzano/trustML/blob/modular/CODE_OF_CONDUCT.md) and the specific [Contributing articles](https://github.com/martimanzano/trustML/tree/modular/contributing) all contributions are welcomed.

If you want to citate _TrustML_, please cite the following paper:

TrustML: A Python package for computing the trustworthiness of ML models
Martí Manzano, Claudia Ayala, Cristina Gómez, SoftwareX,
Volume 26,
2024,
101740,
ISSN 2352-7110,
https://doi.org/10.1016/j.softx.2024.101740.

Bibtex entry:
```
@article{MANZANO2024101740,
  title = {TrustML: A Python package for computing the trustworthiness of ML models},
  journal = {SoftwareX},
  volume = {26},
  pages = {101740},
  year = {2024},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2024.101740},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711024001110},
  author = {Martí Manzano and Claudia Ayala and Cristina Gómez}
}
```
