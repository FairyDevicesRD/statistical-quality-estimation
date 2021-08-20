This is an implementation of the paper [Quality estimation for partially subjective classification tasks via crowdsourcing](https://aclanthology.org/2020.lrec-1.29/).

# Install
```
pip install -r requirements.txt

# apply the patch
cd dirichlet
patch -p1 < ../dirichet.patch
```

# Configuration
Create a configuration file in the TOML format using `config.toml` as a reference.

# Input data
All input files should be placed under the directory specified by `base_dir` in the configuration file.
The following files are required:
- `creation.tsv`
- `review.tsv`

The following file is optional:
- `qualification_review.tsv`

`creation.tsv` is a TSV file that provides data of the creation stage, specifically `creator_id`, `class_id`, and `artifact_id`.
`class_id` represents the class ID instructed to the creator.
The first line should be a header.

`review.tsv` is a TSV file that provides data of the evaluation stage, specifically `artifact_id`, `reviewer_id`, and `class_id`.
`class_id` represents the class ID the reviewer classified the artifact.
The first line should be a header.

`qulification_review.tsv` is a TSV file that provides data of the qualification round to select reviewers.
The columns are `reviewer_id`, `class_id_correct`, and `class_id_reviewer`, where `class_id_correct` and `class_is_reviewer` are the class ID that is supposed to be correct and the reviewer answered.

Note that all IDs should be zero-based sequential indexing.

# Usage
```
usage: optimize.py [-h] [--config-path CONFIG_PATH] [-v]

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        configuration file in the TOML format
  -v, --verbose         verbose mode
```

# Output data
Onput files are created under `${base_dir}/${exp_id}`, where `base_dir` and `exp_dir` are specified in the configuration file.
For each iteration, `${iteration}.npz` is created in an uncompressed `.npz` format of NumPy.
This file includes the estimated values of `q`, `alpha`, and `beta`.

# License
This code is available under the MIT License. See `LICENSE` for details.

`dirichlet.path` is a patch to [Dirichlet](https://github.com/ericsuh/dirichlet), which is published under [this](https://github.com/ericsuh/dirichlet/blob/master/LICENSE.txt) license.

# Citation
```
@inproceedings{sato-miyazawa-2020-quality,
    title = "Quality Estimation for Partially Subjective Classification Tasks via Crowdsourcing",
    author = "Sato, Yoshinao and Miyazawa, Kouki",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.29",
    pages = "229--235",
    language = "English",
    ISBN = "979-10-95546-34-4",
 }
```
